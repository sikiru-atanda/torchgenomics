"""Per-SNP LD score computation for LDSC regression.

LD score for SNP j is the sum of r-squared values between j and all SNPs k
within a specified genomic window on the same chromosome:

    l_j = sum_{k in window} r^2(j, k)

This includes r^2(j, j) = 1, so independent SNPs have l_j = 1.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from ..ld._pairwise import compute_r2_matrix


def compute_ld_scores(
    G: Tensor,
    pos: list[int] | Tensor,
    chr_labels: list[str] | list[int],
    window_kb: float = 1000.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute per-SNP LD scores within a sliding genomic window.

    For each SNP j, the LD score l_j = sum of r^2(j, k) for all k on the
    same chromosome within ``window_kb`` kilobases.

    Parameters
    ----------
    G : (n, m) Tensor
        Genotype dosage matrix.
    pos : list[int] or Tensor
        Physical positions in base pairs for each of m SNPs.
    chr_labels : list[str] or list[int]
        Chromosome label for each SNP.
    window_kb : float
        One-sided window size in kilobases (default 1000 = 1 Mb).
    device : torch.device or str
        Device for computation.

    Returns
    -------
    ld_scores : (m,) float64
        Per-SNP LD scores.
    """
    G = G.to(torch.float64).to(device)
    m = G.shape[1]

    if isinstance(pos, Tensor):
        pos_arr = pos.cpu().tolist()
    else:
        pos_arr = list(pos)

    chr_arr = [str(c) for c in chr_labels]
    if len(pos_arr) != m:
        raise ValueError(
            f"pos length ({len(pos_arr)}) must equal the number of SNPs in G ({m})"
        )
    if len(chr_arr) != m:
        raise ValueError(
            f"chr_labels length ({len(chr_arr)}) must equal the number of SNPs in G ({m})"
        )
    window_bp = window_kb * 1000.0

    ld_scores = torch.ones(m, dtype=torch.float64, device=device)

    # Group SNPs by chromosome
    chr_to_indices: dict[str, list[int]] = {}
    for i, c in enumerate(chr_arr):
        chr_to_indices.setdefault(c, []).append(i)

    for chrom, snp_indices in chr_to_indices.items():
        if len(snp_indices) <= 1:
            continue  # l_j = 1 (self only)

        idx = torch.tensor(snp_indices, dtype=torch.long)
        G_chr = G[:, idx]  # (n, m_chr)
        pos_chr = [pos_arr[i] for i in snp_indices]

        # Compute full r² matrix for this chromosome
        r2_chr = compute_r2_matrix(G_chr)  # (m_chr, m_chr)

        # Apply window mask
        m_chr = len(snp_indices)
        pos_t = torch.tensor(pos_chr, dtype=torch.float64, device=device)
        # (m_chr, m_chr) distance matrix
        dist = (pos_t.unsqueeze(1) - pos_t.unsqueeze(0)).abs()
        window_mask = dist <= window_bp  # includes self (dist=0)

        # Sum r² within window for each SNP
        r2_windowed = r2_chr.to(device) * window_mask.to(torch.float64)
        scores_chr = r2_windowed.sum(dim=1)  # (m_chr,)

        ld_scores[idx] = scores_chr

    return ld_scores


def compute_cross_ld_scores(
    G: Tensor,
    pos: list[int] | Tensor,
    chr_labels: list[str] | list[int],
    window_kb: float = 1000.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute cross-trait LD scores for genetic correlation.

    For LDSC rg, the cross-trait LD score uses the same formula as
    univariate LD scores since both traits share the same LD structure.
    This is a convenience alias that returns the same result as
    ``compute_ld_scores`` but makes the intent explicit.

    Parameters
    ----------
    G, pos, chr_labels, window_kb, device
        Same as :func:`compute_ld_scores`.

    Returns
    -------
    ld_scores : (m,) float64
    """
    return compute_ld_scores(G, pos, chr_labels, window_kb, device)


# ── Streaming variant ────────────────────────────────────────────────


def compute_ld_scores_streaming(
    chunk_iter: Iterable[tuple[Tensor, object]],
    *,
    window_kb: float = 1000.0,
    device: torch.device | str = "cpu",
    impute: bool = True,
) -> tuple[Tensor, list[str], list[int]]:
    """Streaming variant of :func:`compute_ld_scores`.

    Same per-SNP LD score formula as :func:`compute_ld_scores` but uses a
    sliding-window buffer over a chunk iterator. The buffer holds, at all
    times, every SNP that could still gain r² contribution from a future
    SNP — i.e. every SNP whose right-edge ``pos + window_bp`` has not yet
    been crossed by the latest streamed position. Once the latest position
    moves past a focal SNP's right edge, that SNP's LD score is final.

    Peak memory is ``O(n × window_size × 8 B)`` where ``window_size`` is
    the number of SNPs in the densest 2×window_kb stretch — typically a
    few hundred to a few thousand SNPs at biobank scale.

    Parameters
    ----------
    chunk_iter : iterable of ``(G_chunk, vmeta)``
        Chunks delivered in genomic order. ``vmeta.pos`` and ``vmeta.chr``
        provide per-variant position / chromosome.
    window_kb : float
        One-sided window size in kilobases (default 1000 = 1 Mb).
    device : torch.device or str
        Device for r²/buffer computation.
    impute : bool
        If True (default), mean-impute each chunk on entry to the buffer
        so NaNs from missing genotypes do not corrupt running r².

    Returns
    -------
    ld_scores : (m,) float64
        Per-SNP LD scores in the order SNPs appeared in the chunk iterator.
    chr_out : list[str]
        Per-SNP chromosome label, parallel to ``ld_scores``.
    pos_out : list[int]
        Per-SNP physical position, parallel to ``ld_scores``.
    """
    from ..preprocess.impute import impute_mean

    window_bp = float(window_kb) * 1000.0

    # Per-chromosome streaming buffer. ``buf_G`` holds n × buf_m float64
    # genotypes; ``buf_pos`` parallel positions; ``buf_score`` running
    # partial-sum r² for each buffered SNP (against the SNPs already
    # encountered in its left window). When a SNP's right edge is crossed,
    # we move it from the buffer into the output and reset its score.
    buf_G: torch.Tensor | None = None
    buf_pos: list[int] = []
    buf_score: list[float] = []
    buf_chr: str | None = None
    out_pos_in_chr: list[int] = []  # per-SNP output positions for ordering

    out_scores: list[float] = []
    out_chr: list[str] = []
    out_pos: list[int] = []

    def _finalize_left(head_pos: int | None) -> None:
        """Move all buffered SNPs whose right edge is < ``head_pos`` to
        the output list. If ``head_pos`` is None, drain everything (used
        on chromosome boundary / end of stream).
        """
        nonlocal buf_G, buf_pos, buf_score
        if buf_G is None or buf_G.shape[1] == 0:
            return
        n_buf = buf_G.shape[1]
        n_drop = 0
        while n_drop < n_buf:
            if head_pos is not None and (head_pos - buf_pos[n_drop]) <= window_bp:
                break
            n_drop += 1
        if n_drop == 0:
            return

        for k in range(n_drop):
            out_scores.append(buf_score[k])
            out_chr.append(str(buf_chr))
            out_pos.append(buf_pos[k])
            out_pos_in_chr.append(buf_pos[k])

        if n_drop == n_buf:
            buf_G = None
            buf_pos = []
            buf_score = []
        else:
            buf_G = buf_G[:, n_drop:].contiguous()
            buf_pos = buf_pos[n_drop:]
            buf_score = buf_score[n_drop:]

    def _flush_chrom() -> None:
        """Drain everything in the buffer (called on chrom transition / EOS)."""
        nonlocal buf_G, buf_pos, buf_score, buf_chr
        _finalize_left(None)
        buf_G = None
        buf_pos = []
        buf_score = []
        buf_chr = None

    def _push(col: torch.Tensor, pos_j: int) -> None:
        """Append SNP ``col`` at position ``pos_j`` to the buffer.

        Updates running r² partial sums:
        - For each previously buffered SNP k within window of pos_j,
          add r²(k, j) to buf_score[k].
        - The new SNP's score starts at 1.0 (self r² contribution) plus
          all r²(j, k) for k in buffer within window.
        """
        nonlocal buf_G, buf_pos, buf_score
        if buf_G is None or buf_G.shape[1] == 0:
            buf_G = col.clone()
            buf_pos = [pos_j]
            buf_score = [1.0]  # self r² = 1
            return

        # Compute r²(col, every buffered column) in one vectorized pass.
        # compute_r2_pairs would loop; we use the dense pair formula.
        gj = col.squeeze(1)  # (n,)
        gj_c = gj - gj.mean()
        sj = gj_c.std().clamp(min=1e-10)

        gi = buf_G  # (n, buf_m)
        gi_c = gi - gi.mean(dim=0, keepdim=True)
        si = gi_c.std(dim=0).clamp(min=1e-10)

        n = gi.shape[0]
        cov = (gi_c * gj_c.unsqueeze(1)).sum(dim=0) / max(n - 1, 1)
        r = (cov / (si * sj)).clamp(-1.0, 1.0)
        r2 = (r * r).cpu().tolist()

        # Window mask in Python — buf_pos is a small list.
        score_j = 1.0
        for k in range(len(buf_pos)):
            if abs(buf_pos[k] - pos_j) <= window_bp:
                buf_score[k] += r2[k]
                score_j += r2[k]

        buf_G = torch.cat([buf_G, col], dim=1)
        buf_pos.append(pos_j)
        buf_score.append(score_j)

    for G_chunk, vmeta in chunk_iter:
        if impute:
            G_chunk = impute_mean(G_chunk)
        G_chunk = G_chunk.to(device=device, dtype=torch.float64)
        chr_arr = [str(c) for c in vmeta.chr]
        pos_arr = [int(p) for p in vmeta.pos]

        for j in range(G_chunk.shape[1]):
            chrom_j = chr_arr[j]
            pos_j = pos_arr[j]

            if buf_chr is None:
                buf_chr = chrom_j

            if chrom_j != buf_chr:
                _flush_chrom()
                buf_chr = chrom_j

            # Drain SNPs whose right edge is now strictly behind pos_j.
            _finalize_left(pos_j)

            _push(G_chunk[:, j : j + 1], pos_j)

    _flush_chrom()

    if not out_scores:
        return (
            torch.zeros(0, dtype=torch.float64, device=device),
            [],
            [],
        )
    return (
        torch.tensor(out_scores, dtype=torch.float64, device=device),
        out_chr,
        out_pos,
    )
