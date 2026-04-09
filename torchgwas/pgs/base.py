"""Base classes for the PGS module.

Provides:
    - LDReference: GPU LD reference panel (block-diagonal or full)
    - PGSResult: per-SNP weights with method metadata and diagnostics
    - BasePGSMethod: ABC for all PGS construction methods
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

from ..postgwas._sumstats import SumStats


@dataclass
class LDReference:
    """GPU LD reference panel for PGS methods.

    Either holds a full ``(m, m)`` LD correlation matrix (``mode="full"``)
    or a block-diagonal partition as a list of ``(b_k, b_k)`` tensors
    (``mode="block"``). SNPs are stored in genomic order; for block mode
    ``block_index[j]`` gives the 0-based block id of SNP ``j``.
    """

    snp: list[str]
    chr: list[str]
    pos: list[int]
    a1: list[str]
    a2: list[str]
    af: Tensor  # (m,)
    mode: Literal["full", "block"] = "block"
    R_full: Tensor | None = None  # (m, m) if mode == "full"
    R_blocks: list[Tensor] | None = None  # list of (b_k, b_k) if mode == "block"
    block_index: Tensor | None = None  # (m,) long
    n_ref: int = 0
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.mode == "full":
            if self.R_full is None:
                raise ValueError("LDReference mode='full' requires R_full.")
            if self.R_full.shape[0] != self.R_full.shape[1] != self.m:
                raise ValueError(
                    f"R_full shape {tuple(self.R_full.shape)} does not match m={self.m}."
                )
        elif self.mode == "block":
            if self.R_blocks is None or self.block_index is None:
                raise ValueError(
                    "LDReference mode='block' requires R_blocks and block_index."
                )
            if int(self.block_index.shape[0]) != self.m:
                raise ValueError(
                    f"block_index length {int(self.block_index.shape[0])} != m={self.m}."
                )
            n_blocks = len(self.R_blocks)
            if int(self.block_index.max().item()) >= n_blocks:
                raise ValueError(
                    "block_index references a block id beyond R_blocks length."
                )
        else:
            raise ValueError(f"Unknown LDReference mode: {self.mode!r}")

    @property
    def m(self) -> int:
        return len(self.snp)

    def to(self, device: torch.device | str) -> "LDReference":
        """Move all tensors to ``device`` and return a new LDReference."""
        new_R_full = self.R_full.to(device) if self.R_full is not None else None
        new_R_blocks = (
            [b.to(device) for b in self.R_blocks] if self.R_blocks is not None else None
        )
        new_block_index = (
            self.block_index.to(device) if self.block_index is not None else None
        )
        return LDReference(
            snp=list(self.snp),
            chr=list(self.chr),
            pos=list(self.pos),
            a1=list(self.a1),
            a2=list(self.a2),
            af=self.af.to(device),
            mode=self.mode,
            R_full=new_R_full,
            R_blocks=new_R_blocks,
            block_index=new_block_index,
            n_ref=self.n_ref,
            device=device,
            dtype=self.dtype,
        )

    def block_of(self, snp_idx: int) -> int:
        """Return the block id containing SNP index ``snp_idx``."""
        if self.mode == "full":
            return 0
        assert self.block_index is not None
        return int(self.block_index[snp_idx].item())

    def block_slices(self) -> list[Tensor]:
        """Return per-block index tensors (sorted SNP positions per block).

        For ``mode="full"`` returns a single slice covering all SNPs.
        """
        if self.mode == "full":
            return [torch.arange(self.m, device=self.af.device)]
        assert self.block_index is not None
        n_blocks = len(self.R_blocks) if self.R_blocks is not None else 0
        out: list[Tensor] = []
        for k in range(n_blocks):
            mask = self.block_index == k
            out.append(torch.nonzero(mask, as_tuple=False).flatten())
        return out

    def solve(self, v: Tensor, ridge: float = 0.0) -> Tensor:
        """Apply ``(R + ridge * I)^{-1} v`` block-aware.

        Parameters
        ----------
        v : Tensor of shape (m,) or (m, k)
            Right-hand side.
        ridge : float
            Tikhonov regularizer added to the diagonal of each block.

        Returns
        -------
        Tensor with the same shape as ``v``.
        """
        if v.shape[0] != self.m:
            raise ValueError(f"solve: v.shape[0]={v.shape[0]} != m={self.m}.")

        out = torch.zeros_like(v)

        if self.mode == "full":
            assert self.R_full is not None
            R = self.R_full
            if ridge != 0.0:
                R = R + ridge * torch.eye(self.m, dtype=R.dtype, device=R.device)
            return torch.linalg.solve(R, v)

        # block mode
        assert self.R_blocks is not None and self.block_index is not None
        for k, R_b in enumerate(self.R_blocks):
            idx = torch.nonzero(self.block_index == k, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            R_eff = R_b
            if ridge != 0.0:
                R_eff = R_b + ridge * torch.eye(
                    R_b.shape[0], dtype=R_b.dtype, device=R_b.device
                )
            v_b = v[idx]
            out[idx] = torch.linalg.solve(R_eff, v_b)
        return out


@dataclass
class PGSResult:
    """Per-SNP PGS weights with method metadata and diagnostics."""

    method: str
    snp: list[str]
    chr: list[str]
    pos: list[int]
    a1: list[str]
    a2: list[str]
    weight: Tensor  # (m,)
    weight_sd: Tensor | None = None  # (m,)
    pip: Tensor | None = None  # (m,)
    af: Tensor | None = None  # (m,)

    h2: float | None = None
    p_causal: float | None = None
    phi: float | None = None
    grid: dict[str, Any] | None = None
    best_grid_idx: int | None = None

    n_iter: int = 0
    n_burnin: int = 0
    elbo_trace: list[float] = field(default_factory=list)
    rhat: Tensor | None = None  # (m,)
    ess: Tensor | None = None  # (m,)
    converged: bool = False

    n_input: int = 0
    n_matched: int = 0
    n_flipped: int = 0
    n_ambiguous_removed: int = 0
    n_maf_mismatch: int = 0

    def __len__(self) -> int:
        return len(self.snp)

    @property
    def m(self) -> int:
        return len(self.snp)

    def to_dataframe(self):
        """Return a pandas DataFrame view of the per-SNP weights."""
        import pandas as pd

        cols: dict[str, Any] = {
            "chr": self.chr,
            "pos": self.pos,
            "snp": self.snp,
            "a1": self.a1,
            "a2": self.a2,
            "weight": self.weight.detach().cpu().numpy(),
        }
        if self.weight_sd is not None:
            cols["weight_sd"] = self.weight_sd.detach().cpu().numpy()
        if self.pip is not None:
            cols["pip"] = self.pip.detach().cpu().numpy()
        if self.af is not None:
            cols["af"] = self.af.detach().cpu().numpy()
        return pd.DataFrame(cols)

    def save(self, path: str) -> None:
        """Save weights to TSV with a JSON sidecar of diagnostics.

        Writes ``<path>`` (TSV with per-SNP columns) and ``<path>.json``
        with method metadata, hyperparameters, and convergence info.
        """
        df = self.to_dataframe()
        df.to_csv(path, sep="\t", index=False)

        sidecar = {
            "method": self.method,
            "h2": self.h2,
            "p_causal": self.p_causal,
            "phi": self.phi,
            "grid": self.grid,
            "best_grid_idx": self.best_grid_idx,
            "n_iter": self.n_iter,
            "n_burnin": self.n_burnin,
            "converged": self.converged,
            "n_input": self.n_input,
            "n_matched": self.n_matched,
            "n_flipped": self.n_flipped,
            "n_ambiguous_removed": self.n_ambiguous_removed,
            "n_maf_mismatch": self.n_maf_mismatch,
        }
        if self.rhat is not None:
            sidecar["rhat_max"] = float(self.rhat.max().item())
            sidecar["rhat_mean"] = float(self.rhat.mean().item())
        if self.ess is not None:
            sidecar["ess_min"] = float(self.ess.min().item())
            sidecar["ess_mean"] = float(self.ess.mean().item())

        with open(str(path) + ".json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PGSResult":
        """Load a PGSResult from TSV + JSON sidecar produced by ``save``."""
        import pandas as pd

        df = pd.read_csv(path, sep="\t")
        sidecar_path = str(path) + ".json"
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                meta = json.load(f)
        except FileNotFoundError:
            meta = {}

        weight_sd = (
            torch.tensor(df["weight_sd"].values, dtype=torch.float64)
            if "weight_sd" in df.columns
            else None
        )
        pip = (
            torch.tensor(df["pip"].values, dtype=torch.float64)
            if "pip" in df.columns
            else None
        )
        af = (
            torch.tensor(df["af"].values, dtype=torch.float64)
            if "af" in df.columns
            else None
        )

        return cls(
            method=str(meta.get("method", "unknown")),
            snp=df["snp"].astype(str).tolist(),
            chr=df["chr"].astype(str).tolist(),
            pos=df["pos"].astype(int).tolist(),
            a1=df["a1"].astype(str).tolist(),
            a2=df["a2"].astype(str).tolist(),
            weight=torch.tensor(df["weight"].values, dtype=torch.float64),
            weight_sd=weight_sd,
            pip=pip,
            af=af,
            h2=meta.get("h2"),
            p_causal=meta.get("p_causal"),
            phi=meta.get("phi"),
            grid=meta.get("grid"),
            best_grid_idx=meta.get("best_grid_idx"),
            n_iter=int(meta.get("n_iter", 0)),
            n_burnin=int(meta.get("n_burnin", 0)),
            converged=bool(meta.get("converged", False)),
            n_input=int(meta.get("n_input", 0)),
            n_matched=int(meta.get("n_matched", 0)),
            n_flipped=int(meta.get("n_flipped", 0)),
            n_ambiguous_removed=int(meta.get("n_ambiguous_removed", 0)),
            n_maf_mismatch=int(meta.get("n_maf_mismatch", 0)),
        )


# ----------------------------------------------------------------------
# Allele harmonization helpers
# ----------------------------------------------------------------------

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _is_palindromic(a1: str, a2: str) -> bool:
    """Return True iff (a1, a2) is an A/T or C/G strand-ambiguous pair."""
    return _COMPLEMENT.get(a1.upper()) == a2.upper()


def _alleles_match(
    ss_a1: str, ss_a2: str, ref_a1: str, ref_a2: str
) -> tuple[bool, int]:
    """Compare a SumStats allele pair to an LD-reference allele pair.

    Returns
    -------
    (matched, sign) :
        ``matched`` is True if the alleles refer to the same biallelic
        variant (possibly after a flip or strand complement). ``sign`` is
        +1 if the SumStats effect aligns to the reference's effect allele
        and -1 if it must be sign-flipped.
    """
    ss_a1 = ss_a1.upper()
    ss_a2 = ss_a2.upper()
    ref_a1 = ref_a1.upper()
    ref_a2 = ref_a2.upper()

    if ss_a1 == ref_a1 and ss_a2 == ref_a2:
        return True, +1
    if ss_a1 == ref_a2 and ss_a2 == ref_a1:
        return True, -1

    # Strand complement
    cs_a1 = _COMPLEMENT.get(ss_a1)
    cs_a2 = _COMPLEMENT.get(ss_a2)
    if cs_a1 is None or cs_a2 is None:
        return False, 0
    if cs_a1 == ref_a1 and cs_a2 == ref_a2:
        return True, +1
    if cs_a1 == ref_a2 and cs_a2 == ref_a1:
        return True, -1
    return False, 0


# ----------------------------------------------------------------------
# Abstract base method
# ----------------------------------------------------------------------


class BasePGSMethod(ABC):
    """Abstract base for all PGS construction methods.

    Subclasses implement :meth:`fit`. The base class provides
    :meth:`_harmonize` for shared SumStats <-> LD-reference alignment.
    """

    name: str = "base"

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed

    @abstractmethod
    def fit(
        self,
        sumstats: SumStats,
        ld_ref: LDReference,
        **kwargs: Any,
    ) -> PGSResult:
        """Construct PGS weights from sumstats given an LD reference."""

    def _harmonize(
        self,
        sumstats: SumStats,
        ld_ref: LDReference,
        maf_tol: float = 0.20,
        drop_ambiguous: bool = True,
        palindromic_maf_tol: float = 0.42,
    ) -> tuple[SumStats, LDReference, dict[str, int]]:
        """Align SumStats to an LD reference by SNP id and alleles.

        Returns harmonized (sumstats, ld_ref) restricted to the matched
        SNPs in LD-reference order, with sumstats betas / AFs sign-flipped
        where the effect allele was swapped, plus an audit dict with
        keys: ``n_input``, ``n_matched``, ``n_flipped``,
        ``n_ambiguous_removed``, ``n_maf_mismatch``.
        """
        n_input = sumstats.m
        ss_idx_by_snp = {s: i for i, s in enumerate(sumstats.snp)}

        keep_ref: list[int] = []
        keep_ss: list[int] = []
        signs: list[int] = []
        n_flipped = 0
        n_ambiguous_removed = 0
        n_maf_mismatch = 0

        ss_af = sumstats.af
        ref_af = ld_ref.af

        for j_ref, snp_id in enumerate(ld_ref.snp):
            j_ss = ss_idx_by_snp.get(snp_id)
            if j_ss is None:
                continue

            ref_a1 = ld_ref.a1[j_ref]
            ref_a2 = ld_ref.a2[j_ref]
            ss_a1 = sumstats.a1[j_ss]
            ss_a2 = sumstats.a2[j_ss]

            if drop_ambiguous and _is_palindromic(ss_a1, ss_a2):
                # Strand-ambiguous: drop unless MAF is far from 0.5 in both
                # sides and we can disambiguate via AF concordance.
                if ss_af is None or ref_af is None:
                    n_ambiguous_removed += 1
                    continue
                ss_maf = float(ss_af[j_ss].item())
                ref_maf = float(ref_af[j_ref].item())
                ss_maf_dist = abs(ss_maf - 0.5)
                ref_maf_dist = abs(ref_maf - 0.5)
                cutoff = 0.5 - palindromic_maf_tol
                if ss_maf_dist < cutoff or ref_maf_dist < cutoff:
                    n_ambiguous_removed += 1
                    continue
                # Disambiguate by AF: same side -> +1, opposite -> -1
                same_side = (ss_maf > 0.5) == (ref_maf > 0.5)
                sign = 1 if same_side else -1
                if sign == -1:
                    n_flipped += 1
                keep_ref.append(j_ref)
                keep_ss.append(j_ss)
                signs.append(sign)
                continue

            matched, sign = _alleles_match(ss_a1, ss_a2, ref_a1, ref_a2)
            if not matched:
                continue

            if ss_af is not None and ref_af is not None:
                ss_freq = float(ss_af[j_ss].item())
                ref_freq = float(ref_af[j_ref].item())
                aligned_ss_freq = ss_freq if sign == +1 else 1.0 - ss_freq
                if abs(aligned_ss_freq - ref_freq) > maf_tol:
                    n_maf_mismatch += 1
                    continue

            if sign == -1:
                n_flipped += 1
            keep_ref.append(j_ref)
            keep_ss.append(j_ss)
            signs.append(sign)

        if not keep_ref:
            raise ValueError(
                "No SNPs remained after harmonization between sumstats and LD reference."
            )

        ref_idx = torch.tensor(keep_ref, dtype=torch.long)
        ss_idx = torch.tensor(keep_ss, dtype=torch.long)
        signs_t = torch.tensor(signs, dtype=sumstats.beta.dtype)

        # Build harmonized SumStats (in LD-reference order, with flipped signs)
        new_beta = sumstats.beta[ss_idx].clone() * signs_t.to(sumstats.beta.device)
        new_se = sumstats.se[ss_idx].clone()
        new_p = sumstats.p[ss_idx].clone()
        new_n = sumstats.n[ss_idx].clone()
        new_af: Tensor | None = None
        if sumstats.af is not None:
            af_aligned = sumstats.af[ss_idx].clone()
            flip_mask = (signs_t < 0).to(af_aligned.device)
            af_aligned[flip_mask] = 1.0 - af_aligned[flip_mask]
            new_af = af_aligned

        new_ss = SumStats(
            chr=[ld_ref.chr[i] for i in keep_ref],
            pos=[ld_ref.pos[i] for i in keep_ref],
            snp=[ld_ref.snp[i] for i in keep_ref],
            a1=[ld_ref.a1[i] for i in keep_ref],
            a2=[ld_ref.a2[i] for i in keep_ref],
            beta=new_beta,
            se=new_se,
            p=new_p,
            n=new_n,
            af=new_af,
        )

        # Build harmonized LD reference subset
        new_ref = _ld_reference_subset(ld_ref, ref_idx)

        audit = {
            "n_input": int(n_input),
            "n_matched": int(len(keep_ref)),
            "n_flipped": int(n_flipped),
            "n_ambiguous_removed": int(n_ambiguous_removed),
            "n_maf_mismatch": int(n_maf_mismatch),
        }
        return new_ss, new_ref, audit


def _ld_reference_subset(ld_ref: LDReference, idx: Tensor) -> LDReference:
    """Return a new LDReference restricted to ``idx`` (long tensor of indices).

    For block mode, blocks are re-indexed contiguously and each retained
    block is sub-selected to the kept SNPs within it.
    """
    idx_list = idx.tolist()
    new_snp = [ld_ref.snp[i] for i in idx_list]
    new_chr = [ld_ref.chr[i] for i in idx_list]
    new_pos = [ld_ref.pos[i] for i in idx_list]
    new_a1 = [ld_ref.a1[i] for i in idx_list]
    new_a2 = [ld_ref.a2[i] for i in idx_list]
    new_af = ld_ref.af[idx]

    if ld_ref.mode == "full":
        assert ld_ref.R_full is not None
        new_R = ld_ref.R_full[idx][:, idx].clone()
        return LDReference(
            snp=new_snp,
            chr=new_chr,
            pos=new_pos,
            a1=new_a1,
            a2=new_a2,
            af=new_af,
            mode="full",
            R_full=new_R,
            n_ref=ld_ref.n_ref,
            device=ld_ref.device,
            dtype=ld_ref.dtype,
        )

    # block mode: restrict each block, drop empty ones, renumber
    assert ld_ref.R_blocks is not None and ld_ref.block_index is not None
    old_block_id = ld_ref.block_index[idx]  # (m_new,) old block ids
    unique_old = torch.unique(old_block_id, sorted=True)
    new_R_blocks: list[Tensor] = []
    new_block_index = torch.empty_like(old_block_id)
    for new_k, old_k in enumerate(unique_old.tolist()):
        old_R = ld_ref.R_blocks[old_k]
        # positions in new subset that came from this old block
        new_in_block = torch.nonzero(old_block_id == old_k, as_tuple=False).flatten()
        # within-old-block positions of those SNPs
        old_block_mask = ld_ref.block_index == old_k
        old_within = torch.nonzero(old_block_mask, as_tuple=False).flatten()
        # which old-within indices are kept?
        kept_old_pos = idx[new_in_block]
        within_idx = torch.tensor(
            [int((old_within == p).nonzero(as_tuple=False).item()) for p in kept_old_pos],
            dtype=torch.long,
            device=old_R.device,
        )
        sub_R = old_R[within_idx][:, within_idx].clone()
        new_R_blocks.append(sub_R)
        new_block_index[new_in_block] = new_k

    return LDReference(
        snp=new_snp,
        chr=new_chr,
        pos=new_pos,
        a1=new_a1,
        a2=new_a2,
        af=new_af,
        mode="block",
        R_blocks=new_R_blocks,
        block_index=new_block_index,
        n_ref=ld_ref.n_ref,
        device=ld_ref.device,
        dtype=ld_ref.dtype,
    )
