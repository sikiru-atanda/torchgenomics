"""Summary statistics I/O for post-GWAS analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SumStats:
    """Container for GWAS summary statistics.

    Stores per-SNP association results in a format suitable for
    post-GWAS analyses (LDSC, meta-analysis, clumping).
    """

    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    beta: Tensor  # (m,) or (m, d) effect sizes
    se: Tensor  # (m,) or (m, d) standard errors
    p: Tensor  # (m,) p-values
    n: Tensor  # (m,) per-SNP sample sizes
    af: Tensor | None = None  # (m,) allele frequencies

    @property
    def chi2(self) -> Tensor:
        """Chi-squared statistics: (beta / se)^2."""
        return (self.beta / self.se) ** 2

    @property
    def z(self) -> Tensor:
        """Z-scores: beta / se."""
        return self.beta / self.se

    @property
    def m(self) -> int:
        """Number of SNPs."""
        return self.beta.shape[0]

    def to(self, device: torch.device | str) -> SumStats:
        """Move tensors to device."""
        return SumStats(
            chr=self.chr,
            pos=self.pos,
            snp=self.snp,
            a1=self.a1,
            a2=self.a2,
            beta=self.beta.to(device),
            se=self.se.to(device),
            p=self.p.to(device),
            n=self.n.to(device),
            af=self.af.to(device) if self.af is not None else None,
        )


def load_sumstats(
    path: str,
    chr_col: str = "chr",
    pos_col: str = "pos",
    snp_col: str = "snp",
    a1_col: str = "a1",
    a2_col: str = "a2",
    beta_col: str = "beta",
    se_col: str = "se",
    p_col: str = "p",
    n_col: str | None = "n",
    af_col: str | None = "af",
    sep: str = "\t",
    device: torch.device | str = "cpu",
) -> SumStats:
    """Load summary statistics from a TSV/CSV file.

    Reads TorchGWAS .assoc.tsv output or any tabular summary stats file
    with configurable column names.

    Parameters
    ----------
    path : str
        Path to summary statistics file.
    chr_col, pos_col, snp_col, a1_col, a2_col : str
        Column names for chromosome, position, SNP ID, effect/other allele.
    beta_col, se_col, p_col : str
        Column names for effect size, standard error, p-value.
    n_col : str or None
        Column name for sample size. If None or missing, uses NaN.
    af_col : str or None
        Column name for allele frequency. If None or missing, stores None.
    sep : str
        Delimiter (default tab).
    device : torch.device or str
        Device for output tensors.

    Returns
    -------
    SumStats
        Loaded summary statistics.
    """
    import pandas as pd

    df = pd.read_csv(path, sep=sep)

    chr_list = df[chr_col].astype(str).tolist()
    pos_list = df[pos_col].astype(int).tolist()
    snp_list = df[snp_col].astype(str).tolist()
    a1_list = df[a1_col].astype(str).tolist()
    a2_list = df[a2_col].astype(str).tolist()

    beta = torch.tensor(df[beta_col].values, dtype=torch.float64, device=device)
    se = torch.tensor(df[se_col].values, dtype=torch.float64, device=device)
    p = torch.tensor(df[p_col].values, dtype=torch.float64, device=device)

    if n_col is not None and n_col in df.columns:
        n = torch.tensor(df[n_col].values, dtype=torch.float64, device=device)
    else:
        n = torch.full((len(df),), float("nan"), dtype=torch.float64, device=device)

    af: Tensor | None = None
    if af_col is not None and af_col in df.columns:
        af = torch.tensor(df[af_col].values, dtype=torch.float64, device=device)

    return SumStats(
        chr=chr_list,
        pos=pos_list,
        snp=snp_list,
        a1=a1_list,
        a2=a2_list,
        beta=beta,
        se=se,
        p=p,
        n=n,
        af=af,
    )


def align_sumstats(
    ss_list: list[SumStats],
    match_alleles: bool = True,
) -> list[SumStats]:
    """Align multiple SumStats to a common set of SNPs.

    Intersects SNPs across all inputs by SNP ID and optionally flips
    effect alleles for strand consistency.

    Parameters
    ----------
    ss_list : list[SumStats]
        Two or more SumStats to align.
    match_alleles : bool
        If True, flip beta signs when alleles are swapped between studies.

    Returns
    -------
    list[SumStats]
        Aligned SumStats with identical SNP order.
    """
    if len(ss_list) < 2:
        return ss_list

    # Find common SNPs (intersection by ID)
    common = set(ss_list[0].snp)
    for ss in ss_list[1:]:
        common &= set(ss.snp)

    if len(common) == 0:
        raise ValueError("No common SNPs found across summary statistics.")

    # Use first study's order as reference
    ref = ss_list[0]
    ref_order = [s for s in ref.snp if s in common]
    ref_snp_to_idx = {s: i for i, s in enumerate(ref.snp)}

    result = []
    for k, ss in enumerate(ss_list):
        snp_to_idx = {s: i for i, s in enumerate(ss.snp)}
        indices = [snp_to_idx[s] for s in ref_order]
        idx_t = torch.tensor(indices, dtype=torch.long)

        # Build allele lookup for flipping
        ref_indices = [ref_snp_to_idx[s] for s in ref_order]

        new_beta = ss.beta[idx_t].clone()
        new_se = ss.se[idx_t].clone()
        new_p = ss.p[idx_t]
        new_n = ss.n[idx_t]
        new_af = ss.af[idx_t] if ss.af is not None else None

        chr_list = [ss.chr[i] for i in indices]
        pos_list = [ss.pos[i] for i in indices]
        snp_list = [ss.snp[i] for i in indices]
        a1_list = [ss.a1[i] for i in indices]
        a2_list = [ss.a2[i] for i in indices]

        # Flip effects when alleles are swapped relative to reference
        if match_alleles and k > 0:
            for j, snp_id in enumerate(ref_order):
                ri = ref_snp_to_idx[snp_id]
                ref_a1, ref_a2 = ref.a1[ri], ref.a2[ri]
                cur_a1, cur_a2 = a1_list[j], a2_list[j]
                if cur_a1 == ref_a2 and cur_a2 == ref_a1:
                    # Alleles swapped — flip effect
                    new_beta[j] = -new_beta[j]
                    if new_af is not None:
                        new_af[j] = 1.0 - new_af[j]
                    a1_list[j] = ref_a1
                    a2_list[j] = ref_a2

        result.append(
            SumStats(
                chr=chr_list,
                pos=pos_list,
                snp=snp_list,
                a1=a1_list,
                a2=a2_list,
                beta=new_beta,
                se=new_se,
                p=new_p,
                n=new_n,
                af=new_af,
            )
        )

    return result
