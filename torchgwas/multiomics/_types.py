"""Result dataclasses for `torchgwas.multiomics`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MediationResult:
    """Single-triple mediation: SNP → M → Y under an LMM with GRM K."""

    a: float
    a_se: float
    b: float
    b_se: float
    c: float
    c_se: float
    c_prime: float
    c_prime_se: float
    indirect: float
    indirect_se: float
    indirect_ci_lower: float
    indirect_ci_upper: float
    indirect_pvalue: float
    total: float
    proportion_mediated: float
    inconsistent: bool
    sensitivity_rho: float | None
    n: int
    se_method: str
    fit_method: str

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class MediationScanResult:
    """Genome × feature mediation scan."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    n_pairs: int = 0
    cis_window_bp: int | None = None
    se_method: str = "monte-carlo"
    fdr_method: str = "bh"

    def to_dataframe(self):  # type: ignore[no-untyped-def]
        import pandas as pd

        return pd.DataFrame(self.rows)

    def to_tsv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, sep="\t", index=False)

    def top_hits(self, q_threshold: float = 0.05):  # type: ignore[no-untyped-def]
        df = self.to_dataframe()
        if "q_indirect" not in df.columns or len(df) == 0:
            return df.iloc[0:0]
        return df.loc[df["q_indirect"] <= q_threshold].copy()


@dataclass
class MultiKernelH2Result:
    """Variance-component partition over a dict of kernels (e.g. SNP + expression)."""

    sigma2: dict[str, float]
    h2: dict[str, float]
    h2_total: float
    kernel_names: list[str]
    n: int
    nullfit: Any = None
