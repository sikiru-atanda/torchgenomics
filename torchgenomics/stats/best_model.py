"""Best gene-action model selection per SNP (charter Section 9g).

After scanning all gene-action models, select the model with the highest
-log10(p) per marker, with optional BIC penalty for multi-column models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..models.base import ScanResult


@dataclass
class BestModelResult:
    """Per-marker best gene-action model selection result."""

    snp: list[str]
    chr: list[str]
    pos: list[int]
    best_model: list[str]
    best_neglog10p: Tensor  # (m,)
    best_p: Tensor  # (m,)
    bic_penalty_applied: bool
    all_model_p: dict[str, Tensor] = field(default_factory=dict)


# Number of parameters for each gene-action model type
_MODEL_PARAMS = {
    "additive": 1,
    "diplo-additive": 1,
    "overdominant": 1,
    "general": None,  # k-1, set dynamically from ploidy
    "diplo-general": None,  # 2 for tetraploid
}


def _n_params(model: str, ploidy: int = 4) -> int:
    """Number of regression parameters for a gene-action model."""
    if model in _MODEL_PARAMS and _MODEL_PARAMS[model] is not None:
        return _MODEL_PARAMS[model]
    if model == "general":
        # Full-rank genotypic model: k dummies for dosage classes 1..k (class 0
        # is the reference). See recode_gene_action("general").
        return ploidy
    if model == "diplo-general":
        return 2  # diploidized then one-hot: 3 classes - 1
    # j-dom models: always 1 parameter
    if "-dom" in model:
        return 1
    return 1  # default


def select_best_model(
    scan_results: dict[str, ScanResult],
    n_samples: int,
    bic_penalty: bool = True,
    ploidy: int = 4,
) -> BestModelResult:
    """Select the best gene-action model per marker by highest -log10(p).

    Parameters
    ----------
    scan_results : dict[str, ScanResult]
        Mapping from model name to ScanResult. All must have same SNP order.
    n_samples : int
        Sample size (for BIC penalty computation).
    bic_penalty : bool
        If True, subtract ``0.5 * (n_params - 1) * log10(n)`` from -log10(p)
        for models with > 1 parameter (general, diplo-general).
    ploidy : int
        Organism ploidy level (used to determine n_params for general model).

    Returns
    -------
    BestModelResult
    """
    if not scan_results:
        raise ValueError("scan_results is empty.")

    model_names = list(scan_results.keys())
    first = scan_results[model_names[0]]
    m = len(first.snp)

    # Validate all results have same SNPs
    for name, res in scan_results.items():
        if len(res.snp) != m:
            raise ValueError(
                f"Model '{name}' has {len(res.snp)} SNPs, "
                f"expected {m} (from '{model_names[0]}')."
            )

    # Build score matrix: (n_models, m)
    all_model_p = {}
    scores = torch.full((len(model_names), m), float("-inf"), dtype=torch.float64)

    for i, name in enumerate(model_names):
        p = scan_results[name].p.to(torch.float64)
        all_model_p[name] = p

        # -log10(p), handling p=0 and NaN
        neglog10p = -torch.log10(torch.clamp(p, min=1e-300))
        neglog10p = torch.where(torch.isfinite(neglog10p), neglog10p, torch.zeros_like(neglog10p))

        if bic_penalty:
            np_ = _n_params(name, ploidy)
            if np_ > 1:
                penalty = 0.5 * (np_ - 1) * math.log10(n_samples)
                neglog10p = neglog10p - penalty

        scores[i] = neglog10p

    # Select best model per marker
    best_idx = scores.argmax(dim=0)  # (m,)
    best_model = [model_names[idx.item()] for idx in best_idx]

    # Get the actual (unpenalized) p-values for the best model
    best_p = torch.zeros(m, dtype=torch.float64)
    best_neglog10p = torch.zeros(m, dtype=torch.float64)
    for j in range(m):
        bm = best_model[j]
        best_p[j] = all_model_p[bm][j]
        best_neglog10p[j] = -torch.log10(torch.clamp(best_p[j:j+1], min=1e-300))[0]

    return BestModelResult(
        snp=first.snp,
        chr=first.chr,
        pos=first.pos,
        best_model=best_model,
        best_neglog10p=best_neglog10p,
        best_p=best_p,
        bic_penalty_applied=bic_penalty,
        all_model_p=all_model_p,
    )
