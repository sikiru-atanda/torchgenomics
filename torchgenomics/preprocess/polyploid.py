"""Polyploid dosage encoding, gene-action model recoding, ploidy detection."""

from __future__ import annotations

import torch
from torch import Tensor

# Gene-action model names for arbitrary ploidy k
GENE_ACTION_MODELS = [
    "general",
    "additive",
    # 1-dom through (k-1)-dom generated dynamically
    "diplo-additive",
    "overdominant",
]


def detect_ploidy(G: Tensor) -> int:
    """Infer ploidy from the maximum non-NaN dosage value in G.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix.

    Returns
    -------
    int
        Detected ploidy level (2, 4, 6, etc.).
    """
    valid = G[~torch.isnan(G)]
    if len(valid) == 0:
        return 2  # default

    max_dosage = int(torch.round(valid.max()).item())

    # Ploidy must be even and >= 2
    if max_dosage <= 2:
        return 2
    elif max_dosage <= 4:
        return 4
    elif max_dosage <= 6:
        return 6
    elif max_dosage <= 8:
        return 8
    else:
        return max_dosage


def recode_gene_action(G: Tensor, model: str, ploidy: int) -> Tensor:
    """Recode dosage tensor under a specific gene-action model.

    For ploidy k, implements GWASpoly-equivalent coding:

    - ``"additive"``: dosage as-is (0..k)
    - ``"1-dom"``: 1 if dosage >= 1, else 0
    - ``"j-dom"``: 1 if dosage >= j, else 0 (for j in 1..k-1)
    - ``"diplo-additive"``: min(dosage, k - dosage)
    - ``"overdominant"``: 1 if 0 < dosage < k, else 0
    - ``"general"``: returns (n, m, k) one-hot-like tensor for all genotype classes

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix (values in [0, k]).
    model : str
        Gene-action model name.
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    Tensor
        Recoded dosage. Shape (n, m) for scalar models, (n, m, k) for "general".
    """
    nan_mask = torch.isnan(G)

    if model == "additive":
        return G

    elif model == "overdominant":
        G_safe = torch.where(nan_mask, torch.zeros_like(G), G)
        result = ((G_safe > 0) & (G_safe < ploidy)).to(G.dtype)
        result[nan_mask] = float("nan")
        return result

    elif model == "diplo-additive":
        # Diploidize the polyploid dosage (GWASpoly; Rosyara et al. 2016):
        # nulliplex (dosage 0) -> 0, full homozygous-alt (dosage k) -> 2, and
        # every heterozygous class (0 < dosage < k) -> 1. This is monotone in
        # dosage. The previous torch.min(G, k-G) produced a *folded* coding
        # ([0,1,2,1,0] for tetraploid) that mapped homozygous-alt onto
        # homozygous-ref and is non-monotonic -- corrupting both the design
        # matrix and the polyploid GRM built from it.
        G_safe = torch.where(nan_mask, torch.zeros_like(G), G)
        result = torch.where(
            G_safe >= ploidy,
            torch.full_like(G_safe, 2.0),
            (G_safe > 0).to(G.dtype),  # any heterozygote -> 1, nulliplex -> 0
        )
        result[nan_mask] = float("nan")
        return result

    elif model.endswith("-dom"):
        # Parse j from "j-dom"
        j_str = model.split("-")[0]
        try:
            j = int(j_str)
        except ValueError:
            raise ValueError(f"Invalid dom model: {model}. Expected format: '1-dom', '2-dom', etc.")

        if j < 1 or j >= ploidy:
            raise ValueError(f"j-dom requires 1 <= j < ploidy ({ploidy}), got j={j}")

        G_safe = torch.where(nan_mask, torch.zeros_like(G), G)
        result = (G_safe >= j).to(G.dtype)
        result[nan_mask] = float("nan")
        return result

    elif model == "general":
        # Full-rank genotypic ("general") model: one dummy per non-reference
        # dosage class. There are k+1 classes {0,1,...,k}; with dosage 0 as the
        # reference we emit k dummies for classes 1..k -> (n, m, k). The
        # previous code used k-1 dummies (excluding BOTH dosage 0 and dosage k),
        # which collapsed the two homozygous classes (0 and k) to the same
        # all-zeros encoding, making them indistinguishable and dropping a
        # degree of freedom. NOTE: this encoding is a latent path -- the poly
        # scan feeds raw additive dosages and grm_polyploid_gene_action falls
        # back to the additive GRM for "general" -- so fixing it here does not
        # change current GWAS outputs, but any consumer of the encoding now
        # gets the correct full-rank genotypic design.
        n, m = G.shape
        n_classes = ploidy  # dummies for dosage classes 1, 2, ..., k
        result = torch.zeros(n, m, n_classes, dtype=G.dtype)

        G_safe = torch.where(nan_mask, torch.full_like(G, -1), G)
        G_rounded = torch.round(G_safe).long()

        for c in range(n_classes):
            dosage_class = c + 1  # 1, 2, ..., k
            indicator = (G_rounded == dosage_class).to(G.dtype)
            indicator[nan_mask] = float("nan")
            result[:, :, c] = indicator

        return result

    else:
        raise ValueError(
            f"Unknown gene-action model: '{model}'. "
            f"Choose from: additive, 1-dom..{ploidy-1}-dom, diplo-additive, overdominant, general"
        )


def list_gene_action_models(ploidy: int) -> list[str]:
    """Return all valid gene-action model names for the given ploidy.

    Parameters
    ----------
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    list[str]
        Model names: additive, 1-dom...(k-1)-dom, diplo-additive, overdominant, general.
    """
    models = ["additive"]
    for j in range(1, ploidy):
        models.append(f"{j}-dom")
    models.extend(["diplo-additive", "overdominant", "general"])
    return models
