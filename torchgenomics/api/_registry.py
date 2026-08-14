"""Model registry: short aliases over the models/api layer for tg.gwas(models=…)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    """Frozen specification of one model in the registry.

    This dataclass describes a single model that can be used for GWAS scans.
    The registry maps short user-facing aliases (e.g., 'lmm') to their
    corresponding model implementations and run-time dispatch targets.

    Attributes
    ----------
    alias : str
        Short user-facing alias, unique across the registry (e.g., 'lmm', 'glm',
        'farmcpu'). Used as the primary key in MODEL_REGISTRY and accepted by
        :func:`resolve_model` or MCP tool calls.
    label : str
        Internal model class label (e.g., 'SingleTraitLMM', 'GLM', 'FarmCPU').
        Maps to the actual implementation in :mod:`torchgenomics.models`.
    trait_types : tuple[str, ...]
        Tuple of supported trait types, each a string like 'continuous', 'binary',
        or 'categorical'. Guides user selection in the API layer.
    description : str
        One-line human-readable summary of the model (e.g., 'GRM mixed model
        (GEMMA-equivalent)'). Shown in :func:`list_models` output.
    runner : str
        Dispatch target for the high-level API layer. Valid values include:
        'api_lmm' (use :mod:`torchgenomics.api.scans` LMM path),
        'api_glm' (use :mod:`torchgenomics.api.scans` GLM path), or
        'lowlevel:<ClassName>' (instantiate the low-level model class directly).
    """
    alias: str
    label: str
    trait_types: tuple[str, ...]
    description: str
    runner: str  # "api_lmm" | "api_glm" | "lowlevel:<ClassName>"


_SPECS = [
    ModelSpec("lmm", "SingleTraitLMM", ("continuous",), "GRM mixed model (GEMMA-equivalent)", "api_lmm"),
    ModelSpec("glm", "GLM", ("continuous", "binary", "categorical"), "Fixed-effects; PCs as covariates", "api_glm"),
    ModelSpec("farmcpu", "FarmCPU", ("continuous",), "Iterative multi-locus", "lowlevel:FarmCPU"),
    ModelSpec("blink", "BLINK", ("continuous",), "Fast multi-locus", "lowlevel:BLINK"),
    ModelSpec("mvlmm", "MultiTraitLMM", ("continuous",), "Multi-trait LMM", "lowlevel:MultiTraitLMM"),
    ModelSpec("glmm", "BinaryGLMM", ("binary", "categorical"), "PQL GLMM (SAIGE-style)", "lowlevel:BinaryGLMM"),
    ModelSpec("mklmm", "MultiKernelLMM", ("continuous",), "Additive+dominance kernels", "lowlevel:MultiKernelLMM"),
    ModelSpec("gxe", "GxELMM", ("continuous",), "Genotype × environment", "lowlevel:GxELMM"),
    ModelSpec("set", "SetBasedScanner", ("continuous", "binary"), "Region/gene-based (SKAT)", "lowlevel:SetBasedScanner"),
    ModelSpec("bayes", "BayesianVS", ("continuous",), "SuSiE fine-mapping scan", "lowlevel:BayesianVS"),
]
MODEL_REGISTRY: dict[str, ModelSpec] = {s.alias: s for s in _SPECS}

# GAPIT-name synonyms (case-insensitive) for migrants
# Keys map to primary registry aliases; entries like "glm": "glm" are redundant
# (primary aliases resolve directly) and are omitted.
_SYNONYMS = {"mlm": "lmm", "cmlm": "lmm", "mlmm": "farmcpu"}


def resolve_model(name: str) -> ModelSpec:
    """Resolve a user-supplied model name to its ModelSpec.

    Handles user-facing aliases (e.g., 'lmm', 'glm'), GAPIT-name synonyms
    (e.g., 'MLM' → 'lmm', 'Blink' → 'blink'), and case-insensitive input.

    Parameters
    ----------
    name : str
        A model alias or GAPIT-name synonym (case-insensitive). Examples:
        'lmm', 'MLM', 'glm', 'Blink', 'FarmCPU'.

    Returns
    -------
    ModelSpec
        The resolved model specification, ready for use by the API layer
        or MCP tools. Contains alias, label, trait_types, description, and
        dispatch target (runner).

    Raises
    ------
    ValueError
        If `name` is not a valid alias, synonym, or GAPIT name. The error
        message includes a list of valid models and a pointer to
        :func:`list_models` for interactive browsing.

    Examples
    --------
    >>> resolve_model("lmm")
    ModelSpec(alias='lmm', label='SingleTraitLMM', ...)
    >>> resolve_model("Blink")  # case-insensitive GAPIT name
    ModelSpec(alias='blink', label='BLINK', ...)
    >>> resolve_model("mlm")  # synonym for lmm
    ModelSpec(alias='lmm', ...)
    """
    key = str(name).strip().lower()
    key = _SYNONYMS.get(key, key)
    if key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Valid models: {valid}. "
            f"(GAPIT names like MLM/Blink/FarmCPU are accepted too.) See tg.list_models()."
        )
    return MODEL_REGISTRY[key]


def list_models() -> pd.DataFrame:
    """List all available models as a tidy DataFrame.

    Returns a display-ready table of all registered models with their
    aliases, labels, supported trait types, and descriptions. Useful for
    interactive browsing via notebook or CLI.

    Returns
    -------
    pd.DataFrame
        A table with columns:

        - **alias** : short user-facing name (unique, used by :func:`resolve_model`)
        - **label** : internal model class name
        - **trait_types** : comma-separated list of supported trait types
          (e.g., 'continuous', 'continuous,binary', etc.)
        - **description** : one-line human-readable summary

    Examples
    --------
    >>> df = list_models()
    >>> print(df[["alias", "trait_types", "description"]])
       alias     trait_types                         description
    0    lmm        continuous      GRM mixed model (GEMMA-equivalent)
    1    glm continuous,binary,...  Fixed-effects; PCs as covariates
    ...
    """
    return pd.DataFrame(
        [{"alias": s.alias, "label": s.label,
          "trait_types": ",".join(s.trait_types), "description": s.description}
         for s in _SPECS]
    )
