"""Model registry: short aliases over the models/api layer for tg.gwas(models=…)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
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
_SYNONYMS = {"mlm": "lmm", "cmlm": "lmm", "glm": "glm", "blink": "blink",
             "farmcpu": "farmcpu", "mlmm": "farmcpu"}


def resolve_model(name: str) -> ModelSpec:
    key = str(name).strip().lower()
    key = _SYNONYMS.get(key, key)
    if key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Valid models: {valid}. "
            f"(GAPIT names like MLM/Blink/FarmCPU are accepted too.) See tg.models()."
        )
    return MODEL_REGISTRY[key]


def list_models() -> pd.DataFrame:
    return pd.DataFrame(
        [{"alias": s.alias, "label": s.label,
          "trait_types": ",".join(s.trait_types), "description": s.description}
         for s in _SPECS]
    )
