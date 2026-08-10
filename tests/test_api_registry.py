# tests/test_api_registry.py
import pytest
from torchgenomics.api._registry import resolve_model, list_models, MODEL_REGISTRY

def test_registry_core_aliases_and_synonyms():
    assert resolve_model("lmm").runner == "api_lmm"
    assert resolve_model("glm").runner == "api_glm"
    # GAPIT-name synonyms map in
    assert resolve_model("MLM").alias == "lmm"
    assert resolve_model("Blink").alias == "blink"
    assert resolve_model("FarmCPU").alias == "farmcpu"
    # every spec advertises trait types + a description
    for spec in MODEL_REGISTRY.values():
        assert spec.trait_types and spec.description
    # listing is a tidy table
    df = list_models()
    assert {"alias", "label", "trait_types", "description"}.issubset(df.columns)
    assert "lmm" in set(df["alias"])

def test_unknown_model_raises_friendly_error():
    with pytest.raises(ValueError) as e:
        resolve_model("mlmx")
    msg = str(e.value)
    assert "mlmx" in msg and "lmm" in msg  # names the bad input + lists valid options
