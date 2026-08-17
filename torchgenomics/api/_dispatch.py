"""Shared dispatch core for the friendly-API entry point ``tg.gwas(...)``.

This module is the linchpin between the resolved user inputs
(:class:`~torchgenomics.api._inputs.GwasInputs`, Task 2) and the *already
tested* scan machinery of the library. It deliberately re-implements **no**
model math: every route delegates to a battle-tested code path —

  * ``spec.runner in {"api_lmm", "api_glm"}`` → the tier-1 api functions
    :func:`torchgenomics.api.scans.lmm_scan` / :func:`~torchgenomics.api.scans.glm_scan`
    (which themselves wrap :func:`torchgenomics.cli._run_lmm_scan` /
    ``_run_glm_scan``).
  * ``spec.runner.startswith("lowlevel:")`` → the CLI single-trait runner for
    that model (e.g. :func:`torchgenomics.cli._cmd_blink_scan_single`), which
    already encodes the correct per-model streaming scan strategy and writes a
    ``<prefix>.assoc.tsv`` association table; the output is then read back into
    a uniform :class:`~torchgenomics.api._results.GwasResult`.

Both routes require **file paths** (there is no in-memory entry point in the
existing scan functions). When ``inputs`` carries an in-memory genotype
(:class:`~torchgenomics.api._inputs.ArrayReader` or any duck-typed
``GenotypeReader``), the matrix and phenotype are materialized to temp files
in the numeric-dosage "3-column" CSV format
(:class:`torchgenomics.io.numeric.NumericDosageReader`) and a phenotype TSV.
A file-path genotype is passed straight through untouched.

Four low-level models are fully wired here — ``blink``, ``farmcpu``, ``glmm``
(``--family`` inferred from the trait type), and ``mklmm`` (a friendly
default ``--kernels additive,dominance``) — because each needs nothing
beyond ``(phenotype, genotype[, family])``. Models that need extra inputs
(``gxe`` needs an environment, ``set`` needs regions, ``mvlmm`` needs
multiple traits, ``bayes`` needs signal priors) raise a clear
:class:`NotImplementedError` rather than silently returning a wrong result.

.. note::
   Kinship/PC reuse: for now the LMM path lets ``lmm_scan`` compute its own
   VanRaden GRM when ``kinship == "auto"`` (``grm=None``). Task 5 may compute a
   GRM once and pass it through :attr:`RunOptions.kinship` so a multi-model
   comparison shares one GRM instead of recomputing it per model.

.. note::
   **Consistency invariant (Task 7 audit).** :func:`run_model` is the single
   unified scan path for the friendly API: every alias that
   :func:`~torchgenomics.api._registry.resolve_model` can resolve to a wired
   runner (``api_lmm``, ``api_glm``, or a ``lowlevel:`` entry present in
   :data:`_LOWLEVEL_CLI`) returns the same shape — a
   :class:`~torchgenomics.api._results.GwasResult` with ``.trait_type`` tagged
   from ``inputs.trait_type`` and ``.hits`` / ``.diagnostics`` / ``.summary()``
   all populated. ``tg.gwas(...)`` (:func:`torchgenomics.api.gwas.gwas`) never
   calls a scan function directly — it always routes through this function, so
   that uniformity holds for both the single-model (``GwasResult``) and
   multi-model (``GwasComparison``, one ``GwasResult`` per requested model)
   return shapes. Aliases whose runner is not yet wired (e.g. ``gxe``,
   ``set``, ``mvlmm``, ``bayes``) raise a clear
   :class:`NotImplementedError` here rather than returning a divergent or
   partially-populated result. The low-level :func:`torchgenomics.api.scans.lmm_scan`
   / :func:`~torchgenomics.api.scans.glm_scan` functions, called directly
   (bypassing ``run_model``/``tg.gwas``), also return a ``GwasResult`` but
   leave ``.trait_type`` at its default (``""``) — that enrichment is applied
   here, one line after each call, and is a property of the friendly-API
   surface (``tg.gwas`` / ``run_model``), not of the lower-level per-model
   functions.
"""

from __future__ import annotations

import shutil
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..preprocess.qc import QCFilterConfig
from ._helpers import top_hits
from ._inputs import GwasInputs
from ._registry import resolve_model
from ._results import GwasResult

# Trait-type → GLM family map for the ``api_glm`` route. Ordinal / multinomial
# need an explicit ``n_categories`` that the friendly-input layer does not
# currently carry, so ``categorical`` is intentionally left unwired here.
_GLM_FAMILY = {"continuous": "gaussian", "binary": "binary"}

_VALID_CORRECTIONS = frozenset(
    {
        "bonferroni", "bh", "by", "holm", "storey", "weighted-bh",
        "lfdr", "hierarchical", "ihw", "adapt", "none",
    }
)


@dataclass
class RunOptions:
    """User-tunable knobs for a single :func:`run_model` invocation.

    All fields have friendly defaults so a novice can call
    ``run_model("lmm", inputs)`` with no options at all. Values are validated
    at the top of :func:`run_model`, raising ``ValueError`` with an actionable
    message rather than letting a bad value surface as an opaque downstream
    error.

    Parameters
    ----------
    kinship : str | bool | None | Any, default "auto"
        GRM / kinship policy for mixed-model (LMM) routes.
        ``"auto"`` lets :func:`~torchgenomics.api.scans.lmm_scan` compute a
        streaming VanRaden GRM itself (``grm=None``). A filesystem path
        (``str``) to a pre-computed GRM is passed straight through to
        ``lmm_scan(grm=...)``. ``False`` / ``None`` also fall through to an
        auto GRM for LMM (an LMM cannot run without one); they are meaningful
        only for fixed-effects / multi-locus models (e.g. ``blink``,
        ``farmcpu``) that ignore kinship entirely.
    pcs : str | int | bool | None, default "auto"
        Number of genotype principal components to add as covariates. An
        ``int`` is used verbatim. ``0`` / ``None`` / ``False`` mean "no PCs".
        ``"auto"`` currently resolves to ``0`` (no PCs); Task 5 may replace
        this with a data-driven choice.
    qc : bool, default True
        Whether per-variant QC (MAF / missingness / Hardy-Weinberg) is
        applied. When ``True`` (default) the standard triplet is used —
        ``maf_min=0.01``, ``miss_max=0.1``, and HWE filtering at the library
        default (:data:`torchgenomics.preprocess.qc.QCFilterConfig.hwe_p_min`,
        currently ``1e-6``) — matching the CLI and the rest of the library.
        When ``False`` all per-variant filters are disabled (pure
        pass-through: ``maf_min=0.0``, ``miss_max=1.0``, ``hwe_p_min=0.0``).
    hwe_p_min : float | None, default None
        Optional power-user override of the HWE p-value filter threshold,
        used only when :attr:`qc` is ``True``. ``None`` (default) means "use
        the library default" (see :attr:`qc` above). Set explicitly — e.g.
        ``hwe_p_min=0.0`` — to disable *just* the HWE filter while keeping
        MAF / missingness QC on, without setting ``qc=False`` (which would
        also relax MAF/missingness). Ignored when ``qc=False`` (HWE is
        already off in that case).
    correction : str, default "bh"
        Multiple-testing correction, one of the values accepted by the
        underlying scans (``"bh"``, ``"bonferroni"``, ``"holm"``, ``"by"``,
        ``"storey"``, ``"none"``, ...). Validated against
        :data:`_VALID_CORRECTIONS`.
    device : str | None, default None
        ``"cpu"`` / ``"cuda"`` / ``"auto"``. ``None`` resolves to ``"auto"``
        (CUDA when available) for the api routes and to the CLI default for
        low-level routes.
    output : str | pathlib.Path | None, default None
        Directory for the results tables. ``None`` (default) writes to a
        process-local temp directory (still referenced by
        :attr:`GwasResult.output_files` so plotting / report helpers keep
        working within the process) that is removed automatically — via a
        :func:`weakref.finalize` callback registered on the returned
        :class:`~torchgenomics.api._results.GwasResult` in :func:`run_model`
        — once that result object is garbage-collected; nothing accumulates
        across repeated calls in a long-running process (e.g. a loop over
        many traits). Pass an explicit directory here to persist results
        instead of relying on GC timing.
    top_k : int, default 50
        Number of top hits returned inline in :attr:`GwasResult.top_hits`.
    verbose : bool, default True
        Reserved for future progress/logging control; currently unused by the
        dispatch core (kept so callers can pass it uniformly).
    model_options : dict[str, dict] | None, default None
        Per-model advanced settings for low-level (CLI-backed) models, keyed
        by model alias — e.g. ``{"glmm": {"family": "ordinal"}, "mklmm":
        {"kernels": "additive,dominance,epistatic"}}``. Each inner dict is
        translated into the model's CLI flags (key ``k`` -> ``--k-with-dashes``;
        a bool ``True`` -> a bare store-true flag, ``False`` -> omitted; any
        other scalar -> ``--flag value``). Ignored by api-backed models
        (``lmm``/``glm``). Absent keys fall back to each model's friendly
        defaults (see :func:`_lowlevel_extra_argv`).
    """

    kinship: Any = "auto"
    pcs: Any = "auto"
    qc: bool = True
    hwe_p_min: float | None = None
    correction: str = "bh"
    device: str | None = None
    output: str | Path | None = None
    top_k: int = 50
    verbose: bool = True
    model_options: dict[str, dict] | None = None


def _n_pcs_from_opts(opts: RunOptions) -> int:
    """Resolve :attr:`RunOptions.pcs` to a concrete non-negative ``n_pcs``.

    ``"auto"`` → ``0`` (no PCs by default; Task 5 may refine). ``0`` /
    ``None`` / ``False`` → ``0``. An ``int`` (or int-like) is used verbatim
    after a non-negativity check. Anything else raises ``ValueError``.
    """
    pcs = opts.pcs
    if pcs is None or pcs is False or pcs == "auto":
        return 0
    if isinstance(pcs, bool):  # True — meaningless as a PC count
        raise ValueError("pcs=True is not a valid PC count; pass an int, 0, or 'auto'.")
    try:
        n = int(pcs)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"pcs={pcs!r} is not a valid number of principal components. Pass a "
            "non-negative int, 0/None/False for none, or 'auto'."
        ) from e
    if n < 0:
        raise ValueError(f"pcs must be non-negative; got {n}.")
    return n


def _grm_arg_from_opts(opts: RunOptions) -> str | None:
    """Resolve :attr:`RunOptions.kinship` to the ``grm=`` argument of ``lmm_scan``.

    A filesystem path (``str`` / :class:`~pathlib.Path`) is returned as a
    ``str`` so ``lmm_scan`` loads it. ``"auto"`` / ``False`` / ``None``
    return ``None`` (``lmm_scan`` then computes a streaming VanRaden GRM —
    an LMM requires a GRM, so "no kinship" collapses to "auto" here).
    """
    kinship = opts.kinship
    if isinstance(kinship, (str, Path)) and str(kinship) != "auto":
        return str(kinship)
    return None


def _device_for_api(opts: RunOptions) -> str:
    """Resolve device for the api routes: ``None`` → ``"auto"``."""
    return "auto" if opts.device is None else str(opts.device)


def _write_phenotype_tsv(inputs: GwasInputs, path: Path) -> None:
    """Write ``inputs.phenotype`` as a two-column ``sample_id\\t<trait>`` TSV.

    The index (sample ids) becomes the ``sample_id`` column so the downstream
    loader (:func:`torchgenomics.io.phenotype.load_phenotype` /
    :func:`~torchgenomics.io.phenotype._detect_id_column`) auto-detects it and
    aligns by id against the genotype.
    """
    series = inputs.phenotype.copy()
    series.name = inputs.trait_name
    df = series.to_frame()
    df.index = [str(i) for i in df.index]
    df.index.name = "sample_id"
    df.to_csv(path, sep="\t")


def _write_genotype_csv(reader: Any, path: Path) -> None:
    """Materialize an in-memory genotype reader to a 3-column-rule dosage CSV.

    Streams ``reader.iter_chunks()`` (never assuming a dense ``.G`` attribute,
    so it works for both :class:`~torchgenomics.api._inputs.ArrayReader` and
    :class:`~torchgenomics.io.aligned.SampleAlignedReader`) and writes a
    markers-as-rows CSV whose header is ``SNP,Chr,Pos,<sample_id_1>,...`` —
    exactly the "structured" layout that
    :class:`torchgenomics.io.numeric.NumericDosageReader` reads back without a
    companion ``.map`` file.
    """
    sample_ids = [str(s) for s in reader.sample_ids]
    snp: list[str] = []
    chrom: list[str] = []
    pos: list[int] = []
    blocks: list[np.ndarray] = []
    for g_chunk, vmeta in reader.iter_chunks():
        # g_chunk is (n_samples, m_chunk); dosage rows are markers → transpose.
        blocks.append(np.asarray(g_chunk.detach().cpu().numpy(), dtype=np.float64).T)
        snp.extend(str(s) for s in vmeta.snp)
        chrom.extend(str(c) for c in vmeta.chr)
        pos.extend(int(p) for p in vmeta.pos)

    dosage = np.vstack(blocks) if blocks else np.empty((0, len(sample_ids)))
    df = pd.DataFrame(dosage, columns=sample_ids)
    df.insert(0, "Pos", pos)
    df.insert(0, "Chr", chrom)
    df.insert(0, "SNP", snp)
    df.to_csv(path, index=False)


def _materialize_inputs(inputs: GwasInputs, workdir: Path) -> tuple[str, str]:
    """Return ``(genotype_path, phenotype_path)`` for the file-based scan APIs.

    The phenotype is always written to a TSV (built from the aligned
    :attr:`GwasInputs.phenotype` Series). The genotype is passed through
    untouched when it is already a path, or materialized to a dosage CSV when
    it is an in-memory reader.
    """
    pheno_path = workdir / "phenotype.tsv"
    _write_phenotype_tsv(inputs, pheno_path)

    geno = inputs.genotype_path_or_reader
    if isinstance(geno, (str, Path)):
        geno_path = str(geno)
    else:
        geno_path = str(workdir / "genotype.csv")
        _write_genotype_csv(geno, workdir / "genotype.csv")

    return geno_path, str(pheno_path)


def _read_scan_output(
    output_prefix: str | Path,
    *,
    model_label: str,
    test: str,
    correction: str,
    trait_type: str,
    n_samples: int,
    significance_threshold: float,
    top_k: int,
    runtime_s: float,
) -> GwasResult:
    """Read a CLI scan's ``<prefix>.assoc.{tsv,parquet}`` back into a ``GwasResult``.

    Mirrors the read-back logic already used by
    :func:`torchgenomics.api.scans.lmm_scan`: prefers Parquet when present,
    computes ``n_significant`` against ``significance_threshold``, ``lambda_gc``
    via :func:`torchgenomics.api.scans._genomic_inflation`, and ``top_hits`` via
    :func:`torchgenomics.api._helpers.top_hits` — deliberately reusing those
    shared helpers rather than duplicating the statistics.
    """
    from . import scans

    prefix = str(output_prefix)
    tsv_path = Path(prefix + ".assoc.tsv")
    if not tsv_path.exists():
        tsv_path = Path(prefix + ".tsv")
    parquet_path = Path(prefix + ".assoc.parquet")
    if not parquet_path.exists():
        parquet_path = Path(prefix + ".parquet")

    output_files: dict[str, Path] = {}
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        output_files["parquet"] = parquet_path
        if tsv_path.exists():
            output_files["tsv"] = tsv_path
    elif tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
        output_files["tsv"] = tsv_path
    else:
        raise RuntimeError(
            f"scan completed but no results file was found at {prefix}.assoc.tsv"
        )

    p_col = "P" if "P" in df.columns else "P_JOINT" if "P_JOINT" in df.columns else None
    n_sig = int((df[p_col] < significance_threshold).sum()) if p_col else 0
    lambda_gc = scans._genomic_inflation(df[p_col]) if p_col else None
    top = top_hits(df, k=top_k, p_column=p_col or "P")

    return GwasResult(
        runtime_s=runtime_s,
        output_files=output_files,
        model=model_label,
        test=test,
        correction=correction,
        n_variants=int(len(df)),
        n_significant=n_sig,
        significance_threshold=significance_threshold,
        lambda_gc=lambda_gc,
        n_samples=n_samples,
        top_hits=top,
        trait_type=trait_type,
    )


def _qc_kwargs(opts: RunOptions) -> dict[str, float]:
    """Friendly-API per-variant QC thresholds for the ``lmm_scan`` / ``glm_scan`` calls.

    ``tg.gwas`` matches the rest of the library by default: when
    :attr:`RunOptions.qc` is ``True`` (default) the standard MAF /
    missingness / Hardy-Weinberg triplet is applied, with HWE filtering at
    the library default (:data:`torchgenomics.preprocess.qc.QCFilterConfig.hwe_p_min`,
    currently ``1e-6``) unless the caller supplies an explicit
    :attr:`RunOptions.hwe_p_min` override (e.g. ``0.0`` to disable just the
    HWE filter while keeping MAF/missingness on). When ``qc`` is ``False``
    all per-variant filters are disabled so the scan is a pure pass-through.
    """
    if opts.qc:
        hwe = QCFilterConfig.hwe_p_min if opts.hwe_p_min is None else opts.hwe_p_min
        return {"maf_min": 0.01, "miss_max": 0.1, "hwe_p_min": hwe}
    return {"maf_min": 0.0, "miss_max": 1.0, "hwe_p_min": 0.0}


def _run_api_lmm(inputs: GwasInputs, opts: RunOptions, workdir: Path) -> GwasResult:
    """Route the ``api_lmm`` runner through :func:`torchgenomics.api.scans.lmm_scan`."""
    from . import scans

    geno_path, pheno_path = _materialize_inputs(inputs, workdir)
    output = opts.output if opts.output is not None else workdir / "lmm_out"
    result = scans.lmm_scan(
        genotype=geno_path,
        phenotype=pheno_path,
        correction=opts.correction,
        n_pcs=_n_pcs_from_opts(opts),
        grm=_grm_arg_from_opts(opts),
        device=_device_for_api(opts),
        output=str(output),
        top_k=opts.top_k,
        **_qc_kwargs(opts),
    )
    # Backfill metadata that doesn't survive CSV/Parquet round-trip in the scan.
    result.trait_type = inputs.trait_type
    if not result.n_samples:
        result.n_samples = inputs.n_samples
    return result


def _run_api_glm(inputs: GwasInputs, opts: RunOptions, workdir: Path) -> GwasResult:
    """Route the ``api_glm`` runner through :func:`torchgenomics.api.scans.glm_scan`.

    The GLM family is inferred from :attr:`GwasInputs.trait_type`:
    ``continuous`` → ``"gaussian"``, ``binary`` → ``"binary"``. ``categorical``
    is not wired here (it needs an explicit ``n_categories`` the friendly-input
    layer does not carry) and raises :class:`NotImplementedError`.
    """
    from . import scans

    family = _GLM_FAMILY.get(inputs.trait_type)
    if family is None:
        raise NotImplementedError(
            f"GLM route for trait_type={inputs.trait_type!r} is not yet wired through "
            "tg.gwas (ordinal/multinomial need an explicit n_categories); use "
            "`torchgenomics glm-scan --family ordinal/multinomial --n-categories N` "
            "or the low-level API."
        )
    geno_path, pheno_path = _materialize_inputs(inputs, workdir)
    output = opts.output if opts.output is not None else workdir / "glm_out"
    result = scans.glm_scan(
        genotype=geno_path,
        phenotype=pheno_path,
        family=family,  # type: ignore[arg-type]
        correction=opts.correction,
        device=_device_for_api(opts),
        output=str(output),
        top_k=opts.top_k,
        **_qc_kwargs(opts),
    )
    # Backfill metadata that doesn't survive CSV/Parquet round-trip in the scan.
    result.trait_type = inputs.trait_type
    if not result.n_samples:
        result.n_samples = inputs.n_samples
    return result


def _run_lowlevel_cli(
    inputs: GwasInputs,
    opts: RunOptions,
    *,
    subcommand: str,
    runner,
    model_label: str,
    workdir: Path,
    alias: str,
) -> GwasResult:
    """Route a low-level model through its CLI single-trait runner.

    Builds a fully-defaulted :class:`argparse.Namespace` by parsing a minimal
    valid arg vector against the real subparser (so every model-specific
    default is populated exactly as the CLI would), overrides only
    genotype/phenotype/output/correction/device, invokes ``runner`` (which
    fits the null, streams the scan, and writes ``<prefix>.assoc.tsv`` via
    :func:`torchgenomics.cli._apply_correction_and_save`), then reads the
    output back into a uniform :class:`GwasResult`.
    """
    import time

    from .. import cli

    geno_path, pheno_path = _materialize_inputs(inputs, workdir)
    output_prefix = str(opts.output) if opts.output is not None else str(workdir / "lowlevel_out")
    extra_argv = _lowlevel_extra_argv(alias, inputs, opts)

    argv = [
        subcommand,
        "--genotype", geno_path,
        "--phenotype", pheno_path,
        "--output", output_prefix,
        "--correction", opts.correction,
    ]
    if opts.device is not None:
        argv += ["--device", str(opts.device)]
    argv += extra_argv

    parser = cli._build_parser()
    args = parser.parse_args(argv)

    start = time.monotonic()
    exit_code = runner(args)
    runtime_s = time.monotonic() - start
    if exit_code != 0:
        raise RuntimeError(f"{subcommand} returned non-zero status {exit_code}")

    return _read_scan_output(
        output_prefix,
        model_label=model_label,
        test=getattr(args, "test", "wald"),
        correction=opts.correction,
        trait_type=inputs.trait_type,
        n_samples=inputs.n_samples,
        significance_threshold=5e-8,
        top_k=opts.top_k,
        runtime_s=runtime_s,
    )


def _model_options_to_argv(opts_map: dict) -> list[str]:
    """Translate a per-model options dict into CLI argument tokens.

    Key ``k`` becomes ``--k-with-dashes``. A bool ``True`` becomes a bare
    store-true flag; ``False`` is omitted entirely. Any other scalar ``v``
    becomes the pair ``["--flag", str(v)]``. Insertion order is preserved so
    the resulting argv is deterministic.
    """
    argv: list[str] = []
    for key, val in (opts_map or {}).items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        else:
            argv += [flag, str(val)]
    return argv


def _glmm_extra_argv(inputs: GwasInputs, user_opts: dict) -> list[str]:
    """Build glmm CLI flags, inferring ``--family`` from the trait type.

    binary -> ``--family binary``; categorical -> ``--family multinomial``
    with ``--n-categories`` set to the number of distinct phenotype values
    (multinomial is the default because it assumes no category ordering — a
    user with ordered categories can pass ``model_options={"glmm":
    {"family": "ordinal"}}``). A continuous trait raises a friendly
    ``ValueError``. Explicit user options always win over the inference.
    """
    import pandas as pd

    opts_map = dict(user_opts)
    if "family" not in opts_map:
        tt = inputs.trait_type
        if tt == "binary":
            opts_map["family"] = "binary"
        elif tt == "categorical":
            opts_map["family"] = "multinomial"
            opts_map.setdefault("n_categories", int(pd.Series(inputs.phenotype).nunique()))
        else:
            raise ValueError(
                f"Model 'glmm' is for binary/categorical traits, but trait "
                f"'{inputs.trait_name}' is {tt}. Use models='lmm' for a "
                f"continuous trait (or models='glm' for fixed-effects)."
            )
    return _model_options_to_argv(opts_map)


def _mklmm_extra_argv(user_opts: dict) -> list[str]:
    """Build mklmm CLI flags with a friendly default kernel set.

    Defaults to ``--kernels additive,dominance`` (the mklmm-scan CLI default
    also includes the expensive ``epistatic`` kernel; tg.gwas omits it for a
    fast, friendly default). A user can opt back in via
    ``model_options={"mklmm": {"kernels": "additive,dominance,epistatic"}}``.
    """
    opts_map = dict(user_opts)
    opts_map.setdefault("kernels", "additive,dominance")
    return _model_options_to_argv(opts_map)


def _lowlevel_extra_argv(alias: str, inputs: GwasInputs, opts: RunOptions) -> list[str]:
    """Build the model-specific CLI flags for a low-level (CLI-backed) model.

    Merges each model's friendly defaults with any user overrides in
    ``opts.model_options[alias]`` and returns them as argv tokens appended to
    the base ``_run_lowlevel_cli`` argv. Models with no special handling
    (e.g. ``blink``/``farmcpu``) simply forward any user options verbatim.
    ``glmm`` infers ``--family`` from the trait type via
    :func:`_glmm_extra_argv`. ``mklmm`` defaults to a lighter kernel set via
    :func:`_mklmm_extra_argv`. Per-model default/inference logic for other
    models is added by later tasks.
    """
    user_opts = (opts.model_options or {}).get(alias, {})
    if alias == "glmm":
        return _glmm_extra_argv(inputs, user_opts)
    if alias == "mklmm":
        return _mklmm_extra_argv(user_opts)
    return _model_options_to_argv(user_opts)


#: Low-level aliases that are fully wired through their CLI single-trait
#: runner. Each entry is ``alias -> (subcommand, runner_fn_name, model_label)``.
#: The runner is looked up lazily on :mod:`torchgenomics.cli` to avoid importing
#: the (heavy) CLI module at api import time.
_LOWLEVEL_CLI = {
    "blink": ("blink-scan", "_cmd_blink_scan_single", "BLINK"),
    "farmcpu": ("farmcpu-scan", "_cmd_farmcpu_scan_single", "FarmCPU"),
    "glmm": ("glmm-scan", "_cmd_glmm_scan", "GLMM"),
    "mklmm": ("mklmm-scan", "_cmd_mklmm_scan", "MultiKernelLMM"),
}


def _cleanup_workdir(path: Path) -> None:
    """Best-effort ``rmtree`` of a :func:`run_model` temp working directory.

    Registered via :func:`weakref.finalize` on the :class:`GwasResult`
    returned by :func:`run_model` (see there), so it fires exactly once,
    automatically, when that result is garbage-collected — no explicit
    ``close()``/context-manager call is required from callers. ``ignore_errors``
    is deliberate: this runs during GC (possibly interpreter shutdown, or
    after the directory was already moved/removed by the caller), where a
    raised exception would be unrecoverable/unreportable and should never
    propagate out of a finalizer.
    """
    shutil.rmtree(path, ignore_errors=True)


def run_model(
    alias: str,
    inputs: GwasInputs,
    opts: RunOptions | None = None,
) -> GwasResult:
    """Run one resolved model end-to-end and return a uniform :class:`GwasResult`.

    This is the shared dispatch core behind ``tg.gwas(...)``. It resolves
    ``alias`` to a :class:`~torchgenomics.api._registry.ModelSpec`, routes to
    the appropriate *existing* scan path (the tier-1 api ``lmm_scan`` /
    ``glm_scan`` for GRM/fixed-effects models, or the CLI single-trait runner
    for wired low-level models), and normalizes every outcome to a
    :class:`~torchgenomics.api._results.GwasResult`. No model statistics are
    computed here — this layer is pure orchestration.

    Parameters
    ----------
    alias : str
        Model alias or GAPIT-name synonym (case-insensitive), resolved via
        :func:`torchgenomics.api._registry.resolve_model` (e.g. ``"lmm"``,
        ``"glm"``, ``"blink"``, ``"farmcpu"``, ``"MLM"``).
    inputs : GwasInputs
        Sample-aligned inputs from
        :func:`torchgenomics.api._inputs.load_inputs`. An in-memory genotype
        is materialized to temp files; a path genotype is used as-is.
    opts : RunOptions | None, default None
        Run options (kinship / PCs / correction / device / output / top_k).
        Defaults to :class:`RunOptions` with its friendly defaults.

    Returns
    -------
    GwasResult
        Uniform result with summary stats, the top-``k`` hits inline, and
        paths to the full association table on disk. ``trait_type`` is tagged
        from ``inputs``. Its temp working directory (materialized inputs,
        plus outputs when ``opts.output`` was not given) is removed
        automatically once this result is garbage-collected — see the
        "Temp-directory lifecycle" note below.

    Raises
    ------
    ValueError
        On an unknown ``alias`` (from :func:`resolve_model`) or an invalid
        option (bad ``correction``, ``pcs``, or ``top_k``).
    NotImplementedError
        For registry models that need inputs beyond ``(phenotype, genotype[,
        kinship])`` — e.g. ``gxe`` (environment), ``set`` (regions),
        ``mvlmm`` (multiple traits), ``bayes`` (signal priors) — which are
        not yet wired through ``tg.gwas``. The message points at the
        equivalent ``torchgenomics <alias>-scan`` CLI subcommand and the
        low-level API.

    Notes
    -----
    **Temp-directory lifecycle.** Every call creates a fresh
    ``tempfile.mkdtemp(prefix="tg_gwas_")`` working directory to hold
    materialized inputs (when ``inputs`` carries an in-memory genotype) and,
    when ``opts.output`` is ``None``, the scan's own output files (which the
    returned :class:`GwasResult` then references via ``output_files`` for
    ``.manhattan()`` / ``.report()``). Rather than deleting it immediately —
    which would break those methods — a :func:`weakref.finalize` callback
    (:func:`_cleanup_workdir`) is attached to the returned result, so the
    directory is removed automatically once that result is garbage-collected
    (or immediately, on any exception raised before a result exists). This
    means directories never accumulate unboundedly across repeated calls
    (e.g. ``tg.gwas`` in a loop over many traits) even though nothing is
    deleted eagerly. When ``opts.output`` is given, the scan itself writes
    directly to that user-supplied directory — ``output_files`` then points
    there, not at the temp directory — so the temp directory (still created,
    to hold only materialized inputs) is redundant sooner but is cleaned up
    on the same schedule regardless.
    """
    opts = opts if opts is not None else RunOptions()

    # --- Friendly option validation (fail fast, actionable messages) ---
    if opts.correction not in _VALID_CORRECTIONS:
        raise ValueError(
            f"correction={opts.correction!r} is not recognized. Valid options: "
            f"{sorted(_VALID_CORRECTIONS)}."
        )
    if not isinstance(opts.top_k, int) or isinstance(opts.top_k, bool) or opts.top_k <= 0:
        raise ValueError(f"top_k must be a positive int; got {opts.top_k!r}.")
    _n_pcs_from_opts(opts)  # validates pcs early

    spec = resolve_model(alias)

    # A temp working dir holds materialized inputs (and outputs when the caller
    # gave no explicit output=). See "Temp-directory lifecycle" above: it is not
    # deleted here (the returned GwasResult references result files here for
    # .manhattan() / .report()) but is guaranteed cleanup via weakref.finalize
    # below on success, or immediately below on any exception.
    workdir = Path(tempfile.mkdtemp(prefix="tg_gwas_"))

    try:
        if spec.runner == "api_lmm":
            result = _run_api_lmm(inputs, opts, workdir)
        elif spec.runner == "api_glm":
            result = _run_api_glm(inputs, opts, workdir)
        elif spec.runner.startswith("lowlevel:"):
            low = spec.alias.lower()
            wiring = _LOWLEVEL_CLI.get(low)
            if wiring is None:
                raise NotImplementedError(
                    f"Model '{alias}' is not yet wired through tg.gwas; use "
                    f"`torchgenomics {low}-scan` or the low-level API."
                )
            from .. import cli

            subcommand, runner_name, model_label = wiring
            runner = getattr(cli, runner_name)
            result = _run_lowlevel_cli(
                inputs,
                opts,
                subcommand=subcommand,
                runner=runner,
                model_label=model_label,
                workdir=workdir,
                alias=low,
            )
        else:
            raise NotImplementedError(
                f"Model '{alias}' has an unrecognized runner {spec.runner!r} and is not "
                "yet wired through tg.gwas; use the corresponding CLI subcommand or the "
                "low-level API."
            )
    except BaseException:
        # No result object exists to hold a finalizer -- clean up right away
        # so a failed run never leaks its temp directory.
        _cleanup_workdir(workdir)
        raise

    # Success: hand cleanup off to GC via a finalizer on the result itself,
    # so .manhattan() / .report() keep working for as long as the caller
    # holds a reference to `result` (see "Temp-directory lifecycle" above).
    weakref.finalize(result, _cleanup_workdir, workdir)
    return result


# --- Task 6: post-run advisory warnings -------------------------------------

#: λ_GC above this is flagged as inflation. Matches the calibration verdict
#: already printed by :meth:`~torchgenomics.api._results.ScanRun.summary`
#: (``> 1.10`` -> "inflated"), so the two surfaces never disagree.
_LAMBDA_GC_INFLATION = 1.10

#: Below this many post-QC variants, a scan is flagged as likely
#: underpowered/unreliable. This is a UX heuristic threshold (a "did QC eat
#: almost everything" smoke check), not a literature-cited power
#: calculation — deliberately small so it only fires on genuinely tiny
#: post-QC variant sets, never on an ordinary (even small) real-data run.
_LOW_VARIANT_COUNT = 20

#: Minority-class fraction below which a case/control-imbalance warning is
#: raised. 5% is the commonly-used rule-of-thumb below which ordinary
#: (non-Firth, non-SPA) logistic-regression p-values become unreliable due
#: to small-sample/separation bias — the reason Firth (1993)-penalized
#: likelihood and SPA-based tests (e.g. SAIGE, Zhou et al. 2018) exist.
_MINORITY_CLASS_FRACTION = 0.05

#: Fraction of missing raw-phenotype values above which a warning is raised.
_HIGH_PHENOTYPE_MISSINGNESS = 0.10

#: Mean off-diagonal GRM value above which a "closely related samples"
#: warning is raised, when a kinship/GRM matrix happens to be available.
_HIGH_MEAN_RELATEDNESS = 0.05


def _warnings(inputs: GwasInputs, result: GwasResult) -> list[str]:
    """Compute a small set of high-value, one-sentence actionable warnings.

    Called by :func:`torchgenomics.api.gwas.gwas` right after
    :func:`run_model` returns, and assigned to
    :attr:`~torchgenomics.api._results.ScanRun.warnings` — from there they
    are automatically surfaced in :attr:`~torchgenomics.api._results.ScanRun.diagnostics`
    (``result.diagnostics["warnings"]``) and in
    :meth:`~torchgenomics.api._results.ScanRun.summary`. This function
    computes **no new statistics**: every check is a threshold test on a
    quantity already present on ``inputs``/``result``.

    Checks performed (each producing at most one warning, one sentence +
    an action):

    - **Genomic inflation** — :attr:`~torchgenomics.api._results.ScanRun.lambda_gc`
      ``> `` :data:`_LAMBDA_GC_INFLATION` (1.10).
    - **Low post-QC variant count** — :attr:`~torchgenomics.api._results.ScanRun.n_variants`
      ``< `` :data:`_LOW_VARIANT_COUNT`.
    - **Case/control imbalance** — for a binary trait, the minority class's
      share of samples ``< `` :data:`_MINORITY_CLASS_FRACTION` (5%).
    - **High mean relatedness** — only if a kinship/GRM matrix is available
      on ``inputs`` (via an optional ``kinship_matrix`` attribute); the
      friendly-input layer (:mod:`torchgenomics.api._inputs`) does not
      materialize one today (the LMM route computes/consumes its GRM
      entirely inside :func:`run_model`), so this check is skipped cleanly
      — no attribute means no warning, never an ``AttributeError``.
    - **Many-missing phenotype** — the raw (pre-alignment) missing-value
      fraction of :attr:`~torchgenomics.api._inputs.GwasInputs.phenotype`
      ``> `` :data:`_HIGH_PHENOTYPE_MISSINGNESS` (10%).

    Parameters
    ----------
    inputs : GwasInputs
        The resolved, sample-aligned inputs the scan was run on (for
        ``trait_type`` and ``phenotype``).
    result : GwasResult
        The just-completed scan result (for ``lambda_gc`` and
        ``n_variants``).

    Returns
    -------
    list[str]
        Zero or more one-sentence, actionable warning strings, in the fixed
        check order listed above.
    """
    warns: list[str] = []

    if result.lambda_gc is not None and result.lambda_gc > _LAMBDA_GC_INFLATION:
        warns.append(
            f"genomic inflation detected (λ_GC={result.lambda_gc:.2f} > "
            f"{_LAMBDA_GC_INFLATION}); consider adding more principal components "
            "(pcs=) or a kinship/GRM correction (kinship='auto') to control "
            "confounding."
        )

    if result.n_variants < _LOW_VARIANT_COUNT:
        warns.append(
            f"only {result.n_variants} variant(s) survived QC; results may be "
            "unreliable/underpowered -- check your QC thresholds (qc=) and input "
            "genotype size."
        )

    if inputs.trait_type == "binary":
        counts = inputs.phenotype.dropna().value_counts()
        if len(counts) == 2:
            n_minor, n_major = int(counts.min()), int(counts.max())
            total = n_minor + n_major
            frac_minor = n_minor / total if total else 0.0
            if frac_minor < _MINORITY_CLASS_FRACTION:
                warns.append(
                    f"case/control imbalance ({n_minor} minority-class / {n_major} "
                    f"majority-class samples, {frac_minor:.1%} minority); consider a "
                    "Firth-penalized GLM (glm_scan(..., firth=True)) or a SAIGE-style "
                    "GLMM/SPA test for reliable p-values in this regime."
                )

    grm = getattr(inputs, "kinship_matrix", None)
    if grm is not None:
        try:
            arr = np.asarray(grm)
            off_diag = arr[~np.eye(arr.shape[0], dtype=bool)]
            mean_relatedness = float(off_diag.mean()) if off_diag.size else 0.0
            if mean_relatedness > _HIGH_MEAN_RELATEDNESS:
                warns.append(
                    f"high mean off-diagonal relatedness ({mean_relatedness:.3f} > "
                    f"{_HIGH_MEAN_RELATEDNESS}); the sample may include closely "
                    "related individuals -- verify kinship correction is active "
                    "(kinship='auto' or a precomputed GRM path)."
                )
        except Exception:  # noqa: BLE001 -- advisory-only; never block a result
            pass

    n_pheno = len(inputs.phenotype)
    na_frac = float(inputs.phenotype.isna().mean()) if n_pheno else 0.0
    if na_frac > _HIGH_PHENOTYPE_MISSINGNESS:
        warns.append(
            f"{na_frac:.1%} of phenotype values are missing; the scan ran on the "
            "non-missing subset only -- verify this is expected before interpreting "
            "results."
        )

    return warns
