"""``tg.gwas(...)`` — the friendly-API orchestrator (Task 5).

This module is the top of the friendly-API stack: it wires together the
input loader (:mod:`torchgenomics.api._inputs`, Task 2), the model registry
(:mod:`torchgenomics.api._registry`, Task 1), and the shared dispatch core
(:mod:`torchgenomics.api._dispatch`, Task 4) into the single call a
notebook/novice user actually types::

    import torchgenomics as tg
    result = tg.gwas(phenotype, genotype)
    print(result.summary())

It performs **no new statistics** — every number in the returned result
comes from an already-tested model path via
:func:`torchgenomics.api._dispatch.run_model`. This layer only resolves
inputs, picks (or validates) a model, prints a transparent decision log,
and shapes the output as either a single
:class:`~torchgenomics.api._results.GwasResult` (``models="auto"`` or a
bare ``str`` alias) or a :class:`GwasComparison` (``models=`` passed as a
``list``/``tuple``, by input type — even a one-element list; see
:func:`gwas`'s docstring, Fix 2).

Auto-model selection (``models="auto"``, the default) is a documented,
opt-in convenience — never a silent override. Every ``tg.gwas`` call with
``verbose=True`` (the default) prints exactly which model it picked and
why, so the choice is always visible and always overridable via
``models=<alias>`` / ``models=[<alias>, ...]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Sequence

import pandas as pd

from ._dispatch import RunOptions, _n_pcs_from_opts, run_model
from ._inputs import GwasInputs, load_inputs
from ._registry import list_models as _list_models_registry
from ._registry import resolve_model
from ._results import GwasResult, _BaseRun


def _auto_model(inputs: GwasInputs, kinship: Any) -> tuple[str, str]:
    """Pick a sensible default model alias for ``models="auto"``.

    This is a deliberately small, fully-documented decision tree — not a
    statistical recommendation engine. It only looks at two things: the
    trait type detected/declared by :func:`~torchgenomics.api._inputs.load_inputs`
    and whether the caller asked for kinship correction at all.

    **Wired-only policy**: this function must only ever return an alias that
    :func:`torchgenomics.api._dispatch.run_model` can actually execute today
    (currently ``"lmm"``, ``"glm"``, ``"blink"``, ``"farmcpu"`` — see
    :data:`torchgenomics.api._dispatch._LOWLEVEL_CLI` and the ``api_lmm`` /
    ``api_glm`` routes). ``"auto"`` is the *default* for ``models=``, so it
    must never raise ``NotImplementedError`` — a model that scientifically
    "would be nicer" but isn't wired yet (e.g. ``glmm`` for a related-sample
    binary trait) is surfaced only as a *note* in the rationale string, never
    as the actual pick. If a future edit adds a wired alias that scientifically
    dominates one of the branches below, update the branch *and* this
    docstring together.

    Decision tree
    -------------
    - **continuous** trait, kinship on  -> ``"lmm"`` (GRM mixed model;
      corrects for relatedness/population structure — the GEMMA-equivalent
      default for a quantitative trait).
    - **continuous** trait, kinship off -> ``"glm"`` (fixed-effects only;
      the caller explicitly disabled kinship correction).
    - **binary** / **categorical** trait, any kinship setting -> ``"glm"``
      (fixed-effects GLM; the only wired route for a non-Gaussian trait
      today). A mixed model (``glmm``, PQL/SAIGE-style) would be the more
      appropriate choice for *related* samples, but ``glmm`` is not yet
      wired through :func:`~torchgenomics.api._dispatch.run_model`, so
      picking it here would turn the friendly, zero-argument default path
      into a crash. The rationale string still names ``glmm`` as the
      forward-looking alternative so the caller can act on it (via the CLI
      or, once wired, ``models="glmm"``) without ``tg.gwas`` silently
      guessing wrong or blowing up.

    "Kinship on" means anything other than an explicit ``False`` / ``None``
    for the ``kinship=`` argument — a filesystem path *or* the string
    ``"auto"`` both count as "on". It only changes the *continuous* branch
    today, because the binary/categorical branch has just one wired option.

    Parameters
    ----------
    inputs : GwasInputs
        Resolved inputs from :func:`~torchgenomics.api._inputs.load_inputs`;
        only :attr:`~torchgenomics.api._inputs.GwasInputs.trait_type` is used.
    kinship : Any
        The caller's raw ``kinship=`` argument to :func:`gwas` (not yet
        normalized by :class:`~torchgenomics.api._dispatch.RunOptions`).

    Returns
    -------
    tuple[str, str]
        ``(alias, rationale)`` — the chosen registry alias (always a
        currently-wired one; see "Wired-only policy" above) and a one-line,
        human-readable explanation suitable for the decision log printed by
        :func:`gwas` (and for :attr:`GwasResult`-adjacent logging/debugging).
    """
    kinship_on = not (kinship is False or kinship is None)
    trait_type = inputs.trait_type

    if trait_type == "continuous":
        if kinship_on:
            return (
                "lmm",
                "continuous trait + kinship on -> lmm (GRM mixed model; "
                "corrects for relatedness/population structure)",
            )
        return (
            "glm",
            "continuous trait + kinship off -> glm (fixed-effects only; "
            "kinship was explicitly disabled)",
        )

    # binary / categorical: glm is the only wired route today (see
    # "Wired-only policy" above) — auto must never pick the unwired glmm.
    return (
        "glm",
        f"{trait_type} trait -> glm (fixed-effects GLM; the wired option "
        "for a non-Gaussian trait). Note: for related samples a mixed "
        "model (GLMM) is more appropriate -- run `torchgenomics "
        "glmm-scan` or pass models='glmm' once wired.",
    )


def _resolve_models(
    models: str | Sequence[str],
    inputs: GwasInputs,
    kinship: Any,
) -> tuple[list[str], bool, str | None, bool]:
    """Resolve the ``models=`` argument of :func:`gwas` to a validated alias list.

    - ``"auto"`` -> the single alias chosen by :func:`_auto_model`.
    - A bare ``str`` (not ``"auto"``) -> a one-element list, ``[models]``.
    - Any other sequence (``list``/``tuple``) -> used as-is, in order.

    Every resulting alias is validated via
    :func:`~torchgenomics.api._registry.resolve_model`, so an unknown model
    name raises exactly the same friendly ``ValueError`` (with the list of
    valid aliases) that calling :func:`~torchgenomics.api._registry.resolve_model`
    directly would.

    Return-type note (Fix 2, per spec): whether :func:`gwas` returns a single
    :class:`~torchgenomics.api._results.GwasResult` or a
    :class:`GwasComparison` is decided by the **input type** of ``models``,
    not by how many aliases it resolves to. A ``list``/``tuple`` — even a
    one-element one like ``["lmm"]`` — always means "the caller asked for
    the comparison shape" and must return a :class:`GwasComparison`; only a
    bare ``str`` (including the resolved ``"auto"`` single pick) returns a
    plain :class:`GwasResult`. This function reports that original-type flag
    back to :func:`gwas` as ``models_was_sequence`` so the count-based
    ``len(aliases) == 1`` shortcut is never used for the branch decision.

    Returns
    -------
    tuple[list[str], bool, str | None, bool]
        ``(aliases, was_auto, rationale, models_was_sequence)``. ``rationale``
        is the one-line explanation from :func:`_auto_model` when
        ``was_auto`` is ``True``, else ``None``. ``models_was_sequence`` is
        ``True`` iff the caller passed a ``list``/``tuple`` for ``models``
        (as opposed to a ``str``, including ``"auto"``).

    Raises
    ------
    ValueError
        If ``models`` resolves to an empty list, or if any alias is
        unrecognized (bubbled up from :func:`resolve_model`).
    """
    rationale: str | None = None
    models_was_sequence = not isinstance(models, str)
    if isinstance(models, str) and models == "auto":
        alias, rationale = _auto_model(inputs, kinship)
        aliases = [alias]
        was_auto = True
    elif isinstance(models, str):
        aliases = [models]
        was_auto = False
    else:
        aliases = list(models)
        was_auto = False

    if not aliases:
        raise ValueError(
            "models=[] is empty; pass a model alias (e.g. 'lmm'), a list of "
            "aliases, or 'auto' to let tg.gwas pick one. See tg.models()."
        )

    for alias in aliases:
        resolve_model(alias)  # friendly ValueError on an unknown name

    return aliases, was_auto, rationale, models_was_sequence


def _print_decision_log(
    inputs: GwasInputs,
    aliases: list[str],
    was_auto: bool,
    rationale: str | None,
    opts: RunOptions,
) -> None:
    """Print the ``tg.gwas`` decision log to stdout (``verbose=True`` only).

    Always names the trait, its detected/declared type, the sample count,
    the model(s) chosen (with each one's registry label, e.g.
    ``'lmm'/SingleTraitLMM``), the auto-selection rationale when relevant,
    the number of PCs, and the multiple-testing correction — so a novice
    user can see exactly what ran without reading any code.
    """
    n_pcs = _n_pcs_from_opts(opts)
    print(
        f"tg.gwas: trait '{inputs.trait_name}' ({inputs.trait_type}), "
        f"n={inputs.n_samples} samples"
    )
    if was_auto:
        spec = resolve_model(aliases[0])
        print(
            f"  model: auto -> '{aliases[0]}' / {spec.label}  ({rationale})"
        )
    else:
        labels = ", ".join(f"'{a}'/{resolve_model(a).label}" for a in aliases)
        print(f"  model(s): {labels}  (user-selected)")
    print(
        f"  kinship={opts.kinship!r}  pcs={n_pcs}  correction={opts.correction!r}"
    )


@dataclass
class GwasComparison(_BaseRun):
    """Multi-model result of ``tg.gwas(..., models=[...])``.

    Returned instead of a bare :class:`~torchgenomics.api._results.GwasResult`
    whenever :func:`gwas` is called with ``models=`` passed as a
    ``list``/``tuple`` — **by input type, not by resolved count** (Fix 2):
    even a *one-element* list like ``models=["lmm"]`` returns a
    :class:`GwasComparison`, not a bare
    :class:`~torchgenomics.api._results.GwasResult`. Only ``models="auto"``
    or an explicit single ``str`` alias (e.g. ``models="lmm"``) return a
    plain :class:`~torchgenomics.api._results.GwasResult` instead. Bundles
    one result per requested model plus comparison-level conveniences:

    - :attr:`results` — ``{alias: GwasResult}``, in request order.
    - :meth:`summary` — plain-language header plus a per-model
      λ_GC / #significant-hits table, plus a pairwise top-hit overlap count.
    - :meth:`report` — writes each model's own :meth:`~torchgenomics.api._results.ScanRun.report`
      into a same-named subfolder of ``dir``, plus a top-level
      ``comparison_summary.txt``.
    - :meth:`to_dict` / :meth:`to_json` (inherited from
      :class:`~torchgenomics.api._results._BaseRun`) — JSON-safe, with each
      per-model :class:`GwasResult` recursively serialized.

    Attributes
    ----------
    results : dict[str, GwasResult]
        Model alias -> its scan result, in the order the models were
        requested.
    trait_name : str
        Name of the scanned trait (used in :meth:`summary`'s header).
    trait_type : str
        Trait type (``"continuous"``/``"binary"``/``"categorical"``), also
        used in :meth:`summary`'s header.
    runtime_s : float
        (Inherited from :class:`~torchgenomics.api._results._BaseRun`, Fix
        3.) Total wall-clock time for the *whole comparison*, computed by
        :func:`gwas` as the **sum** of each per-model
        :attr:`~torchgenomics.api._results.ScanRun.runtime_s` (not the max)
        — i.e. the cost of running every requested model, since they run
        sequentially. Left at its ``0.0`` default only when
        :class:`GwasComparison` is constructed directly rather than via
        :func:`gwas`.
    """

    results: dict[str, GwasResult] = field(default_factory=dict)
    trait_name: str = ""
    trait_type: str = ""

    _kind: ClassVar[str] = "gwas_comparison"

    def __repr__(self) -> str:
        return f"GwasComparison(models={list(self.results)}, trait={self.trait_name!r})"

    def summary(self) -> str:
        """Plain-language comparison report: header + per-model table + overlap.

        The table has one row per requested model (alias, test, n_variants,
        n_significant, λ_GC); overlap is reported as the size of the
        pairwise intersection of each pair of models' :attr:`~torchgenomics.api._results.ScanRun.top_hits`
        SNP sets (only when a ``"SNP"`` column is present — always true for
        results produced by :func:`torchgenomics.api._dispatch.run_model`).
        """
        trait_label = (
            f"trait '{self.trait_name}' ({self.trait_type})" if self.trait_name else "trait"
        )
        lines = [
            f"GWAS model comparison — {trait_label}",
            f"Models compared: {', '.join(self.results)}",
            "",
        ]

        header = f"{'model':<12}{'test':<10}{'n_variants':>12}{'n_sig':>8}{'lambda_gc':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for alias, res in self.results.items():
            lgc = f"{res.lambda_gc:.3f}" if res.lambda_gc is not None else "n/a"
            lines.append(
                f"{alias:<12}{res.test or '(n/a)':<10}{res.n_variants:>12}"
                f"{res.n_significant:>8}{lgc:>12}"
            )

        top_sets = {
            alias: set(res.top_hits["SNP"])
            for alias, res in self.results.items()
            if "SNP" in res.top_hits.columns
        }
        aliases = list(top_sets)
        if len(aliases) >= 2:
            lines.append("")
            lines.append("Top-hit overlap (pairwise, among top_hits shown):")
            for i in range(len(aliases)):
                for j in range(i + 1, len(aliases)):
                    a, b = aliases[i], aliases[j]
                    shared = top_sets[a] & top_sets[b]
                    lines.append(f"  {a} vs {b}: {len(shared)} shared SNP(s)")

        lines.append("")
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        lines.append("Next: call .report(dir) to save each model's report + this comparison")

        return "\n".join(lines)

    def report(self, dir: str | Path) -> Path:
        """Write a comparison report folder: one subfolder per model + a top-level summary.

        Always writes ``comparison_summary.txt`` (the :meth:`summary` text)
        at the top of ``dir``. For each model, calls
        ``result.report(dir / alias)`` inside its own ``try/except`` (mirroring
        :meth:`~torchgenomics.api._results.ScanRun.report`'s best-effort
        philosophy) so one model's plotting failure (e.g. a headless
        environment, or an unexpected ``top_hits`` schema) never blocks the
        others' reports or the comparison summary.

        Parameters
        ----------
        dir : str | pathlib.Path
            Directory to create (including parents) and populate.

        Returns
        -------
        pathlib.Path
            The report directory (``dir``, resolved to a :class:`Path`).
        """
        out_dir = Path(dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "comparison_summary.txt").write_text(self.summary())

        for alias, res in self.results.items():
            try:
                res.report(out_dir / alias)
            except Exception:  # noqa: BLE001
                pass

        return out_dir


def gwas(
    phenotype: Any,
    genotype: Any,
    *,
    covariates: Any = None,
    kinship: Any = "auto",
    pcs: Any = "auto",
    trait: str | None = None,
    trait_type: str | None = None,
    models: str | Sequence[str] = "auto",
    qc: bool = True,
    correction: str = "bh",
    output: str | Path | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> "GwasResult | GwasComparison":
    """Run a GWAS scan end-to-end from whatever inputs a notebook user has on hand.

    This is the single entry point of the friendly API: it resolves and
    sample-aligns ``phenotype``/``genotype``/``covariates``
    (:func:`torchgenomics.api._inputs.load_inputs`), picks a model (or
    validates the caller's choice) via the registry
    (:func:`torchgenomics.api._registry.resolve_model`), runs it through the
    shared dispatch core (:func:`torchgenomics.api._dispatch.run_model`),
    and returns a uniform, notebook-friendly result. No model statistics
    are computed in this function — it is pure orchestration over
    already-tested code paths.

    Parameters
    ----------
    phenotype : str | pathlib.Path | pandas.Series | pandas.DataFrame | numpy.ndarray
        Phenotype source; see :func:`torchgenomics.api._inputs.load_inputs`
        for every accepted shape.
    genotype : str | pathlib.Path | numpy.ndarray | pandas.DataFrame | GenotypeReader
        Genotype source; see :func:`~torchgenomics.api._inputs.load_inputs`.
    covariates : str | pathlib.Path | pandas.DataFrame | numpy.ndarray | None, default None
        Optional covariates, aligned to the same samples as ``phenotype``.
    kinship : str | bool | None, default "auto"
        GRM / kinship policy. ``"auto"`` computes a streaming VanRaden GRM
        when a mixed model is used; a path loads a pre-computed GRM;
        ``False``/``None`` disables kinship correction for models that
        support running without it (drives ``models="auto"`` toward
        ``"glm"`` instead of ``"lmm"`` for a continuous trait — see
        :func:`_auto_model`; it does not change the binary/categorical pick,
        which is always ``"glm"`` today).
        Passed straight through to :class:`~torchgenomics.api._dispatch.RunOptions`.
    pcs : str | int | bool | None, default "auto"
        Number of genotype principal components to add as covariates.
        Resolved by :func:`torchgenomics.api._dispatch._n_pcs_from_opts`
        (``"auto"`` currently means 0) — passed through unmodified here so
        that resolution happens in exactly one place.
    trait : str | None, default None
        Column name to select when ``phenotype`` is a DataFrame or path.
    trait_type : str | None, default None
        Force ``"continuous"``/``"binary"``/``"categorical"`` instead of
        letting :func:`~torchgenomics.api._inputs.detect_trait_type` infer
        it. Also feeds ``models="auto"``'s decision (:func:`_auto_model`).
    models : str | list[str], default "auto"
        ``"auto"`` picks exactly one model via :func:`_auto_model` (printed
        and always overridable — never a silent choice) and returns a plain
        :class:`~torchgenomics.api._results.GwasResult`. A single alias
        (``str``, e.g. ``"lmm"``) or GAPIT-name synonym runs that one model
        and likewise returns a bare :class:`~torchgenomics.api._results.GwasResult`.
        **The return type is decided by the type you pass, not the count**
        (Fix 2): any ``list``/``tuple`` — including a *one-element* list
        like ``["lmm"]`` — runs every alias in it and always returns a
        :class:`GwasComparison`, never a bare
        :class:`~torchgenomics.api._results.GwasResult`. Pass a bare
        ``str`` when you want a single result; wrap it in a list only when
        you want the comparison shape (``.results``, comparison
        :meth:`GwasComparison.summary`/:meth:`GwasComparison.report`), even
        for one model. Every alias is validated via
        :func:`torchgenomics.api._registry.resolve_model` (see
        :func:`tg.models() <models>` for the full registry).
    qc : bool, default True
        Per-variant QC (MAF / missingness / Hardy-Weinberg). See
        :class:`torchgenomics.api._dispatch.RunOptions`.
    correction : str, default "bh"
        Multiple-testing correction (``"bh"``, ``"bonferroni"``, ``"holm"``,
        ``"by"``, ``"storey"``, ``"none"``, ...).
    output : str | pathlib.Path | None, default None
        If given: for a single model, the association table is written
        there directly and :meth:`~torchgenomics.api._results.ScanRun.report`
        is additionally called on the same directory (adds
        ``summary.txt``/plots on top of the raw results table). For a
        model comparison, ``output`` becomes the comparison report
        directory (:meth:`GwasComparison.report`), with each model's own
        report in a same-named subfolder. ``None`` (default) writes to a
        process-local temp directory that the returned result still
        references via ``output_files`` / per-model ``output_files``.
    device : str | None, default None
        ``"cpu"`` / ``"cuda"`` / ``"auto"`` / ``None`` (resolves to
        ``"auto"``); see :class:`torchgenomics.api._dispatch.RunOptions`.
    verbose : bool, default True
        Print the decision log (trait name + type, sample count, the
        model(s) chosen and why, #PCs, and the correction method) to
        stdout before running. Set ``False`` for silent/scripted use (e.g.
        inside a loop over many traits).

    Returns
    -------
    GwasResult | GwasComparison
        A single :class:`~torchgenomics.api._results.GwasResult` when
        ``models`` was passed as a ``str`` (``"auto"`` or an explicit single
        alias), else a :class:`GwasComparison` bundling one
        :class:`~torchgenomics.api._results.GwasResult` per requested model
        when ``models`` was passed as a ``list``/``tuple`` — **by input
        type, not by resolved count**; a one-element list still returns a
        :class:`GwasComparison` (see the ``models`` parameter above, Fix 2).

    Raises
    ------
    ValueError
        On any input mismatch (bubbled up from
        :func:`~torchgenomics.api._inputs.load_inputs`) or unknown/invalid
        model alias, option, or empty ``models=[]`` (bubbled up from
        :func:`_resolve_models` / :func:`~torchgenomics.api._registry.resolve_model`
        / :func:`torchgenomics.api._dispatch.run_model`).
    NotImplementedError
        For registry models that :func:`torchgenomics.api._dispatch.run_model`
        does not yet wire through the friendly API (e.g. ``gxe``, ``set``,
        ``mvlmm``, ``bayes``, and currently ``glmm`` — see that function's
        docstring); the message points at the equivalent
        ``torchgenomics <alias>-scan`` CLI subcommand or the low-level API.

    Examples
    --------
    >>> r = tg.gwas(y, G)                      # doctest: +SKIP
    >>> print(r.summary())                     # doctest: +SKIP
    >>> cmp = tg.gwas(y, G, models=["lmm", "blink"])  # doctest: +SKIP
    >>> print(cmp.summary())                   # doctest: +SKIP
    """
    inputs = load_inputs(phenotype, genotype, covariates, trait, trait_type)
    aliases, was_auto, rationale, models_was_sequence = _resolve_models(models, inputs, kinship)
    # Fix 2: the single-vs-comparison branch is decided by the *input type*
    # of `models` (str -> GwasResult, list/tuple -> GwasComparison), never
    # by how many aliases it happened to resolve to — so models=["lmm"]
    # (a one-element list) still returns a GwasComparison. See
    # _resolve_models' docstring and the `models`/Returns sections above.
    single = not models_was_sequence

    opts = RunOptions(
        kinship=kinship,
        pcs=pcs,
        qc=qc,
        correction=correction,
        device=device,
        output=output if single else None,
        verbose=verbose,
    )

    if verbose:
        _print_decision_log(inputs, aliases, was_auto, rationale, opts)

    results = {alias: run_model(alias, inputs, opts) for alias in aliases}

    if single:
        result = results[aliases[0]]
        if output is not None:
            result.report(output)
        return result

    # Fix 3: runtime_s for the comparison is the SUM of each per-model
    # GwasResult.runtime_s (documented on GwasComparison.runtime_s below) —
    # this is the wall-clock cost of the tg.gwas(models=[...]) call as a
    # whole, not just whichever model happened to run last/longest.
    comparison = GwasComparison(
        results=results,
        trait_name=inputs.trait_name,
        trait_type=inputs.trait_type,
        runtime_s=sum(r.runtime_s for r in results.values()),
    )
    if output is not None:
        comparison.report(output)
    return comparison


def models() -> pd.DataFrame:
    """List every model available to ``tg.gwas(models=...)`` as a tidy DataFrame.

    Thin, documented re-export of
    :func:`torchgenomics.api._registry.list_models` under the friendly-API
    name ``tg.models()`` (a *function*, distinct from the
    :mod:`torchgenomics.models` *subpackage* — both remain independently
    usable: ``tg.models()`` calls the registry, while
    ``from torchgenomics.models import SingleTraitLMM`` /
    ``import torchgenomics.models`` continue to resolve the subpackage, since
    Python's ``from package.sub import name`` / cached ``import package.sub``
    forms look the submodule up directly in ``sys.modules`` rather than via
    this shadowed top-level attribute).

    Returns
    -------
    pandas.DataFrame
        Columns: ``alias``, ``label``, ``trait_types``, ``description``.
        See :func:`torchgenomics.api._registry.list_models`.

    Examples
    --------
    >>> import torchgenomics as tg
    >>> "lmm" in set(tg.models()["alias"])
    True
    """
    return _list_models_registry()
