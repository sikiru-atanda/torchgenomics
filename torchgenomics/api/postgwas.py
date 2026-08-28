"""Post-GWAS tier-1 API: :func:`clump`, :func:`meta`, :func:`mr`, :func:`smr`,
:func:`mr_mega`."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from ._decorator import tool
from ._helpers import ProgressCallback, emit_progress, resolve_output_dir, timed
from ._results import (
    ClumpRun,
    ColocRun,
    EnrichmentRun,
    HessRun,
    MetaRun,
    MRMegaRun,
    MRRun,
    PowerRun,
    SMRRun,
    WinnersCurseRun,
)


@tool(
    name="tg_clump",
    title="LD clumping for independent loci",
    description=(
        "LD-clump GWAS summary statistics to identify independent index variants. "
        "Operates per-chromosome (an index SNP on chr 1 cannot clump a SNP on chr 2); "
        "streams the genotype matrix to bound memory. Returns ClumpRun with a "
        "DataFrame of clumps (each row = one index variant with its tag SNPs)."
    ),
    long_running=False,
    category="postgwas",
    tags=["clumping", "independence", "ld", "postgwas"],
)
def clump(
    sumstats: str | Path,
    genotype: str | Path,
    *,
    output: str | Path | None = None,
    p_threshold: float = 5e-8,
    r2: float = 0.1,
    window_kb: float = 250.0,
    progress_callback: ProgressCallback | None = None,
) -> ClumpRun:
    """LD-clump GWAS sumstats against an LD-reference genotype panel.

    Parameters
    ----------
    sumstats, genotype
        Paths to the GWAS summary statistics and the LD-reference genotype panel.
    output
        Output directory or file prefix. If None, auto-creates a unique run dir.
    p_threshold
        P-value cutoff for index variants (default ``5e-8``).
    r2
        r² ceiling for clumping (default ``0.1``).
    window_kb
        Clumping window in kilobases (default ``250``).
    """
    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_clump")
    output_prefix = output_dir / "clumps"

    emit_progress(progress_callback, 0.0, "clumping")

    with timed() as elapsed:
        from ..cli import _run_clump

        info = _run_clump(
            sumstats_path=str(sumstats),
            genotype_path=str(genotype),
            output_prefix=str(output_prefix),
            r2=r2,
            p_threshold=p_threshold,
            window_kb=window_kb,
        )

        out_tsv = Path(info["output_path"])
        output_files: dict[str, Path] = {}
        if out_tsv.exists():
            output_files["tsv"] = out_tsv
            df = pd.read_csv(out_tsv, sep="\t")
        else:
            df = pd.DataFrame()

        # Count input variants for context
        n_input = 0
        try:
            n_input = int(sum(1 for _ in open(sumstats)) - 1)  # minus header
        except Exception:  # noqa: BLE001
            pass

        return ClumpRun(
            runtime_s=elapsed(),
            output_files=output_files,
            n_input_variants=n_input,
            n_clumps=int(info["n_clumps"]),
            n_index_variants=int(info["n_clumps"]),
            p_threshold=p_threshold,
            r2_threshold=r2,
            clumps=df,
        )


@tool(
    name="tg_meta",
    title="Meta-analysis across GWAS sumstats",
    description=(
        "Combine GWAS summary statistics across studies (fixed effect / random "
        "effect / Han-Eskin / Stouffer sample-size). Inputs are aligned by SNP; "
        "writes a single TSV with per-variant meta beta/se/p plus heterogeneity. "
        "Returns MetaRun with the results DataFrame inline and a summary I²."
    ),
    long_running=False,
    category="postgwas",
    tags=["meta-analysis", "ivw", "stouffer", "random-effect", "han-eskin"],
)
def meta(
    inputs: list[str | Path],
    *,
    output: str | Path | None = None,
    method: Literal["fixed", "random", "han_eskin", "stouffer"] = "fixed",
    significance_threshold: float = 5e-8,
) -> MetaRun:
    """Meta-analyze GWAS summary statistics across studies.

    Parameters
    ----------
    inputs
        List of paths to sumstats files (at least 2).
    method
        ``"fixed"`` (IVW), ``"random"`` (DerSimonian-Laird), ``"han_eskin"``
        (RE2 with heterogeneity-aware test), or ``"stouffer"`` (sample-size
        weighted z-combination).
    """
    if len(inputs) < 2:
        raise ValueError(f"meta requires at least 2 input sumstats; got {len(inputs)}")

    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_meta")
    output_prefix = output_dir / "meta"

    with timed() as elapsed:
        from ..cli import _run_meta

        info = _run_meta(
            inputs=[str(p) for p in inputs],
            output_prefix=str(output_prefix),
            method=method,
        )

        meta_tsv = Path(info["output_path"])
        output_files: dict[str, Path] = {}
        if meta_tsv.exists():
            output_files["tsv"] = meta_tsv
            df = pd.read_csv(meta_tsv, sep="\t")
        else:
            df = pd.DataFrame()

        n_sig = int((df["p_meta"] < significance_threshold).sum()) if "p_meta" in df.columns else 0
        het_i2 = float(df["i2"].median()) if "i2" in df.columns and not df.empty else None

        return MetaRun(
            runtime_s=elapsed(),
            output_files=output_files,
            method=method,
            n_studies=len(inputs),
            n_variants=int(len(df)),
            n_significant=n_sig,
            results=df,
            heterogeneity_i2_median=het_i2,
        )


_MR_METHODS = ("ivw", "egger", "weighted_median", "presso", "all")

#: `postgwas.mr_presso`'s `MRResult.method` field is literally `"mr_presso"`
#: (see `torchgenomics/postgwas/_mr.py`); normalize it to the same short
#: label used everywhere else in this API (`method="presso"` request, CLI
#: flag, R wrapper) so the results table is self-consistent.
_MR_RESULT_LABEL = {"mr_presso": "presso"}


def mr(
    exposure: str | Path,
    outcome: str | Path,
    *,
    method: Literal["ivw", "egger", "weighted_median", "presso", "all"] = "ivw",
    output: str | Path | None = None,
    n_boot: int = 1000,
    n_perm: int = 1000,
    seed: int = 42,
    sep: str = "\t",
) -> MRRun:
    """Two-sample Mendelian randomization from exposure + outcome sumstats.

    Loads both sumstats files, runs the requested estimator(s), and returns a
    :class:`~torchgenomics.api.MRRun` with a per-method results table.

    Parameters
    ----------
    exposure, outcome
        Paths to instrument sumstats TSVs (columns chr/pos/snp/a1/a2/beta/se/p,
        harmonized to the same effect allele) — the shared instruments define
        the MR panel.
    method
        One of ``"ivw"``, ``"egger"``, ``"weighted_median"``, ``"presso"``, or
        ``"all"`` (runs the full sensitivity panel, one row per method).
    output
        Optional path to write the results TSV.
    n_boot, n_perm, seed
        Bootstrap / permutation / seed controls for weighted-median and
        MR-PRESSO.
    sep
        Field delimiter for the sumstats files (default tab).

    Returns
    -------
    MRRun
    """
    from ..postgwas import (
        load_sumstats,
        mr_all,
        mr_egger,
        mr_ivw,
        mr_presso,
        mr_weighted_median,
    )

    m = str(method).lower()
    if m not in _MR_METHODS:
        raise ValueError(
            f"Unknown MR method '{method}'. Valid: {', '.join(_MR_METHODS)}."
        )

    exp = load_sumstats(str(exposure), sep=sep)
    out = load_sumstats(str(outcome), sep=sep)

    with timed() as elapsed:
        if m == "all":
            results = mr_all(exp, out, n_boot=n_boot, n_perm=n_perm, seed=seed)
        elif m == "ivw":
            results = [mr_ivw(exp, out)]
        elif m == "egger":
            results = [mr_egger(exp, out)]
        elif m == "weighted_median":
            results = [mr_weighted_median(exp, out, n_boot=n_boot, seed=seed)]
        else:  # presso
            results = [mr_presso(exp, out, n_perm=n_perm, seed=seed)]

        rows = [
            {
                "method": _MR_RESULT_LABEL.get(r.method, r.method),
                "beta": r.beta_hat,
                "se": r.se,
                "pval": r.p_value,
                "n_instruments": r.n_instruments,
                "egger_intercept": (
                    r.intercept if r.method == "egger" else float("nan")
                ),
                "egger_intercept_p": (
                    r.intercept_p if r.method == "egger" else float("nan")
                ),
                "n_outliers": (
                    r.n_outliers if r.method == "mr_presso" else float("nan")
                ),
                "beta_corrected": (
                    r.beta_corrected if r.method == "mr_presso" else float("nan")
                ),
                "pval_corrected": (
                    r.p_corrected if r.method == "mr_presso" else float("nan")
                ),
            }
            for r in results
        ]
        df = pd.DataFrame(rows)

        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))

        primary = results[0]
        return MRRun(
            runtime_s=elapsed(),
            output_files=output_files,
            method=m,
            n_instruments=int(primary.n_instruments),
            primary_beta=float(primary.beta_hat),
            primary_p=float(primary.p_value),
            results=df,
        )


def coloc(
    sumstats,
    sumstats2: str | Path | None = None,
    *,
    method: Literal["pairwise", "hyprcoloc"] = "pairwise",
    output: str | Path | None = None,
    prior_1: float = 1e-4,
    prior_2: float = 1e-4,
    prior_12: float = 1e-5,
    trait_names: list[str] | None = None,
    sep: str = "\t",
) -> ColocRun:
    """Colocalization of two (pairwise) or ≥2 (hyprcoloc) GWAS sumstats.

    Parameters
    ----------
    sumstats
        For ``method="pairwise"``: a path to the first sumstats TSV. For
        ``method="hyprcoloc"``: a list of ≥2 sumstats paths.
    sumstats2
        The second sumstats path for ``method="pairwise"`` (required there;
        must be ``None`` for hyprcoloc).
    method
        ``"pairwise"`` (Giambartolomei 2-trait, PP.H0-H4) or ``"hyprcoloc"``
        (N-trait).
    output
        Optional path to write a posteriors TSV.
    prior_1, prior_2, prior_12
        Coloc priors, forwarded to :func:`torchgenomics.postgwas.coloc_pairwise`
        for ``method="pairwise"``. For ``method="hyprcoloc"`` only ``prior_1``
        is forwarded (hyprcoloc's own ``prior_2``/``prior_w`` defaults apply);
        ``prior_12`` has no hyprcoloc analogue and is ignored there.
    trait_names
        Optional trait labels, forwarded to hyprcoloc only.
    sep
        Field delimiter for the sumstats files (default tab).

    Returns
    -------
    ColocRun
    """
    from ..postgwas import coloc_pairwise, hyprcoloc, load_sumstats

    m = str(method).lower()
    if m not in ("pairwise", "hyprcoloc"):
        raise ValueError(
            f"Unknown coloc method '{method}'. Valid: pairwise, hyprcoloc."
        )

    with timed() as elapsed:
        if m == "pairwise":
            if sumstats2 is None:
                raise ValueError(
                    "coloc method='pairwise' needs two sumstats; pass "
                    "sumstats2=<path> (or use method='hyprcoloc' with a list)."
                )
            ss1 = load_sumstats(str(sumstats), sep=sep)
            ss2 = load_sumstats(str(sumstats2), sep=sep)
            res = coloc_pairwise(
                ss1, ss2, prior_1=prior_1, prior_2=prior_2, prior_12=prior_12
            )
            pp = {
                "h0": float(res.pp_h0),
                "h1": float(res.pp_h1),
                "h2": float(res.pp_h2),
                "h3": float(res.pp_h3),
                "h4": float(res.pp_h4),
            }
            table = pd.DataFrame([pp])
            run = ColocRun(
                runtime_s=elapsed(),
                method="pairwise",
                pp=pp,
                candidate_snp=int(res.candidate_snp),
                headline=pp["h4"],
                table=table,
            )
        else:  # hyprcoloc
            if sumstats2 is not None:
                raise ValueError(
                    "coloc method='hyprcoloc' takes a list in `sumstats`; "
                    "do not pass sumstats2."
                )
            paths = (
                list(sumstats)
                if not isinstance(sumstats, (str, Path))
                else [sumstats]
            )
            if len(paths) < 2:
                raise ValueError(
                    "coloc method='hyprcoloc' needs >=2 sumstats paths."
                )
            ss_list = [load_sumstats(str(p), sep=sep) for p in paths]
            res = hyprcoloc(
                ss_list, prior_1=prior_1, trait_names=trait_names
            )
            pp = {
                "all_colocalize": float(res.pp_all_colocalize),
                "null": float(res.pp_null),
            }
            table = pd.DataFrame(
                [
                    {
                        "pp_all_colocalize": res.pp_all_colocalize,
                        "pp_null": res.pp_null,
                        "best_cluster": str(res.best_cluster),
                        "best_cluster_posterior": res.best_cluster_posterior,
                    }
                ]
            )
            run = ColocRun(
                runtime_s=elapsed(),
                method="hyprcoloc",
                pp=pp,
                candidate_snp=int(res.candidate_snp),
                headline=float(res.pp_all_colocalize),
                table=table,
            )

        output_files: dict[str, Path] = {}
        if output is not None:
            run.table.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
            run.output_files = output_files
        return run


def _load_gene_map(gene_map, sep="\t") -> dict:
    """Load a long-format gene->cis-SNP map (columns ``gene``, ``snp``) into a dict.

    Accepts a path (str/Path) to a TSV with ``gene`` and ``snp`` columns (one row
    per gene–SNP pair) or an already-built ``dict[str, list[str]]`` (returned
    as-is). Raises a friendly ``ValueError`` if the columns are missing.
    """
    import pandas as pd
    if isinstance(gene_map, dict):
        return gene_map
    df = pd.read_csv(str(gene_map), sep=sep)
    if "gene" not in df.columns or "snp" not in df.columns:
        raise ValueError(
            f"gene-map file must have 'gene' and 'snp' columns (one row per "
            f"gene–cis-SNP pair); got columns {list(df.columns)}."
        )
    out: dict[str, list[str]] = {}
    for gene, snp in zip(df["gene"].astype(str), df["snp"].astype(str)):
        out.setdefault(gene, [])
        if snp not in out[gene]:
            out[gene].append(snp)
    return out


def _load_gene_annotation(gene_annotation, sep="\t"):
    """Load a gene annotation TSV (columns gene, chr, start, end) into four lists."""
    df = pd.read_csv(str(gene_annotation), sep=sep, keep_default_na=False)
    need = {"gene", "chr", "start", "end"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"gene-annotation file must have columns {sorted(need)}; got {list(df.columns)}."
        )
    return (
        [str(x) for x in df["gene"]],
        [str(x) for x in df["chr"]],
        [int(x) for x in df["start"]],
        [int(x) for x in df["end"]],
    )


def _load_gene_sets(gene_sets, fmt="auto", sep="\t") -> dict:
    """Load a gene-set mapping from a long TSV (set,gene) or a .gmt file."""
    if isinstance(gene_sets, dict):
        return gene_sets
    path = str(gene_sets)
    if fmt == "auto":
        fmt = "gmt" if path.lower().endswith(".gmt") else "tsv"
    out: dict[str, list[str]] = {}
    if fmt == "gmt":
        with open(path) as fh:
            for line in fh:
                parts = [p for p in line.rstrip("\n").split("\t") if p != ""]
                if len(parts) < 3:
                    continue  # need set-name, description, >=1 gene
                name, genes = parts[0], parts[2:]
                out.setdefault(name, [])
                for g in genes:
                    if g not in out[name]:
                        out[name].append(g)
    elif fmt == "tsv":
        # keep_default_na=False: gene-set / gene identifiers are free-text
        # labels, not numeric data — a set literally named "null" or "NA"
        # must not collide with pandas' default missing-value sentinels.
        df = pd.read_csv(path, sep=sep, keep_default_na=False)
        if "set" not in df.columns or "gene" not in df.columns:
            raise ValueError(
                f"gene-sets TSV must have 'set' and 'gene' columns; got {list(df.columns)}."
            )
        for s, g in zip(df["set"].astype(str), df["gene"].astype(str)):
            out.setdefault(s, [])
            if g not in out[s]:
                out[s].append(g)
    else:
        raise ValueError(f"gene_sets_format must be auto/tsv/gmt; got {fmt!r}.")
    if not out:
        raise ValueError(f"no gene sets parsed from {path!r}.")
    return out


def smr(gwas, eqtl, gene_map, *, output=None, eqtl_p_threshold=5e-8,
        smr_p_threshold=0.05, heidi_p_threshold=0.05, heidi_max_snps=20,
        sep="\t") -> "SMRRun":
    """SMR + HEIDI: test whether a gene's expression mediates the GWAS signal.

    Parameters
    ----------
    gwas, eqtl
        Paths to GWAS and eQTL summary-statistics TSVs (columns chr/pos/snp/
        a1/a2/beta/se/p). The eQTL file holds all cis-SNPs across genes.
    gene_map
        Path to a long-format TSV (``gene``, ``snp`` columns; one row per
        gene–cis-SNP pair) or a ``dict[str, list[str]]``.
    output
        Optional path to write the per-gene results TSV.
    eqtl_p_threshold, smr_p_threshold, heidi_p_threshold, heidi_max_snps
        SMR / HEIDI thresholds (see :func:`torchgenomics.postgwas.smr_heidi`).

    Returns
    -------
    SMRRun
    """
    from ..postgwas import load_sumstats, smr_heidi

    g = load_sumstats(str(gwas), sep=sep)
    e = load_sumstats(str(eqtl), sep=sep)
    gm = _load_gene_map(gene_map, sep=sep)

    with timed() as elapsed:
        summ = smr_heidi(g, e, gm, eqtl_p_threshold=eqtl_p_threshold,
                         smr_p_threshold=smr_p_threshold,
                         heidi_p_threshold=heidi_p_threshold,
                         heidi_max_snps=heidi_max_snps)
        rows = [{
            "gene_id": r.gene_id, "probe_snp": r.probe_snp,
            "beta_smr": r.beta_smr, "se_smr": r.se_smr, "p_smr": r.p_smr,
            "chi2_smr": r.chi2_smr, "beta_gwas": r.beta_gwas, "beta_eqtl": r.beta_eqtl,
            "p_heidi": r.p_heidi, "n_heidi_snps": r.n_heidi_snps, "heidi_stat": r.heidi_stat,
        } for r in summ.results]
        df = pd.DataFrame(rows)
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        return SMRRun(
            runtime_s=elapsed(), output_files=output_files, results=df,
            n_genes_tested=int(summ.n_genes_tested),
            n_significant_smr=int(summ.n_significant_smr),
            n_pass_heidi=int(summ.n_pass_heidi),
        )


def mr_mega(sumstats, *, output=None, n_axes=4, random_effects=False,
            sep="\t") -> "MRMegaRun":
    """MR-MEGA: multi-ancestry meta-regression of SNP effects on ancestry axes.

    Parameters
    ----------
    sumstats
        A list of >= ``n_axes + 2`` sumstats TSV paths, one per ancestry.
    output
        Optional path to write the per-SNP results TSV.
    n_axes
        Number of ancestry principal axes to fit.
    random_effects
        Whether to use the random-effects variant.

    Returns
    -------
    MRMegaRun
    """
    from ..postgwas import load_sumstats, mr_mega as _mr_mega

    if isinstance(sumstats, (str, Path)):
        raise ValueError(
            "mr_mega needs a list of sumstats paths (one per ancestry), not a "
            "single path."
        )
    paths = list(sumstats)
    need = int(n_axes) + 2
    if len(paths) < need:
        raise ValueError(
            f"MR-MEGA with n_axes={n_axes} needs at least {need} populations "
            f"(sumstats files); got {len(paths)}."
        )
    ss_list = [load_sumstats(str(p), sep=sep) for p in paths]

    with timed() as elapsed:
        res = _mr_mega(ss_list, n_axes=int(n_axes), random_effects=bool(random_effects))

        m = int(res.n_snps)

        def _col(x):
            # log10_bf / posterior_effect are MANTRA-only fields and are
            # None for mr_mega's MultiAncestryResult; fill with NaN so the
            # results DataFrame keeps a uniform per-SNP row shape.
            if x is None:
                return [float("nan")] * m
            try:
                return [float(v) for v in x.tolist()]
            except Exception:
                return list(x)

        df = pd.DataFrame({
            "beta_meta": _col(res.beta_meta), "se_meta": _col(res.se_meta),
            "p_meta": _col(res.p_meta), "p_heterogeneity": _col(res.p_heterogeneity),
            "p_ancestry": _col(res.p_ancestry), "p_residual": _col(res.p_residual),
            "log10_bf": _col(res.log10_bf), "posterior_effect": _col(res.posterior_effect),
        })
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        pmeta = df["p_meta"].dropna()
        return MRMegaRun(
            runtime_s=elapsed(), output_files=output_files, results=df,
            method=str(res.method), n_axes=int(res.n_axes),
            n_populations=int(res.n_populations), n_snps=int(res.n_snps),
            min_p_meta=float(pmeta.min()) if len(pmeta) else None,
        )


def power(gwas, *, n=None, alpha=5e-8, target_power=0.8,
          power_curve=False, af_grid=None, output=None, sep="\t") -> "PowerRun":
    """Per-variant GWAS detection power, NCP, min-detectable-beta and required-N.

    Loads ``af``/``beta`` from a sumstats TSV; ``n`` defaults to the median of
    the finite ``n`` column. With ``power_curve=True`` also returns the
    min-detectable-|beta| envelope over an allele-frequency grid.
    """
    import torch
    from ..postgwas import load_sumstats, gwas_power, required_n
    from ..postgwas import power_curve as _power_curve  # avoid shadowing the bool param

    ss = load_sumstats(str(gwas), sep=sep)
    if ss.af is None:
        raise ValueError("power needs an allele-frequency column ('af') in the sumstats.")
    if n is None:
        finite = ss.n[torch.isfinite(ss.n)]
        if finite.numel() == 0:
            raise ValueError("power needs a sample size: pass n=... (no usable 'n' column).")
        n = float(finite.median().item())
    n = float(n)

    with timed() as elapsed:
        pr = gwas_power(n, ss.af, ss.beta, alpha=alpha, target_power=target_power)
        req = required_n(ss.af, ss.beta, alpha=alpha, target_power=target_power)
        df = pd.DataFrame({
            "snp": ss.snp,
            "af": [float(v) for v in ss.af.tolist()],
            "beta": [float(v) for v in ss.beta.tolist()],
            "power": [float(v) for v in pr.power.tolist()],
            "ncp": [float(v) for v in pr.ncp.tolist()],
            "min_detectable_beta": [float(v) for v in pr.min_detectable_beta.tolist()],
            "required_n": [float(v) for v in req.tolist()],
        })
        curve = None
        if power_curve:
            if af_grid is None:
                grid = torch.linspace(0.01, 0.5, 50, dtype=torch.float64)
            elif isinstance(af_grid, str):
                grid = torch.tensor([float(x) for x in af_grid.split(",") if x.strip()],
                                    dtype=torch.float64)
            else:
                grid = torch.as_tensor(list(af_grid), dtype=torch.float64)
            mdb = _power_curve(n, grid, alpha=alpha, target_power=target_power)
            curve = pd.DataFrame({
                "af": [float(v) for v in grid.tolist()],
                "min_detectable_beta": [float(v) for v in mdb.tolist()],
            })
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        n_powered = int((df["power"] >= target_power).sum())
        return PowerRun(
            runtime_s=elapsed(), output_files=output_files, results=df, curve=curve,
            alpha=float(alpha), n=n, target_power=float(target_power),
            n_variants=int(len(df)), n_powered=n_powered,
        )


def winners_curse(gwas, *, method="conditional_likelihood", alpha=5e-8,
                  n_boot=10000, seed=None, output=None, sep="\t") -> "WinnersCurseRun":
    """Winner's-curse effect-size de-biasing (conditional_likelihood / fiqt / bootstrap)."""
    from ..postgwas import load_sumstats, correct_winners_curse

    valid = {"conditional_likelihood", "fiqt", "bootstrap"}
    if method not in valid:
        raise ValueError(f"method must be one of {sorted(valid)}; got {method!r}.")
    ss = load_sumstats(str(gwas), sep=sep)

    with timed() as elapsed:
        # CL/fiqt take no **kwargs and raise TypeError on n_boot/seed — forward only for bootstrap
        extra = {"n_boot": int(n_boot), "seed": seed} if method == "bootstrap" else {}
        res = correct_winners_curse(ss.beta, ss.se, method=method, alpha=alpha, **extra)
        beta_orig = [float(v) for v in ss.beta.tolist()]
        beta_adj = [float(v) for v in res.beta_adjusted.tolist()]
        shrink = [float(v) for v in res.shrinkage_factor.tolist()]
        m = len(beta_orig)
        se_adj = ([float("nan")] * m if res.se_adjusted is None
                  else [float(v) for v in res.se_adjusted.tolist()])
        df = pd.DataFrame({
            "snp": ss.snp, "beta_original": beta_orig, "beta_adjusted": beta_adj,
            "se_adjusted": se_adj, "shrinkage_factor": shrink,
        })
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        return WinnersCurseRun(
            runtime_s=elapsed(), output_files=output_files, results=df,
            method=str(res.method), n_corrected=int(res.n_corrected), n_variants=int(m),
        )


def gene_set_enrichment(gwas, gene_annotation, gene_sets, *, window_kb=0.0,
                        covariate_gene_size=True, covariate_log_size=True,
                        gene_sets_format="auto", output=None, sep="\t") -> "EnrichmentRun":
    """MAGMA-style competitive gene-set enrichment (snp_to_gene then set test)."""
    from ..postgwas import load_sumstats, snp_to_gene
    from ..postgwas import gene_set_enrichment as _gse

    ss = load_sumstats(str(gwas), sep=sep)
    gid, gchr, gstart, gend = _load_gene_annotation(gene_annotation, sep=sep)
    sets = _load_gene_sets(gene_sets, fmt=gene_sets_format, sep=sep)

    with timed() as elapsed:
        gene_result = snp_to_gene(ss, gid, gchr, gstart, gend, window_kb=window_kb)
        enr = _gse(gene_result, sets, covariate_gene_size=covariate_gene_size,
                   covariate_log_size=covariate_log_size)
        df = pd.DataFrame({
            "gene_set_name": list(enr.gene_set_name),
            "n_genes_in_set": [int(x) for x in enr.n_genes_in_set],
            "beta_enrichment": [float(v) for v in enr.beta_enrichment.tolist()],
            "se": [float(v) for v in enr.se.tolist()],
            "p": [float(v) for v in enr.p.tolist()],
        })
        genes_df = pd.DataFrame({
            "gene_id": list(gene_result.gene_id),
            "gene_chr": list(gene_result.gene_chr),
            "gene_start": [int(x) for x in gene_result.gene_start],
            "gene_end": [int(x) for x in gene_result.gene_end],
            "n_snps": [int(x) for x in gene_result.n_snps],
            "stat": [float(v) for v in gene_result.stat.tolist()],
            "p": [float(v) for v in gene_result.p.tolist()],
        })
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        return EnrichmentRun(
            runtime_s=elapsed(), output_files=output_files, results=df, genes=genes_df,
            n_genes_total=int(enr.n_genes_total), n_gene_sets=int(len(df)),
            n_significant=int((df["p"] < 0.05).sum()),
        )


def _load_matrix(path):
    """Load an (m, m) LD matrix from a .npy or .pt/.pth file as a float64 Tensor."""
    import numpy as np
    import torch
    p = str(path)
    if p.lower().endswith(".npy"):
        arr = np.load(p)
        mat = torch.as_tensor(arr, dtype=torch.float64)
    elif p.lower().endswith((".pt", ".pth")):
        obj = torch.load(p, map_location="cpu")
        mat = torch.as_tensor(obj, dtype=torch.float64)
    else:
        raise ValueError(f"ld_matrix must be a .npy or .pt/.pth file; got {p!r}.")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"ld_matrix must be square (m, m); got shape {tuple(mat.shape)}.")
    return mat


def _load_regions(regions, sep="\t"):
    """Load a genomic regions TSV (columns chrom, start, end) into a list of tuples."""
    # keep_default_na=False: a chromosome literally named "NA" must stay the
    # string "NA" so it matches ss.chr (mirrors the enrichment loaders).
    df = pd.read_csv(str(regions), sep=sep, keep_default_na=False)
    need = {"chrom", "start", "end"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"regions file must have columns {sorted(need)} (bp); got {list(df.columns)}."
        )
    return [(str(c), int(s), int(e))
            for c, s, e in zip(df["chrom"], df["start"], df["end"])]


def hess(gwas, ld_matrix, regions, *, n=None, n2=None, gwas2=None,
         eigenvalue_threshold=1.0, output=None, sep="\t") -> "HessRun":
    """HESS local heritability (or local genetic correlation with ``gwas2``).

    ``ld_matrix`` (.npy/.pt) must be (m, m) aligned to the sumstats row order.
    ``regions`` is a TSV (chrom,start,end in bp); each region is mapped to a
    CONTIGUOUS SNP-index block, so the sumstats + LD matrix must be sorted by
    genomic position.
    """
    import warnings
    import torch
    from ..postgwas import load_sumstats, hess_local_h2, hess_local_rg

    ss = load_sumstats(str(gwas), sep=sep)
    ld = _load_matrix(ld_matrix)
    if ld.shape[0] != ss.m:
        raise ValueError(f"ld_matrix is {ld.shape[0]}x{ld.shape[0]} but the sumstats "
                         f"has {ss.m} SNPs — they must align 1:1.")
    reg = _load_regions(regions, sep=sep)

    def _resolve_n(nval, label):
        if nval is not None:
            return float(nval)
        finite = ss.n[torch.isfinite(ss.n)]
        if finite.numel() == 0:
            raise ValueError(f"hess needs {label}: pass it explicitly (no usable 'n' column).")
        return float(finite.median().item())

    # map bp regions -> contiguous (start, end) index blocks
    chr_str = [str(c) for c in ss.chr]
    pos = [int(p) for p in ss.pos]
    bounds: list[tuple[int, int]] = []
    labels: list[str] = []
    bp_meta: list[tuple[str, int, int]] = []
    for chrom, start, end in reg:
        idx = [i for i in range(ss.m)
               if chr_str[i] == str(chrom) and start <= pos[i] <= end]
        if not idx:
            warnings.warn(f"hess: region {chrom}:{start}-{end} matched no SNPs — skipping.")
            continue
        first, last = min(idx), max(idx)
        if sorted(idx) != list(range(first, last + 1)):
            raise ValueError(
                f"region {chrom}:{start}-{end} maps to non-contiguous SNP indices — "
                f"the sumstats and LD matrix must be sorted by genomic position."
            )
        bounds.append((first, last + 1))
        labels.append(f"{chrom}:{start}-{end}")
        bp_meta.append((str(chrom), int(start), int(end)))
    if not bounds:
        raise ValueError(
            f"none of the {len(reg)} regions matched any SNP in the sumstats — "
            f"check chromosome naming and coordinates."
        )

    with timed() as elapsed:
        if gwas2 is not None:
            ss2 = load_sumstats(str(gwas2), sep=sep)
            if ss2.m != ss.m:
                raise ValueError("gwas2 must have the same SNPs (same m) as gwas.")
            n1v = _resolve_n(n, "n (trait 1)")
            n2v = _resolve_n(n2 if n2 is not None else n, "n2 (trait 2)")
            hres = hess_local_rg(ss.z, ss2.z, ld, n1v, n2v, bounds,
                                 region_labels=labels,
                                 eigenvalue_threshold=eigenvalue_threshold)
            mode = "rg"
        else:
            nv = _resolve_n(n, "n")
            hres = hess_local_h2(ss.z, ld, nv, bounds, region_labels=labels,
                                 eigenvalue_threshold=eigenvalue_threshold)
            mode = "h2"
        rows = []
        for rr, (chrom, start, end) in zip(hres.regions, bp_meta):
            rows.append({
                "region_id": rr.region_id, "chrom": chrom, "start": start, "end": end,
                "h2_local": rr.h2_local, "h2_local_se": rr.h2_local_se,
                "n_snps": rr.n_snps, "n_eigenvalues_kept": rr.n_eigenvalues_kept,
            })
        df = pd.DataFrame(rows)
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        return HessRun(
            runtime_s=elapsed(), output_files=output_files, results=df, mode=mode,
            h2_total=float(hres.h2_total), h2_total_se=float(hres.h2_total_se),
            n_regions=int(hres.n_regions), n_snps_total=int(hres.n_snps_total),
        )
