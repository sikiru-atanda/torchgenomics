"""CLI smoke-matrix specification (Pillar C / spec §6).

This module defines the *cell* for every CLI subcommand under test. A cell is
the triple ``(subcommand, format, device)`` along with the CLI args that
produce a deterministic, smoke-testable invocation on the tiny fixtures
committed under ``tests/fixtures/``.

The contract:

* Every public ``torchgenomics`` CLI subcommand has at least one entry in
  ``CELL_SPEC``. Subcommands that genuinely cannot run on the small
  fixture set (require Julia, Docker, NCBI network, BEAGLE binary, …) are
  still listed with ``skip_reason`` so the matrix's coverage check stays
  exhaustive.
* Each cell either ships its inputs from the committed fixtures or
  declares them as files that the matrix harness produces on the fly
  (per-cell scratch files written to ``tmp_path``).

The harness in ``tests/test_cli_matrix.py`` enumerates ``CELL_SPEC``,
expands each entry into one parametrized pytest cell per supported
format, builds the subprocess argv, runs the CLI, and asserts:

1. exit code 0,
2. each declared output exists with non-zero size,
3. for scan subcommands, the canonical ``.assoc.tsv`` row count equals
   the committed fixture's variant count and every ``P`` / ``P_*`` /
   ``P_ADJ`` column lies in ``[0, 1]`` (or is NaN with explicit
   ``allow_nan_p=True``),
4. for ``convert``, the output has the expected magic bytes / store
   layout,
5. for ``validate``, the JSON report has the documented schema keys.

GPU cells live in the ``GPU_SUBSET`` list and are gated on
``torch.cuda.is_available()`` at parametrization time.

This file is *data*, not policy. Edit it freely as new subcommands land.

Schema for a CELL entry (CellSpec)
----------------------------------

::

    {
        # Required ─ which formats the subcommand can ingest. The harness
        # parametrizes one cell per format. Use "n/a" for subcommands that
        # don't take a genotype file (mediate / mediate-scan / pgs-fit / ...).
        "formats": ["bed", "hmp", "csv", "zarr", "hdf5", "n/a"],

        # Required ─ build the CLI argv for one (subcommand, format) cell.
        # Receives a `Ctx` dataclass with paths to fixtures + tmp_path for
        # scratch files. Returns a list of strings: subcommand args after
        # `python -m torchgenomics.cli <subcommand>`.
        "build_args": Callable[[Ctx, str], list[str]],

        # Required ─ list of expected output files (paths relative to ctx.tmp).
        # Resolved per cell; the harness asserts each exists and is non-empty.
        "expected_outputs": Callable[[Ctx, str], list[str]],

        # Optional ─ skip reason for the entire subcommand (or per format).
        "skip_reason": str | dict[format, str],

        # Optional ─ output category for assertion routing:
        #   "scan"     ─ canonical {prefix}.assoc.tsv (default for *-scan)
        #   "set_scan" ─ {prefix}_set_based.tsv
        #   "bayes"    ─ {prefix}_bayesian_vs.tsv
        #   "met_scan" ─ {prefix}_met_scan.tsv
        #   "knockoff" ─ {prefix}.knockoff.tsv + {prefix}.knockoff_blocks.tsv
        #   "rr"       ─ {prefix}.rr.tsv
        #   "rr_met"   ─ {prefix}.rr_met.tsv
        #   "mediate"  ─ json blob
        #   "mediate_scan" ─ {prefix}.tsv + {prefix}.top.tsv
        #   "ld_blocks"    ─ {prefix}.bed + {prefix}.blocks.det
        #   "ldsc"     ─ {prefix}.ldsc.txt
        #   "ldsc_rg"  ─ {prefix}.ldsc_rg.txt
        #   "meta"     ─ {prefix}.meta.tsv
        #   "clump"    ─ {prefix}.clumps.tsv
        #   "pgs_fit"  ─ {output} TSV
        #   "pgs_score"─ {output} TSV
        #   "annotate" ─ {prefix}.genes.tsv
        #   "convert"  ─ converted file/store
        #   "impute"   ─ imputed .pt file
        #   "validate" ─ JSON report
        # Default: "scan".
        "kind": "scan",

        # Optional ─ override the n_variants assertion for "scan" kind cells.
        "expected_n_variants": 20,

        # Optional ─ tolerate NaN p-values (e.g., monomorphic SNPs).
        "allow_nan_p": False,
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: Number of samples in `tests/fixtures/tiny.*` (see create_fixtures.py).
N_SAMPLES = 10
#: Number of variants in `tests/fixtures/tiny.*`.
N_VARIANTS = 20
#: Sample IDs in the committed fixtures.
SAMPLE_IDS = [f"IND{i:03d}" for i in range(N_SAMPLES)]


@dataclass
class Ctx:
    """Per-cell context handed to ``build_args`` / ``expected_outputs``.

    Attributes
    ----------
    fixtures :
        Path to the committed ``tests/fixtures/`` directory.
    tmp :
        Per-cell scratch directory (``tmp_path`` from pytest). The harness
        runs the subprocess with ``cwd=tmp`` so relative output paths land
        here.
    output_prefix :
        Prefix for the canonical CLI ``--output`` argument inside ``tmp``.
        Subcommand-specific suffixes (``.assoc.tsv``, ``.bed``, …) are
        appended by the CLI itself.
    """

    fixtures: Path
    tmp: Path
    output_prefix: str = "out"


# ---------------------------------------------------------------------------
# Genotype-format mapping
# ---------------------------------------------------------------------------

#: Map from format-id → committed genotype file (or directory) under
#: ``tests/fixtures/``. ``None`` means the format isn't currently
#: represented by a committed fixture (we may generate a synthetic one
#: per cell, or skip).
FORMAT_GENOTYPE: dict[str, Path | None] = {
    "bed":  FIXTURES / "tiny.bed",       # PLINK BED — primary fixture
    "hmp":  FIXTURES / "tiny.hmp.txt",   # HapMap text — secondary fixture
    "csv":  FIXTURES / "tiny_dosage.csv",
    "zarr": FIXTURES / "tiny.zarr",
    "hdf5": FIXTURES / "tiny.h5",
    # vcf and bgen aren't committed; the matrix has no committed VCF/BGEN
    # fixture and we skip those cells with documented reason.
    "vcf":  None,
    "bgen": None,
}


# ---------------------------------------------------------------------------
# Helpers used by individual CELL_SPEC entries
# ---------------------------------------------------------------------------


def _genotype_args(fmt: str, ctx: Ctx) -> list[str]:
    """Return ``--genotype <path>`` for the requested format.

    The CSV ``NumericDosageReader`` autodiscovers the sibling ``.map`` file
    when one exists at ``<csv_basename>.map`` (see
    ``torchgenomics.io.map_file.discover_map_file``); we don't pass ``--map``
    explicitly since most subcommands' parsers don't expose it.
    """
    geno = FORMAT_GENOTYPE.get(fmt)
    if geno is None:
        raise ValueError(f"format {fmt!r} has no committed fixture")
    return ["--genotype", str(geno)]


def _scan_outputs(ctx: Ctx) -> list[str]:
    """Default scan output: ``{prefix}.assoc.tsv``."""
    return [f"{ctx.output_prefix}.assoc.tsv"]


def _basic_scan_args(fmt: str, ctx: Ctx, *extra: str) -> list[str]:
    """Standard ``[--genotype, --phenotype, --traits, --device cpu, --output]`` args."""
    return [
        *_genotype_args(fmt, ctx),
        "--phenotype", str(FIXTURES / "tiny_pheno.txt"),
        "--traits", "Y1",
        "--device", "cpu",
        "--output", ctx.output_prefix,
        *extra,
    ]


# ---------------------------------------------------------------------------
# CELL_SPEC ─ one entry per CLI subcommand
# ---------------------------------------------------------------------------

#: Formats supported by every "ordinary" scan subcommand. We exercise the
#: three text-based formats (bed/hmp/csv) on every scan to confirm format
#: autodetection still works under the new harness. We skip zarr / hdf5
#: at scan tier because they share the dosage-shape contract with csv and
#: don't introduce a new code path on the scan side.
SCAN_FORMATS = ["bed", "hmp", "csv"]

# Subcommands that can't run on the tiny fixture for documented reasons
# (insufficient n / m to converge, external binary, etc).
_SKIP_TINY_SMALL_N: dict[str, str] = {
    "ldsc":    "LDSC weighted-LS underdetermined on m=20 SNPs / n=10 samples",
    "ldsc-rg": "LDSC weighted-LS underdetermined on m=20 SNPs / n=10 samples",
    # SuSiE-RSS needs a non-degenerate LD matrix; in-sample R from n=10
    # samples on m=20 SNPs is rank-deficient and the IBSS update diverges.
    # Real-fixture parity is exercised by validation/external/susieR/ at
    # n>=279, p>=200 (3-fixture head-to-head, all metrics 1.000 vs susieR).
    "bayes-scan-rss": "SuSiE-RSS degenerate on rank-deficient n=10 / m=20 LD",
}

CELL_SPEC: dict[str, dict] = {
    # ── Validation / format / imputation ───────────────────────────────
    "validate": {
        "formats": SCAN_FORMATS + ["zarr", "hdf5"],
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(FIXTURES / "tiny_pheno.txt"),
            "--report", f"{ctx.output_prefix}.report.json",
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.report.json"],
        "kind": "validate",
    },
    "convert": {
        # convert(input → output) doesn't autodetect via --genotype; we
        # pass --input. Cover the three text-based input formats.
        # `convert`'s parser exposes --map (cli.py:3142) so CSV → zarr can
        # carry chr/pos metadata.
        "formats": ["bed", "hmp", "csv"],
        "build_args": lambda ctx, fmt: [
            "--input", str(FORMAT_GENOTYPE[fmt]),
            "--output", f"{ctx.output_prefix}.zarr",
            "--format", "zarr",
            *(["--map", str(FIXTURES / "tiny_dosage.map")] if fmt == "csv" else []),
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.zarr"],
        "kind": "convert",
    },
    "impute": {
        # impute --method mean works on every format; .pt blob output.
        "formats": ["bed", "hmp", "csv"],
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--method", "mean",
            "--output", f"{ctx.output_prefix}.imp.pt",
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.imp.pt"],
        "kind": "impute",
    },
    "dosage-call": {
        "formats": ["n/a"],
        "skip_reason": (
            "requires updog R package + Rscript + AD-format VCF input; "
            "no committed VCF read-count fixture in tests/fixtures/"
        ),
    },
    "phase-poly": {
        "formats": ["n/a"],
        "skip_reason": (
            "requires PolyOrigin Julia package + dosage-call output; "
            "Julia isn't always available in the smoke environment"
        ),
    },

    # ── Scan family ────────────────────────────────────────────────────
    "glm-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(fmt, ctx),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "lmm-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(fmt, ctx),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "mvlmm-scan": {
        "formats": SCAN_FORMATS,
        # mvlmm needs >=2 traits with no-NA; the harness writes a clean
        # phenotype file before invocation.
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_clean.txt"),
            "--traits", "Y1,Y2",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "mvlmm_scan",
    },
    "poly-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(
            fmt, ctx, "--ploidy", "2", "--gene-action", "additive",
        ),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "mklmm-scan": {
        "formats": SCAN_FORMATS,
        # mklmm with default "all" kernels can't fit on n=10; restrict to
        # the additive kernel which is well-determined.
        "build_args": lambda ctx, fmt: _basic_scan_args(
            fmt, ctx, "--kernels", "additive",
        ),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "gxe-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(FIXTURES / "tiny_pheno.txt"),
            "--traits", "Y1",
            "--env", str(ctx.tmp / "env.tsv"),
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "gxe_scan",
    },
    "set-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(FIXTURES / "tiny_pheno.txt"),
            "--traits", "Y1",
            "--regions", str(ctx.tmp / "regions.bed"),
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}_set_based.tsv"],
        "kind": "set_scan",
    },
    "bayes-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(
            fmt, ctx, "--method", "susie", "--n-signals", "2",
        ),
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}_bayesian_vs.tsv"],
        "kind": "bayes",
    },
    # bayes-scan-rss reads sumstats + an LD reference (.pt or .npz). The
    # tiny CLI-matrix fixture (n=10, m=20) cannot produce a non-degenerate
    # LD matrix for SuSiE-RSS — IBSS diverges on rank-deficient R. The
    # entry exists so the per-PR coverage gate accepts the new subcommand;
    # actual parity is exercised by the susieR head-to-head harness at
    # validation/external/susieR/ (3-fixture coverage at n >= 279).
    "bayes-scan-rss": {
        "formats": ["n/a"],
        "build_args": lambda ctx, fmt: [
            "--sumstats", str(ctx.tmp / "sumstats_lc.tsv"),
            "--geno", str(ctx.tmp / "tiny.bed"),
            "--max-num-causal", "2",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.tsv"],
        "kind": "bayes",
        "skip_reason": _SKIP_TINY_SMALL_N["bayes-scan-rss"],
    },
    "met-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_met.txt"),
            "--env-cols", "E1,E2,E3",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}_met_scan.tsv"],
        "kind": "met_scan",
    },
    "farmcpu-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(fmt, ctx),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
        "allow_nan_p": True,  # post-selection FarmCPU may emit NaN p
    },
    "blink-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(fmt, ctx),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
        "allow_nan_p": True,
    },
    "threshold-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_ord.txt"),
            "--traits", "Y1,Y2",
            "--trait-types", "ordinal,continuous",
            "--n-categories", "3,0",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "family-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_fam.txt"),
            "--traits", "Y1",
            "--family-col", "FID",
            "--min-family-size", "2",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "conditional-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(fmt, ctx),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "mtmet-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_mtmet.txt"),
            "--traits", "Y1,Y2",
            "--env-cols", "E1,E2",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        # mtmet writes <prefix>.assoc.tsv (see cli.py:1281).
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "ocf-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(
            fmt, ctx, "--n-folds", "2", "--seed", "42",
        ),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "knockoff-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(
            fmt, ctx, "--ld-method", "r2", "--seed", "42",
        ),
        "expected_outputs": lambda ctx, fmt: [
            f"{ctx.output_prefix}.knockoff.tsv",
            f"{ctx.output_prefix}.knockoff_blocks.tsv",
        ],
        "kind": "knockoff",
    },
    "gu-scan": {
        "formats": SCAN_FORMATS,
        # gu-scan without --probs / --dosage-var falls back to a regular
        # LMM scan (verified during preflight). This is fine for smoke.
        "build_args": lambda ctx, fmt: _basic_scan_args(fmt, ctx),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "lro-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: _basic_scan_args(
            fmt, ctx, "--ld-method", "r2",
        ),
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "glmm-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_bin.txt"),
            "--traits", "Y1",
            "--family", "binary",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "me-glmm-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_me_bin.txt"),
            "--traits", "E1,E2",
            "--env-cols", "E1,E2",
            "--family", "binary",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",  # me-GLMM emits scalar p_joint via the patched _apply_correction_and_save
    },
    "survival-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_surv.txt"),
            "--time-col", "TIME",
            "--event-col", "EVENT",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
    "rr-scan": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_long.txt"),
            "--id-col", "IID",
            "--time-col", "TIME",
            "--pheno-col", "Y",
            "--basis", "legendre",
            "--order", "2",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.rr.tsv"],
        "kind": "rr",
    },
    "rr-met-scan": {
        "formats": SCAN_FORMATS,
        # rr-met needs at least b time-points per env per individual; our
        # synthetic file has 4 (T=1..4) and order=1 → b=2.
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(ctx.tmp / "pheno_rr_met.txt"),
            "--id-col", "IID",
            "--env-col", "ENV",
            "--time-col", "TIME",
            "--pheno-col", "Y",
            "--basis", "legendre",
            "--order", "1",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.rr_met.tsv"],
        "kind": "rr_met",
    },

    # ── LD blocks ───────────────────────────────────────────────────────
    "ld-blocks": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--method", "r2",  # r2 is the most permissive method on tiny data
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [
            f"{ctx.output_prefix}.bed",
            f"{ctx.output_prefix}.blocks.det",
            f"{ctx.output_prefix}.summary.txt",
        ],
        "kind": "ld_blocks",
    },

    # ── Post-GWAS (sumstats consumers) ─────────────────────────────────
    "ldsc": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            "--sumstats", str(ctx.tmp / "sumstats_lc.tsv"),
            *_genotype_args(fmt, ctx),
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.ldsc.txt"],
        "kind": "ldsc",
        "skip_reason": _SKIP_TINY_SMALL_N["ldsc"],
    },
    "ldsc-rg": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            "--sumstats1", str(ctx.tmp / "sumstats_lc.tsv"),
            "--sumstats2", str(ctx.tmp / "sumstats_lc.tsv"),
            *_genotype_args(fmt, ctx),
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.ldsc_rg.txt"],
        "kind": "ldsc_rg",
        "skip_reason": _SKIP_TINY_SMALL_N["ldsc-rg"],
    },
    "meta": {
        "formats": ["n/a"],
        "build_args": lambda ctx, fmt: [
            "--input", str(ctx.tmp / "sumstats_lc.tsv"), str(ctx.tmp / "sumstats_lc.tsv"),
            "--method", "fixed",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.meta.tsv"],
        "kind": "meta",
    },
    "clump": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            "--sumstats", str(ctx.tmp / "sumstats_lc.tsv"),
            *_genotype_args(fmt, ctx),
            "--p-threshold", "0.5",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.clumps.tsv"],
        "kind": "clump",
    },

    # ── PGS family ─────────────────────────────────────────────────────
    "pgs-fit": {
        "formats": ["n/a"],
        "build_args": lambda ctx, fmt: [
            "--sumstats", str(ctx.tmp / "sumstats_lc.tsv"),
            "--ld-ref", str(ctx.tmp / "ld_ref.pt"),
            "--method", "ct",
            "--clump-p", "0.5",
            "--device", "cpu",
            "--output", str(ctx.tmp / f"{ctx.output_prefix}.weights.tsv"),
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.weights.tsv"],
        "kind": "pgs_fit",
    },
    "pgs-score": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--weights", str(ctx.tmp / "pgs_weights.tsv"),
            "--device", "cpu",
            "--output", str(ctx.tmp / f"{ctx.output_prefix}.scores.tsv"),
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.scores.tsv"],
        "kind": "pgs_score",
    },

    # ── Annotate ────────────────────────────────────────────────────────
    "annotate": {
        "formats": ["n/a"],
        "build_args": lambda ctx, fmt: [
            "--sumstats", str(ctx.tmp / "sumstats_lc.tsv"),
            "--crop", "maize",
            "--p-threshold", "0.5",
            "--no-go",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.genes.tsv"],
        "kind": "annotate",
        # Annotate caches NCBI responses under tests/fixtures/ncbi_cassette
        # via the existing test fixture machinery, so it's network-free in
        # the smoke matrix.
    },

    # ── Mediation ──────────────────────────────────────────────────────
    "mediate": {
        "formats": ["n/a"],
        "build_args": lambda ctx, fmt: [
            "--y", str(ctx.tmp / "y.npy"),
            "--snp", str(ctx.tmp / "snp.npy"),
            "--mediator", str(ctx.tmp / "m.npy"),
            "--kinship", str(ctx.tmp / "K.npy"),
            "--no-sensitivity",
            "--output", str(ctx.tmp / f"{ctx.output_prefix}.json"),
        ],
        "expected_outputs": lambda ctx, fmt: [f"{ctx.output_prefix}.json"],
        "kind": "mediate",
    },
    "mediate-scan": {
        "formats": ["n/a"],
        "build_args": lambda ctx, fmt: [
            "--y", str(ctx.tmp / "Y.npy"),
            "--genotype", str(ctx.tmp / "G.npy"),
            "--mediator-matrix", str(ctx.tmp / "M.npy"),
            "--kinship", str(ctx.tmp / "K.npy"),
            "--cis-window", "-1",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: [
            f"{ctx.output_prefix}.tsv",
            f"{ctx.output_prefix}.top.tsv",
        ],
        "kind": "mediate_scan",
    },

    # ── Pipeline ────────────────────────────────────────────────────────
    "pipeline": {
        "formats": SCAN_FORMATS,
        "build_args": lambda ctx, fmt: [
            *_genotype_args(fmt, ctx),
            "--phenotype", str(FIXTURES / "tiny_pheno.txt"),
            "--traits", "Y1",
            "--model", "glm",
            "--device", "cpu",
            "--output", ctx.output_prefix,
        ],
        "expected_outputs": lambda ctx, fmt: _scan_outputs(ctx),
        "kind": "scan",
    },
}

# ---------------------------------------------------------------------------
# GPU subset
# ---------------------------------------------------------------------------

#: Subcommands that exercise heavy LMM / GLMM / GxE math on GPU. The CPU
#: matrix proves correctness; the GPU subset proves the device dispatch
#: works end-to-end. Gated on torch.cuda.is_available() at parametrize time.
GPU_SUBSET: list[str] = [
    "glm-scan",
    "lmm-scan",
    "mvlmm-scan",
    "mklmm-scan",
    "gxe-scan",
    "met-scan",
    "mtmet-scan",
    "knockoff-scan",
    "ocf-scan",
    "glmm-scan",
    "me-glmm-scan",
    "survival-scan",
    "threshold-scan",
]


# ---------------------------------------------------------------------------
# Public helper API
# ---------------------------------------------------------------------------


def expand_cells(spec: dict[str, dict] = CELL_SPEC) -> list[tuple[str, str]]:
    """Yield ``(subcommand, format)`` tuples for every cell in ``spec``.

    Order is deterministic (alphabetical by subcommand, then by the format
    declaration order in CELL_SPEC).
    """
    cells: list[tuple[str, str]] = []
    for subcmd in sorted(spec.keys()):
        for fmt in spec[subcmd]["formats"]:
            cells.append((subcmd, fmt))
    return cells


def expand_gpu_cells(spec: dict[str, dict] = CELL_SPEC) -> list[tuple[str, str]]:
    """Yield ``(subcommand, format)`` tuples for the GPU subset.

    Only ``bed`` is exercised on GPU — the format dispatch is tested in
    the CPU matrix; GPU validates the device codepath only.
    """
    cells: list[tuple[str, str]] = []
    for subcmd in GPU_SUBSET:
        if subcmd not in spec:
            continue
        if "bed" in spec[subcmd]["formats"]:
            cells.append((subcmd, "bed"))
    return cells


__all__ = [
    "CELL_SPEC",
    "Ctx",
    "FIXTURES",
    "FORMAT_GENOTYPE",
    "GPU_SUBSET",
    "N_SAMPLES",
    "N_VARIANTS",
    "SAMPLE_IDS",
    "SCAN_FORMATS",
    "expand_cells",
    "expand_gpu_cells",
]
