"""I/O module: format detection, genotype readers, phenotype/covariate loaders."""

from .aligned import SampleAlignedReader  # noqa: F401
from .base import GenotypeReader  # noqa: F401
from .detect import detect_format  # noqa: F401
from .hapmap import HapMapReader  # noqa: F401
from .hdf5 import HDF5Reader  # noqa: F401
from .numeric import NumericDosageReader  # noqa: F401
from .phenotype import AlignmentManifest, PhenotypeData, load_phenotype  # noqa: F401
from .plink import PlinkBedReader  # noqa: F401
from .regions import (  # noqa: F401
    Region,
    compute_skat_weights,
    load_regions,
    map_regions_to_variants,
)
from .validate import run_preflight  # noqa: F401
from .zarr import ZarrReader  # noqa: F401
