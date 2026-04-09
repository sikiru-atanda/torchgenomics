"""Create tiny test fixtures for Phase 1 tests.

Run once: python tests/fixtures/create_fixtures.py
Creates PLINK BED/BIM/FAM, HapMap, CSV dosage, phenotype, and covariate files.
"""

from pathlib import Path
import numpy as np
import struct

FIXTURE_DIR = Path(__file__).parent
N_SAMPLES = 10
N_VARIANTS = 20
np.random.seed(42)


def create_plink():
    """Create a minimal PLINK BED/BIM/FAM fileset."""
    prefix = FIXTURE_DIR / "tiny"

    # .fam: FID IID father mother sex phenotype
    with open(prefix.with_suffix(".fam"), "w") as fh:
        for i in range(N_SAMPLES):
            fh.write(f"FAM{i:03d} IND{i:03d} 0 0 0 -9\n")

    # .bim: chr snp cm pos a1 a2
    with open(prefix.with_suffix(".bim"), "w") as fh:
        for j in range(N_VARIANTS):
            chrom = str((j % 3) + 1)
            fh.write(f"{chrom}\trs{j:04d}\t0\t{(j+1)*1000}\tA\tG\n")

    # .bed: magic bytes + SNP-major encoded genotypes
    # Generate random genotypes: 0, 1, 2 with one missing (NaN → code 01)
    geno = np.random.randint(0, 3, size=(N_VARIANTS, N_SAMPLES))
    # Insert one missing value
    geno[2, 5] = -1  # will encode as missing

    bytes_per_snp = (N_SAMPLES + 3) // 4

    with open(prefix.with_suffix(".bed"), "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))  # magic

        for j in range(N_VARIANTS):
            byte_arr = np.zeros(bytes_per_snp, dtype=np.uint8)
            for i in range(N_SAMPLES):
                val = geno[j, i]
                if val == -1:
                    code = 0b01  # missing
                elif val == 0:
                    code = 0b00  # hom A1
                elif val == 1:
                    code = 0b10  # het
                else:
                    code = 0b11  # hom A2

                byte_idx = i // 4
                bit_offset = (i % 4) * 2
                byte_arr[byte_idx] |= code << bit_offset

            fh.write(byte_arr.tobytes())

    # Save ground truth for verification
    geno_float = geno.astype(np.float64)
    geno_float[geno == -1] = np.nan
    np.save(FIXTURE_DIR / "tiny_dosage_truth.npy", geno_float)

    print(f"Created PLINK fileset: {prefix} ({N_SAMPLES} samples, {N_VARIANTS} variants)")


def create_phenotype():
    """Create phenotype and covariate files."""
    # Phenotype: some samples match genotype, some don't (for alignment testing)
    with open(FIXTURE_DIR / "tiny_pheno.txt", "w") as fh:
        fh.write("IID\tY1\tY2\n")
        for i in range(N_SAMPLES + 2):
            iid = f"IND{i:03d}" if i < N_SAMPLES else f"EXTRA{i:03d}"
            y1 = np.random.randn()
            y2 = np.random.randn() if i != 3 else "NA"  # one missing
            fh.write(f"{iid}\t{y1:.6f}\t{y2}\n")

    # Covariates: matching genotype samples
    with open(FIXTURE_DIR / "tiny_covar.txt", "w") as fh:
        fh.write("IID\tSEX\tAGE\n")
        for i in range(N_SAMPLES):
            fh.write(f"IND{i:03d}\t{np.random.randint(0,2)}\t{np.random.randint(20,60)}\n")

    print("Created phenotype and covariate files")


def create_hapmap():
    """Create a tiny HapMap file."""
    alleles_list = ["A/G", "C/T", "A/T", "G/C"]
    with open(FIXTURE_DIR / "tiny.hmp.txt", "w") as fh:
        # Header
        cols = ["rs#", "alleles", "chrom", "pos", "strand", "assembly#",
                "center", "protLSID", "assayLSID", "panelLSID", "QCcode"]
        sample_cols = [f"IND{i:03d}" for i in range(N_SAMPLES)]
        fh.write("\t".join(cols + sample_cols) + "\n")

        for j in range(N_VARIANTS):
            allele = alleles_list[j % len(alleles_list)]
            a1, a2 = allele.split("/")
            chrom = str((j % 3) + 1)
            row = [f"rs{j:04d}", allele, chrom, str((j+1)*1000),
                   "+", "NA", "NA", "NA", "NA", "NA", "NA"]

            for i in range(N_SAMPLES):
                dose = np.random.randint(0, 3)
                if dose == 0:
                    gt = a1 + a1
                elif dose == 1:
                    gt = a1 + a2
                else:
                    gt = a2 + a2
                row.append(gt)

            fh.write("\t".join(row) + "\n")

    print("Created HapMap file")


def create_numeric_csv():
    """Create a numeric dosage CSV (markers as rows)."""
    sample_ids = [f"IND{i:03d}" for i in range(N_SAMPLES)]
    marker_ids = [f"rs{j:04d}" for j in range(N_VARIANTS)]

    dosage = np.random.randint(0, 3, size=(N_VARIANTS, N_SAMPLES)).astype(float)

    with open(FIXTURE_DIR / "tiny_dosage.csv", "w") as fh:
        fh.write("," + ",".join(sample_ids) + "\n")
        for j in range(N_VARIANTS):
            fh.write(marker_ids[j] + "," + ",".join(f"{d:.0f}" for d in dosage[j]) + "\n")

    # Companion map file
    with open(FIXTURE_DIR / "tiny_dosage.map", "w") as fh:
        for j in range(N_VARIANTS):
            chrom = str((j % 3) + 1)
            fh.write(f"{chrom}\trs{j:04d}\t0\t{(j+1)*1000}\n")

    print("Created numeric dosage CSV + map file")


def create_zarr():
    """Create a tiny Zarr store with the standard layout."""
    try:
        import zarr
    except ImportError:
        print("Skipping Zarr fixture (zarr not installed)")
        return

    store_path = str(FIXTURE_DIR / "tiny.zarr")

    # Load ground truth dosage (SNP-major → transpose to sample-major)
    truth = np.load(FIXTURE_DIR / "tiny_dosage_truth.npy")  # (n_variants, n_samples)
    # Replace NaN with 0 for zarr (imputation happens at read time)
    dosage = np.nan_to_num(truth.T, nan=0.0).astype(np.float32)  # (n_samples, n_variants)

    root = zarr.open_group(store_path, mode="w")

    def _zarr_create(group, name, data, **kw):
        """Compat wrapper: zarr 3.x create_array vs zarr 2.x create_dataset."""
        fn = getattr(group, "create_array", None) or group.create_dataset
        try:
            return fn(name, data=data, **kw)
        except TypeError:
            # zarr 2.x signature
            return group.create_dataset(name, data=data, **kw)

    calldata = root.create_group("calldata")
    _zarr_create(calldata, "dosage", dosage, chunks=(N_SAMPLES, 5))

    sample_ids = [f"IND{i:03d}" for i in range(N_SAMPLES)]
    _zarr_create(root, "samples", np.array(sample_ids, dtype="U"))

    snp_ids = [f"rs{j:04d}" for j in range(N_VARIANTS)]
    chroms = [str((j % 3) + 1) for j in range(N_VARIANTS)]
    positions = [(j + 1) * 1000 for j in range(N_VARIANTS)]
    refs = ["A"] * N_VARIANTS
    alts = ["G"] * N_VARIANTS

    variants = root.create_group("variants")
    for name, data in [("id", snp_ids), ("chrom", chroms), ("ref", refs), ("alt", alts)]:
        _zarr_create(variants, name, np.array(data, dtype="U"))
    _zarr_create(variants, "pos", np.array(positions, dtype=np.int64))

    print(f"Created Zarr store: {store_path} ({N_SAMPLES} samples, {N_VARIANTS} variants)")


def create_hdf5():
    """Create a tiny HDF5 file with the standard layout."""
    try:
        import h5py
    except ImportError:
        print("Skipping HDF5 fixture (h5py not installed)")
        return

    h5_path = str(FIXTURE_DIR / "tiny.h5")

    # Load ground truth dosage
    truth = np.load(FIXTURE_DIR / "tiny_dosage_truth.npy")  # (n_variants, n_samples)
    dosage = np.nan_to_num(truth.T, nan=0.0).astype(np.float32)  # (n_samples, n_variants)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("calldata/dosage", data=dosage, chunks=(N_SAMPLES, 5))

        sample_ids = [f"IND{i:03d}" for i in range(N_SAMPLES)]
        f.create_dataset("samples", data=np.array(sample_ids, dtype="S"))

        snp_ids = [f"rs{j:04d}" for j in range(N_VARIANTS)]
        chroms = [str((j % 3) + 1) for j in range(N_VARIANTS)]
        positions = [(j + 1) * 1000 for j in range(N_VARIANTS)]
        refs = ["A"] * N_VARIANTS
        alts = ["G"] * N_VARIANTS

        f.create_dataset("variants/id", data=np.array(snp_ids, dtype="S"))
        f.create_dataset("variants/chrom", data=np.array(chroms, dtype="S"))
        f.create_dataset("variants/pos", data=np.array(positions, dtype=np.int64))
        f.create_dataset("variants/ref", data=np.array(refs, dtype="S"))
        f.create_dataset("variants/alt", data=np.array(alts, dtype="S"))

    print(f"Created HDF5 file: {h5_path} ({N_SAMPLES} samples, {N_VARIANTS} variants)")


if __name__ == "__main__":
    FIXTURE_DIR.mkdir(exist_ok=True)
    create_plink()
    create_phenotype()
    create_hapmap()
    create_numeric_csv()
    create_zarr()
    create_hdf5()
    print(f"\nAll fixtures created in {FIXTURE_DIR}")
