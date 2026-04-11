## 00_simulate_csv_data.R
## Simulates small CSV phenotype + genotype files that match TorchGWAS's
## 3-column numeric-dosage layout (SNP, Chromosome, Position_BP, <samples...>).
## Run this first, then run run_ten_models.R.

set.seed(42)

`%||%` <- function(a, b) if (!is.null(a)) a else b

n_samples <- 120
n_snps    <- 500

script_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) ".")
out_dir <- normalizePath(file.path(script_dir %||% ".", "data"),
                         mustWork = FALSE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

iid <- sprintf("S%04d", seq_len(n_samples))

## ---- Diploid genotype (dosages in {0, 1, 2}, ~2% missing) --------------
G <- matrix(sample(0:2, n_snps * n_samples, replace = TRUE,
                   prob = c(0.25, 0.50, 0.25)),
            nrow = n_snps, ncol = n_samples)
miss_idx <- sample(length(G), floor(0.02 * length(G)))
G[miss_idx] <- NA

chrom <- sort(sample(1:5, n_snps, replace = TRUE))
pos   <- unlist(tapply(seq_along(chrom), chrom,
                       function(ix) sort(sample(1e7, length(ix)))))
snp_id <- sprintf("rs%05d", seq_len(n_snps))

geno_df <- data.frame(SNP = snp_id,
                      Chromosome = chrom,
                      Position_BP = pos,
                      G,
                      check.names = FALSE)
colnames(geno_df)[4:ncol(geno_df)] <- iid
write.csv(geno_df, file.path(out_dir, "geno.csv"), row.names = FALSE, na = "")

## ---- Tetraploid genotype (dosages in {0..4}) for poly-scan -------------
G4 <- matrix(sample(0:4, n_snps * n_samples, replace = TRUE),
             nrow = n_snps, ncol = n_samples)
poly_df <- data.frame(SNP = snp_id,
                      Chromosome = chrom,
                      Position_BP = pos,
                      G4,
                      check.names = FALSE)
colnames(poly_df)[4:ncol(poly_df)] <- iid
write.csv(poly_df, file.path(out_dir, "poly_geno.csv"), row.names = FALSE)

## ---- Phenotype: 2 continuous, 1 binary, 1 ordinal, plus env/covars -----
y1    <- rnorm(n_samples)
y2    <- 0.3 * y1 + rnorm(n_samples)
ybin  <- rbinom(n_samples, 1, 0.3)
yord  <- sample(0:2, n_samples, replace = TRUE)
E1    <- rnorm(n_samples)
E2    <- rnorm(n_samples)
E3    <- rnorm(n_samples)
sex   <- sample(c(0, 1), n_samples, replace = TRUE)
age   <- round(rnorm(n_samples, 40, 10), 1)
FID   <- sample(sprintf("F%02d", 1:20), n_samples, replace = TRUE)

pheno <- data.frame(IID  = iid,
                    Y1   = y1,
                    Y2   = y2,
                    Ybin = ybin,
                    Yord = yord,
                    E1   = E1, E2 = E2, E3 = E3,
                    sex  = sex, age = age,
                    FID  = FID)
write.csv(pheno, file.path(out_dir, "pheno.csv"), row.names = FALSE)

## ---- Covariates file ---------------------------------------------------
write.csv(data.frame(IID = iid, sex = sex, age = age),
          file.path(out_dir, "covariates.csv"), row.names = FALSE)

## ---- Single-column environment file for gxe-scan -----------------------
write.table(data.frame(IID = iid, E = E1),
            file.path(out_dir, "env.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

## ---- Regions BED for set-scan (one region per chromosome) --------------
regions <- data.frame(chrom = sort(unique(chrom)),
                      start = 0,
                      end   = 1e8,
                      name  = paste0("chr", sort(unique(chrom)), "_all"))
write.table(regions, file.path(out_dir, "regions.bed"),
            sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)

cat("Wrote simulated CSV data to:", out_dir, "\n")
cat("Files:", list.files(out_dir), sep = "\n  ")
cat("\n")

## small helper used above so the script runs in RStudio and Rscript
`%||%` <- function(a, b) if (!is.null(a)) a else b
