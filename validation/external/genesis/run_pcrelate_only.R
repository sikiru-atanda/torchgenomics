suppressMessages({library(SNPRelate); library(GENESIS); library(GWASTools); library(BiocParallel)})
setwd("out")
g <- snpgdsOpen("fixture.gds"); king <- snpgdsIBDKING(g, type="KING-robust", verbose=FALSE); snpgdsClose(g)
kingmat <- king$kinship; rownames(kingmat) <- colnames(kingmat) <- king$sample.id
geno <- GdsGenotypeReader("fixture.gds"); gd <- GenotypeData(geno)
pca <- pcair(gd, kinobj=kingmat, divobj=kingmat, verbose=FALSE)
it <- GenotypeBlockIterator(gd, snpBlock=5000)
pcr <- pcrelate(it, pcs=pca$vectors[,1:3,drop=FALSE], training.set=pca$unrels,
                ibd.probs=FALSE, verbose=FALSE, BPPARAM=SerialParam())
write.table(pcr$kinBtwn, "genesis_pcrelate.tsv", sep="\t", quote=FALSE, row.names=FALSE)
close(gd)
cat("PCRELATE_DONE rows=", nrow(pcr$kinBtwn), "\n")
