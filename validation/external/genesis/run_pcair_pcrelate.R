suppressMessages({library(SNPRelate); library(GENESIS); library(GWASTools)})
setwd("out")
# KING (already have GDS) for kinship input to pcair
if(!file.exists("fixture.gds")) snpgdsBED2GDS("fixture.bed","fixture.fam","fixture.bim","fixture.gds", verbose=FALSE)
gds <- snpgdsOpen("fixture.gds")
king <- snpgdsIBDKING(gds, type="KING-robust", verbose=FALSE)
kingmat <- king$kinship; rownames(kingmat)<-colnames(kingmat)<-king$sample.id
snpgdsClose(gds)
# PC-AiR
geno <- GdsGenotypeReader("fixture.gds")
gd <- GenotypeData(geno)
pca <- pcair(gd, kinobj=kingmat, divobj=kingmat, snp.include=NULL, verbose=FALSE)
pcs <- pca$vectors[, 1:5, drop=FALSE]
write.table(data.frame(sample.id=pca$sample.id, pcs), "genesis_pcair.tsv", sep="\t", quote=FALSE, row.names=FALSE)
# PC-Relate needs an iterator + the PCs
seqclose <- close(gd)
geno2 <- GdsGenotypeReader("fixture.gds")
gd2 <- GenotypeData(geno2)
iterator <- GenotypeBlockIterator(gd2, snpBlock=5000)
pcrel <- pcrelate(iterator, pcs=pca$vectors[,1:3,drop=FALSE],
                  training.set=pca$unrels, verbose=FALSE)
km <- pcrel$kinBtwn
write.table(km, "genesis_pcrelate.tsv", sep="\t", quote=FALSE, row.names=FALSE)
close(gd2)
cat("GENESIS_DONE pcair_n=", length(pca$sample.id), " unrel=", length(pca$unrels), "\n")
