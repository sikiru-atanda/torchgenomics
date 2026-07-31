suppressMessages(library(SNPRelate))
setwd("out")
snpgdsBED2GDS("fixture.bed","fixture.fam","fixture.bim","fixture.gds", verbose=FALSE)
g <- snpgdsOpen("fixture.gds")
# KING-robust kinship
k <- snpgdsIBDKING(g, type="KING-robust", verbose=FALSE)
kin <- k$kinship
rownames(kin) <- k$sample.id; colnames(kin) <- k$sample.id
write.table(kin, "king_snprelate.tsv", sep="\t", quote=FALSE, col.names=NA)
write.table(data.frame(sample.id=k$sample.id), "king_sample_order.tsv", sep="\t", quote=FALSE, row.names=FALSE)
snpgdsClose(g)
cat("KING_DONE n=", length(k$sample.id), "\n")
