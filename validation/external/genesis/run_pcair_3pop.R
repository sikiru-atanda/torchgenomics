suppressMessages({library(SNPRelate); library(GENESIS); library(GWASTools)})
setwd("out3")
snpgdsBED2GDS("fixture.bed","fixture.fam","fixture.bim","fixture.gds", verbose=FALSE)
g<-snpgdsOpen("fixture.gds"); king<-snpgdsIBDKING(g,type="KING-robust",verbose=FALSE); snpgdsClose(g)
km<-king$kinship; rownames(km)<-colnames(km)<-king$sample.id
geno<-GdsGenotypeReader("fixture.gds"); gd<-GenotypeData(geno)
pca<-pcair(gd, kinobj=km, divobj=km, verbose=FALSE)
write.table(data.frame(sample.id=pca$sample.id, pca$vectors[,1:5]), "genesis_pcair.tsv", sep="\t", quote=FALSE, row.names=FALSE)
close(gd); cat("PCAIR3_DONE n=",length(pca$sample.id)," unrels=",length(pca$unrels),"\n")
