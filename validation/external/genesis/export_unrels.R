suppressMessages({library(SNPRelate); library(GENESIS); library(GWASTools)})
setwd("out")
g<-snpgdsOpen("fixture.gds"); king<-snpgdsIBDKING(g,type="KING-robust",verbose=FALSE); snpgdsClose(g)
kingmat<-king$kinship; rownames(kingmat)<-colnames(kingmat)<-king$sample.id
geno<-GdsGenotypeReader("fixture.gds"); gd<-GenotypeData(geno)
pca<-pcair(gd, kinobj=kingmat, divobj=kingmat, verbose=FALSE)
writeLines(as.character(pca$unrels), "genesis_unrels.txt")
close(gd)
cat("unrels written:", length(pca$unrels), "\n")
