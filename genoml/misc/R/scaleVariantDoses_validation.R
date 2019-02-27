## run with ... 
## Rscript $pathToGenoML/otherPackages/scaleVariantDoses_validation.R $prefix $gwas $geno $trainingGenos

## Parse args and start logging
library(data.table)
args <- commandArgs()
print(args)
prefix <- args[6] # prefix predefined for this dataset by the earlier scripting
gwas <- args[7] # external GWAS file
geno <- args[8] # geno file prefix for validation dataset
trainingGenos <- args[9] # trainig dataset genotype prefix corresponding to *.bim
sink(file = paste(prefix, "_alleleDosageScaling.Rout", sep = ""))

## now read in the data
mapFile <- fread(file = paste(trainingGenos, ".bim", sep = ""))
snpList <- fread(file = paste(prefix, ".temp.snpsToPull2", sep = ""))
gwasResults <- fread(file = gwas)
genoDoses <- fread(file = paste(prefix, ".reduced_genos.raw", sep = ""))

## now make some tweaks to the GWAS
gwasResults$mafWeight <- ifelse(gwasResults$freq < 0.5, gwasResults$b, -1*gwasResults$b)

## build the index file
mapReduced <- mapFile[,c("V2","V5")]
names(mapReduced) <- c("SNP","minorAllele")
snpsReduced <- snpList[,c("SNP","CHR","BP","P")]
gwasReduced <- gwasResults[,c("SNP","mafWeight")]
varIndexTemp <-merge(mapReduced, snpsReduced, by = "SNP")
varIndex <- merge(varIndexTemp, gwasReduced, by = "SNP")
fwrite(varIndex, file = paste(prefix, "variantWeightings", sep = "."), quote = F, sep = "\t", row.names = F, na = NA)
varIndex$rawSnpName <- paste(varIndex$SNP, varIndex$minorAllele, sep = "_")

## now recode the *.raw file with new weights for minor alleles
for (i in 1:length(varIndex$rawSnpName)) 
{
  print(i)
  thisVar <- varIndex$rawSnpName[i]
  print(thisVar)
  thisVarWeight <- varIndex$mafWeight[i]
  print(thisVarWeight)
  thisVarWeighted <- genoDoses[[thisVar]]*thisVarWeight
  print(summary(thisVarWeighted))
  genoDoses[[thisVar]] <- thisVarWeighted
  print(summary(genoDoses[[thisVar]]))
}

## here we just output the weighted file over the old one, not the best idea but...
fwrite(genoDoses, file = paste(prefix, ".reduced_genos.raw", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)

## shut'em down
sink()
q("no")