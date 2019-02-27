## run with ... 
## Rscript $pathToGenoML/otherPackages/prepSblupToScale.R $prefix $gwas $geno

## testing 
prefix <- "20190124-125238"
gwas <- "example_GWAS.txt"
geno <- "20190124-125238"

## Parse args and start logging
args <- commandArgs()
print(args)
prefix <- args[6]
gwas <- args[7]
geno <- args[8]

## load library
library("data.table")

## load SBLUP snps
snpList <- fread(paste(prefix,".reduced_genos_snpList", sep = ""), header = F)
names(snpList) <- c("SNP")
## load GWAS results
gwasResults <- fread(file = gwas, header = T)

## load map file
snpFile <- fread(paste(prefix,".forSblup.bim", sep = ""), header = F)
names(snpFile) <- c("CHR","SNP","CM","BP","minorAllele","majorAllele")

## merge files
temp <- merge(snpList, gwasResults, by = "SNP")
data <- merge(temp, snpFile, by = "SNP")

## reduce data and export list for scaling
dat <- data[,c("SNP","CHR","BP","p")]
names(dat) <- c("SNP","CHR","BP","P")
fwrite(dat, file = paste(prefix, ".temp.snpsToPull2", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)

## close down shop and don't save nuthin
q("no")
