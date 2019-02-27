## Parse args and start logging
args <- commandArgs()
print(args)
prefix <- args[6]

## load library
library("data.table")

## load SBLUP results
data <- fread(paste(prefix,".temp.sblup.cojo", sep = ""), header = F)

## start filters for sign matching and abs > 1 in sblup estimates to get ~25% data
data$match <- ifelse(sign(data$V3) == sign(data$V4), 1, 0)
dat <- subset(data, match == 1 & abs(data$V4) > 0.05)
names(dat) <- c("SNP","effectAllele","gwasBeta","sblupBeta","effectMatch")

## export list of SNPs to pull
fwrite(dat, paste(prefix,".sblupToPull", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)

## close down shop and don't save nuthin
q("no")
