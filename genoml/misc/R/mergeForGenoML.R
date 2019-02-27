## load data here from command script
args <- commandArgs()
print(args)
geno <- args[6]
pheno <- args[7]
cov <- args[8] # if no covs, sepc NA on command line
addit <- args[9] # if no addit, spec NA on command line
prefix <- args[10] # take it from the main python code
## prefix <- paste("g",geno,"p",pheno,"c",cov,"a",addit, sep = "-")
print(prefix)
sink(file = paste(prefix, "_mergeInput.Rout", sep = ""), type = c("output"))
print(paste("GENO PREEFIX file set as ", args[6], sep = ""))
print(paste("PHENO PREEFIX file set as ", args[7], sep = ""))
print(paste("COV PREEFIX file set as ", args[8], sep = "")) # if not specified load NA
print(paste("ADDIT PREEFIX file set as ", args[9], sep = "")) # if not specified load NA
print(paste("ALL RESULTS TAGGED WITH PREFIX -> ", prefix, sep = ""))

## load packages
library("data.table")

### now decide what to merge
genoPheno <- 2
addCov <- ifelse(cov == "NA", 0, 1)
addAddit <- ifelse(addit == "NA", 0, 1)
nFiles <- genoPheno + addCov + addAddit # this specifies the number of files to merge
print(paste("MERGING ", nFiles," FILES", sep = ""))
genotypeInput <- paste(prefix, ".reduced_genos.raw", sep = "")
phenoInput <- paste(pheno, "", sep = "") ## phenoInput <- paste(pheno, ".pheno", sep = "")
covInput <- paste(cov, "", sep = "") ## covInput <- paste(cov, ".cov", sep = "")
additInput <- paste(addit, "", sep = "") ## additInput <- paste(addit, ".addit", sep = "")

### run for only geno and pheno data, ie nFiles = 2
if(nFiles == 2)
{
	genosRaw <- fread(genotypeInput)
	phenoRaw <- fread(phenoInput)
	genosRaw$ID <- paste(genosRaw$FID, genosRaw$IID, sep = "_")
	phenoRaw$ID <- paste(phenoRaw$FID, phenoRaw$IID, sep = "_")
	phenoRaw[, c("FID","IID") := NULL]
	genosRaw[, c("FID","IID","MAT","PAT","SEX","PHENOTYPE") := NULL]
	temp <- merge(phenoRaw, genosRaw, by = "ID")
	names(temp)[2] <- "PHENO"
	fwrite(temp, file = paste(prefix,".dataForML", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
	print("First 100 variable names for your file below, the rest are likely just more genotypes...")
	print(head(names(temp), n = 100))
	print("... and the last 100 variable names for your file below...")
	print(tail(names(temp), n = 100))
	print(paste("Your final file has ", length(temp$ID)," samples, and ", length(names(temp))," predictors for analysis", sep = ""))	
}

### run for studies that have all geno, pheno, cov and addit data availible, ie nFiles = 4
if(nFiles == 4)
{
	genosRaw <- fread(genotypeInput)
	phenoRaw <- fread(phenoInput)
	covRaw <- fread(covInput)
	additRaw <- fread(additInput)
	genosRaw$ID <- paste(genosRaw$FID, genosRaw$IID, sep = "_")
	phenoRaw$ID <- paste(phenoRaw$FID, phenoRaw$IID, sep = "_")
	covRaw$ID <- paste(covRaw$FID, covRaw$IID, sep = "_")
	additRaw$ID <- paste(additRaw$FID, additRaw$IID, sep = "_")
	phenoRaw[, c("FID","IID") := NULL]
	genosRaw[, c("FID","IID","MAT","PAT","SEX","PHENOTYPE") := NULL]
	covRaw[, c("FID","IID") := NULL]
	additRaw[, c("FID","IID") := NULL]
	temp1 <- merge(phenoRaw, covRaw, by = "ID")
	temp2 <- merge(temp1, additRaw, by = "ID")
	temp3 <- merge(temp2, genosRaw, by = "ID")
	names(temp3)[2] <- "PHENO"
	fwrite(temp3, file = paste(prefix,".dataForML", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
	print("First 100 variable names for your file below, the rest are likely just more genotypes...")
	print(head(names(temp3), n = 100))
	print("... and the last 100 variable names for your file below...")
	print(tail(names(temp3), n = 100))
	print(paste("Your final file has ", length(temp3$ID)," samples, and ", length(names(temp3))," predictors for analysis", sep = ""))	
}

### run for studies that have all geno, pheno and cov data availible (addit is missing), ie nFiles = 3
if(nFiles == 3 & addit == "NA")
{
	genosRaw <- fread(genotypeInput)
	phenoRaw <- fread(phenoInput)
	otherRaw <- fread(covInput)
	genosRaw$ID <- paste(genosRaw$FID, genosRaw$IID, sep = "_")
	phenoRaw$ID <- paste(phenoRaw$FID, phenoRaw$IID, sep = "_")
	otherRaw$ID <- paste(otherRaw$FID, otherRaw$IID, sep = "_")
	phenoRaw[, c("FID","IID") := NULL]
	genosRaw[, c("FID","IID","MAT","PAT","SEX","PHENOTYPE") := NULL]
	otherRaw[, c("FID","IID") := NULL]
	temp1 <- merge(phenoRaw, otherRaw, by = "ID")
	temp2 <- merge(temp1, genosRaw, by = "ID")
	names(temp2)[2] <- "PHENO"
	fwrite(temp2, file = paste(prefix,".dataForML", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
	print("First 100 variable names for your file below, the rest are likely just more genotypes...")
	print(head(names(temp2), n = 100))
	print("... and the last 100 variable names for your file below...")
	print(tail(names(temp2), n = 100))
	print(paste("Your final file has ", length(temp2$ID)," samples, and ", length(names(temp2))," predictors for analysis", sep = ""))	
}

### run for studies that have all geno, pheno and addit data availible (cov is missing), ie nFiles = 3
if(nFiles == 3 & cov == "NA")
{
	genosRaw <- fread(genotypeInput)
	phenoRaw <- fread(phenoInput)
	otherRaw <- fread(additInput)
	genosRaw$ID <- paste(genosRaw$FID, genosRaw$IID, sep = "_")
	phenoRaw$ID <- paste(phenoRaw$FID, phenoRaw$IID, sep = "_")
	otherRaw$ID <- paste(otherRaw$FID, otherRaw$IID, sep = "_")
	phenoRaw[, c("FID","IID") := NULL]
	genosRaw[, c("FID","IID","MAT","PAT","SEX","PHENOTYPE") := NULL]
	otherRaw[, c("FID","IID") := NULL]
	temp1 <- merge(phenoRaw, otherRaw, by = "ID")
	temp2 <- merge(temp1, genosRaw, by = "ID")
	names(temp2)[2] <- "PHENO"
	fwrite(temp2, file = paste(prefix,".dataForML", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
	print("First 100 variable names for your file below, the rest are likely just more genotypes...")
	print(head(names(temp2), n = 100))
	print("... and the last 100 variable names for your file below...")
	print(tail(names(temp2), n = 100))
	print(paste("Your final file has ", length(temp2$ID)," samples, and ", length(names(temp2))," predictors for analysis", sep = ""))	
}

### save the log file
sink()

### now quit w/o saving workspace
q("no")
