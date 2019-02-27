## run with ... 
## Rscript $pathToGenoML/otherPackages/checkPrs.R $prefix $pheno

library("data.table")
library("caret")
library("pROC")

args <- commandArgs()
print(args)
prefix <- args[6]
pheno <- args[7]
## testing
# prefix <- "g-banner.harmonized-p-banner.amyloid_disc-c-banner.PCs-a-banner.age_sex_e2_e4"
# pheno <- "banner.amyloid_disc"

predictedInput <- paste(prefix, "_trainingSetPredictions.tab", sep = "")
predictedFile <- fread(predictedInput)
predictedFile$PHENO <- NULL
phenoInput <- paste(pheno, "", sep = "") #paste(pheno, ".pheno", sep = "")
phenoFile <- fread(phenoInput)
phenoFile$ID <- paste(phenoFile$FID, phenoFile$IID, sep = "_")
data <- merge(phenoFile, predictedFile, by = "ID")
minPheno <- min(data$PHENO)
maxPheno <- max(data$PHENO)
if(minPheno == 1 & maxPheno == 2)
{
  data$bin <- data$PHENO - 1
  outPut <- matrix(nrow = 3, ncol = 3)
  rocAuc <- roc(response = data$bin, predictor =  data$probDisease)
  outPut[1,1] <- auc(rocAuc)
  outPut[1,2] <- ci(rocAuc, of="auc")[1]
  outPut[1,3] <- ci(rocAuc, of="auc")[3]
  coords(rocAuc, "best")
  thresh <- coords(rocAuc, "best")[1]
  data$bin50 <- as.numeric(ifelse(data$probDisease > 0.5, 1, 0))
  rocAuc <- roc(response = data$bin, predictor = data$bin50)
  outPut[2,1] <- auc(rocAuc)
  outPut[2,2] <- ci(rocAuc, of="auc")[1]
  outPut[2,3] <- ci(rocAuc, of="auc")[3]
  data$binBest <- as.numeric(ifelse(data$probDisease > thresh, 1, 0))
  rocAuc <- roc(response = data$bin, predictor = data$binBest)
  outPut[3,1] <- auc(rocAuc)
  outPut[3,2] <- ci(rocAuc, of="auc")[1]
  outPut[3,3] <- ci(rocAuc, of="auc")[3]
  row.names(outPut) <- c("probability","partition_at_50percent","partition_at_best")
  write.table(outPut, file = paste(prefix, ".training_summary.txt", sep = ""), quote = F, sep = "\t", col.names = c("AUC","low_95_CI","high_95_CI"))
  ## here is the confmat at best theshold
  confMat <- confusionMatrix(data = as.factor(data$binBest), reference = as.factor(data$bin), positive = "1")
  sink(file = paste(prefix,"_confMatAtBestThresh_training.txt",sep =""))
  print(confMat)
  sink()
}
if(minPheno != 1 & maxPheno != 2)
{
  model <- lm(PHENO ~ predicted, data = data)
  sink(file = paste(prefix, ".training_summary.txt", sep = ""))
  print(summary(model))
  sink()
}
q("no")
