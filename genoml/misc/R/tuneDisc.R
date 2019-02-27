## Parse args and start logging
args <- commandArgs()
print(args)
prefix <- args[6]
sink(file = paste(prefix, "_discreteTuning.Rout", sep = ""), type = c("output"))
ncores <- as.numeric(args[7])
cvReps <- as.numeric(args[8])
gridSearch <- as.numeric(args[9])
imputeMissingData <- args[10]
bestModelName <- args[11]

## Check command arguments from user
print(paste("DATA PREEFIX file set as ", prefix, sep = ""))
print(paste("RUNNIN ON HOW MANY CORES ??? ", ncores, " cores, is that enough??? ", sep = ""))
print(paste("RUNNNING CROSS VALIDATION FOR ", cvReps, " REPS", sep = ""))
print(paste("GRID SEARCH FOR ", gridSearch, " ITERATIONS", sep = ""))
print(paste("ANOTHER LEVEL OF IMPUTATION AND DATA TRANSFORMATION USING ", imputeMissingData, sep = ""))
print(paste("BEST MODEL FROM STEP 2 file set as ", bestModelName, sep = ""))

## Load additional packages
packageList <- c("caret","ggplot2","data.table","rBayesianOptimization","plotROC")
lapply(packageList, library, character.only = TRUE)

## Load dataset
train <- fread(paste(prefix,".dataForML", sep = ""), header = T)

### set outcome as a factor, check missingness, then impute missing data and scale
train$PHENO[train$PHENO == 2] <- "DISEASE"
train$PHENO[train$PHENO == 1] <- "CONTROL"
ID <- train$ID
train[,c("ID") := NULL]
preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
train_processed <- predict(preProcValues, train) # here we make the preprocessed values

## find out what the best model from earlier tune is
bestAlgorithmTemp <- read.table(paste(prefix,"_bestModel.algorithm",sep = ""))
bestAlgorithm <- as.character(bestAlgorithmTemp[1,1])

### to begin tune first begin parallel parameters 
library("parallel")
library("doParallel")
cluster <- makeCluster(ncores) # convention to leave 1 core for OS
registerDoParallel(cluster)
set.seed(123) # makes generally reproducible grid searches from random or bayes tunes at paramter optimization

## here we set the cross validation
fitControl <- trainControl(method = "repeatedcv",
                           number = cvReps,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           search = "random",
                           repeats = cvReps)

## now make the longer tuned model
bestModel <- train(PHENO ~ ., data = train_processed,
                   method = bestAlgorithm,
                   trControl = fitControl,
                   tuneLength = gridSearch,
                   metric = "ROC")

## shut down multicore
stopCluster(cluster)
registerDoSEQ()

## now summarize and export
save(bestModel, file = paste(prefix,"_tuned_bestModel.RData",sep = ""))
varImpList <- varImp(bestModel, scale = F) # here we get the importance matrix for the best model unscaled
varImpTable <- as.matrix(varImpList$importance)
write.table(varImpTable, file = paste(prefix,"_tuned_varImp.tab",sep =""), quote = F, sep = "\t", col.names = F)
train_processed$predicted <- predict(bestModel, train_processed)
train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
trained$ID <- ID
write.table(trained, file = paste(prefix,"_tuned_trainingSetPredictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
ggsave(plot = overlayedRocs, filename = paste(prefix,"_tuned_plotRoc.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
ggsave(plot = densPlot, filename = paste(prefix,"_tuned_plotDensity.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
ggsave(plot = overlayedRocs, filename = paste(prefix,"_tuned_plotRocNoCI.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
confMat <- confusionMatrix(data = as.factor(trained$predicted), reference = as.factor(trained$PHENO), positive = "DISEASE")
sink(file = paste(prefix,"_tuned_confMat.txt",sep =""), type = c("output"))
confMat
sink()
sink()

### sweep the floor, turn everything off and shut the door behind you
q("no")

