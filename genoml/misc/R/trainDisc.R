## Parse args and start logging
args <- commandArgs()
print(args)
prefix <- args[6]
sink(file = paste(prefix, "_discreteTraining.Rout", sep = ""))
ncores <- as.numeric(args[7])
trainSpeed <- args[8]
cvReps <- as.numeric(args[9])
gridSearch <- as.numeric(args[10])
imputeMissingData <- args[11]

## Check command arguments from user
print(paste("DATA PREEFIX file set as ", prefix, sep = ""))
print(paste("RUNNIN ON HOW MANY CORES ??? ", ncores, " cores, is that enough??? ", sep = ""))
print(paste("TRAINING SPEED is ", trainSpeed, sep = ""))
print(paste("RUNNNING CROSS VALIDATION FOR ", cvReps, " REPS", sep = ""))
print(paste("GRID SEARCH FOR ", gridSearch, " ITERATIONS", sep = ""))
print(paste("ANOTHER LEVEL OF IMPUTATION AND DATA TRANSFORMATION USING ", imputeMissingData, sep = ""))

## Load additional packages
packageList <- c("caret","ggplot2","data.table","rBayesianOptimization","plotROC")
lapply(packageList, library, character.only = TRUE)

## Load dataset
train <- fread(paste(prefix,".dataForML", sep = ""), header = T)

### set outcome as a factor, check missingness, then impute missing data and scale
train$PHENO[train$PHENO == 2] <- "DISEASE"
train$PHENO[train$PHENO == 1] <- "CONTROL"
sum(is.na(train))
ID <- train$ID
train[,c("ID") := NULL]
preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
train_processed <- predict(preProcValues, train) # here we make the preprocessed values

## a bit more data munging that doesn't matter most of the time, we'll be ignoring this for now due to prefiltering but worth including in your considerations
nzv <- nearZeroVar(train_processed[,-1], saveMetrics= TRUE) # checks for near non zero and flags
fwrite(nzv, paste(prefix,".nzv", sep = ""), quote = F, sep = "\t", row.names = F, na = NA) # this exports the near and nonzero predictors
if(length(names(train)) < 1001) # if more than 1000 variables this step is skipped for memeory and speed reasons
{
descrCor <- cor(train_processed[,-1]) # builds correlation matrix
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75, names = T, verbose = T) # checks correlated features and summarizes, can write function in later to drop
write.table(highlyCorDescr, paste(prefix,".highCor", sep = ""), quote = F, sep = "\t", row.names = F, col.names = F) # this exports globally correlated predictors
}
## the two files above might want to be used to exclude some data prior to re-analysis
#comboInfo <- findLinearCombos(train_processed) # finds biased linear combinations
#fwrite(comboInfo, paste(prefix,".comboInfo", sep = ""), quote = F, sep = "\t", row.names = F, na = NA) # this exports the possible linear combinations in the dataset that may be problematic and should be examined

### to begin tune first begin parallel parameters 
library("parallel")
library("doParallel")
cluster <- makeCluster(ncores) # convention to leave 1 core for OS
registerDoParallel(cluster)
set.seed(123) # makes generally reproducible grid searches from random or bayes tunes at paramter optimization

## here we set the cross validation
fitControl <- trainControl(method = "cv",
	number = cvReps,
	classProbs = TRUE,
	summaryFunction = twoClassSummary,
	search = "random")

## now we start making models if the trainSpeed is ALL
if(trainSpeed == "ALL")
{

	glm.model <- train(PHENO ~ ., data = train_processed,
		method = "glm",
		trControl = fitControl,
		tuneLength = gridSearch,
		metric = "ROC")

	bayesglm.model <- train(PHENO ~ ., data = train_processed,
		method = "bayesglm",
		trControl = fitControl,
		tuneLength = gridSearch,
		metric = "ROC")

	xgbTree.model <- train(PHENO ~ ., data = train_processed, 
		method = "xgbTree", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	xgbDART.model <- train(PHENO ~ ., data = train_processed, 
		method = "xgbDART", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	rf.model <- train(PHENO ~ ., data = train_processed, 
		method = "rf", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	nb.model <- train(PHENO ~ ., data = train_processed, 
		method = "nb", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	nnet.model <- train(PHENO ~ ., data = train_processed, 
		method = "nnet", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	dnn.model <- train(PHENO ~ ., data = train_processed, 
		method = "dnn", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	C5.0.model <- train(PHENO ~ ., data = train_processed, 
		method = "C5.0", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	glmnet.model <- train(PHENO ~ ., data = train_processed, 
		method = "glmnet", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	svmRadial.model <- train(PHENO ~ ., data = train_processed, 
		method = "svmRadial", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	lda.model <- train(PHENO ~ ., data = train_processed, 
		method = "lda", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	modelsRan <- list(glm=glm.model, bayesglm=bayesglm.model, xgbTree=xgbTree.model, xgbDART=xgbDART.model, rf=rf.model, nb=nb.model, nnet=nnet.model, dnn=dnn.model, C5.0=C5.0.model, glmnet=glmnet.model, svmRadial=svmRadial.model, lda=lda.model)
}

## now we start making models if the trainSpeed is FAST
if(trainSpeed == "FAST")
{

	glm.model <- train(PHENO ~ ., data = train_processed,
		method = "glm",
		trControl = fitControl,
		tuneLength = gridSearch,
		metric = "ROC")

	xgbTree.model <- train(PHENO ~ ., data = train_processed, 
		method = "xgbTree", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	xgbDART.model <- train(PHENO ~ ., data = train_processed, 
		method = "xgbDART", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	nb.model <- train(PHENO ~ ., data = train_processed, 
		method = "nb", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	nnet.model <- train(PHENO ~ ., data = train_processed, 
		method = "nnet", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	dnn.model <- train(PHENO ~ ., data = train_processed, 
		method = "dnn", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	glmnet.model <- train(PHENO ~ ., data = train_processed, 
		method = "glmnet", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	lda.model <- train(PHENO ~ ., data = train_processed, 
		method = "lda", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	modelsRan <- list(glm=glm.model, xgbTree=xgbTree.model, xgbDART=xgbDART.model, nb=nb.model, nnet=nnet.model, dnn=dnn.model, glmnet=glmnet.model, lda=lda.model)
}

## now we start making models if the trainSpeed is FURIOUS
if(trainSpeed == "FURIOUS")
{

	glm.model <- train(PHENO ~ ., data = train_processed,
		method = "glm",
		trControl = fitControl,
		tuneLength = gridSearch,
		metric = "ROC")

	nb.model <- train(PHENO ~ ., data = train_processed, 
		method = "nb", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	nnet.model <- train(PHENO ~ ., data = train_processed, 
		method = "nnet", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	dnn.model <- train(PHENO ~ ., data = train_processed, 
		method = "dnn", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	glmnet.model <- train(PHENO ~ ., data = train_processed, 
		method = "glmnet", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	lda.model <- train(PHENO ~ ., data = train_processed, 
		method = "lda", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	modelsRan <- list(glm=glm.model, nb=nb.model, nnet=nnet.model, dnn=dnn.model, glmnet=glmnet.model, lda=lda.model)
}

## now we start making models if the trainSpeed is BOOSTED
if(trainSpeed == "BOOSTED")
{

	xgbTree.model <- train(PHENO ~ ., data = train_processed, 
		method = "xgbTree", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	xgbDART.model <- train(PHENO ~ ., data = train_processed, 
		method = "xgbDART", 
		trControl = fitControl, 
		tuneLength = gridSearch,
		metric = "ROC")

	modelsRan <- list(xgbTree=xgbTree.model, xgbDART=xgbDART.model)
}

### shut down multicore
stopCluster(cluster)
registerDoSEQ()

### now compare best models
sink(file = paste(prefix,"_methodPerformance.tab",sep =""), type = c("output"))
methodComparisons <- resamples(modelsRan)
summary(methodComparisons)
sink()
sink(file = paste(prefix,"_methodTimings.tab",sep =""), type = c("output"))
methodComparisons$timings
sink()

## pick best model from model compare then output plots in this case, its picked via ROC, maximizing the mean AUC across resamplings
ROCs <- as.matrix(methodComparisons, metric = methodComparisons$metric[1])
meanROCs <- as.data.frame(colMeans(ROCs, na.rm = T))
meanROCs$method <- rownames(meanROCs)
names(meanROCs)[1] <- "meanROC" 
bestFromROC <- subset(meanROCs, meanROC == max(meanROCs$meanROC))
bestAlgorithm <- paste(bestFromROC[1,2])
write.table(bestAlgorithm, file = paste(prefix,"_bestModel.algorithm",sep = ""), quote = F, row.names = F, col.names = F) # exports "method" option for the best algorithm
bestModel <- get(paste(bestAlgorithm, ".model", sep = ""))
save(bestModel, file = paste(prefix,"_bestModel.RData",sep = ""))
varImpList <- varImp(bestModel, scale = F) # here we get the importance matrix for the best model unscaled
varImpTable <- as.matrix(varImpList$importance)
write.table(varImpTable, file = paste(prefix,"_varImp.tab",sep =""), quote = F, sep = "\t", col.names = F)
train_processed$predicted <- predict(bestModel, train_processed)
train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
trained$ID <- ID
write.table(trained, file = paste(prefix,"_trainingSetPredictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
ggsave(plot = overlayedRocs, filename = paste(prefix,"_plotRoc.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
ggsave(plot = densPlot, filename = paste(prefix,"_plotDensity.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
ggsave(plot = overlayedRocs, filename = paste(prefix,"_plotRocNoCI.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
confMat <- confusionMatrix(data = as.factor(trained$predicted), reference = as.factor(trained$PHENO), positive = "DISEASE")
sink(file = paste(prefix,"_confMat.txt",sep =""), type = c("output"))
confMat
sink()
sink()

### sweep the floor, turn everything off and shut the door behind you
q("no")
