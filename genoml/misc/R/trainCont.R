## Parse args and start logging
args <- commandArgs()
print(args)
prefix <- args[6]
sink(file = paste(prefix, "_continuousTraining.Rout", sep = ""))
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
print(paste("ANOTHER LEVEL OF IMPUTATION AND DATa TRANSFORMATION USING ", imputeMissingData, sep = ""))

## Load additional packages
packageList <- c("caret","ggplot2","data.table","rBayesianOptimization","plotROC")
lapply(packageList, library, character.only = TRUE)

## Load dataset
train <- fread(paste(prefix,".dataForML", sep = ""), header = T)

## check missingness, then impute missing data and scale
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
            search = "random")

## now we start making models if the trainSpeed is ALL
if(trainSpeed == "ALL")
{
	
	glm.model <- train(PHENO ~ ., data = train_processed,
	       method = "glm",
	       trControl = fitControl,
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	bayesglm.model <- train(PHENO ~ ., data = train_processed,
	       method = "bayesglm",
	       trControl = fitControl,
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbTree.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbTree", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbLinear.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbLinear", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbDART.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbDART", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	rf.model <- train(PHENO ~ ., data = train_processed, 
	       method = "rf", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       ## Specify which metric to optimize
	       metric = "Rsquared", maximize = TRUE)
	
	ridge.model <- train(PHENO ~ ., data = train_processed, 
	       method = "ridge", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	evtree.model <- train(PHENO ~ ., data = train_processed, 
	       method = "evtree", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	glmnet.model <- train(PHENO ~ ., data = train_processed, 
	       method = "glmnet", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       ## Specify which metric to optimize
	       metric = "Rsquared", maximize = TRUE)
	
	svmRadial.model <- train(PHENO ~ ., data = train_processed, 
	       method = "svmRadial", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	earth.model <- train(PHENO ~ ., data = train_processed, 
	       method = "earth", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	lasso.model <- train(PHENO ~ ., data = train_processed, 
	       method = "lasso", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	modelsRan <- list(glm=glm.model, bayesglm=bayesglm.model, xgbTree=xgbTree.model, xgbDART=xgbDART.model, xgbLinear=xgbLinear.model, rf=rf.model, ridge=ridge.model, evtree=evtree.model, glmnet=glmnet.model, svmRadial=svmRadial.model, earth=earth.model, lasso=lasso.model)
}

## now we start making models if the trainSpeed is FAST
if(trainSpeed == "FAST")
{
	
	glm.model <- train(PHENO ~ ., data = train_processed,
	       method = "glm",
	       trControl = fitControl,
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbTree.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbTree", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbLinear.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbLinear", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbDART.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbDART", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
				
	glmnet.model <- train(PHENO ~ ., data = train_processed, 
	       method = "glmnet", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	svmRadial.model <- train(PHENO ~ ., data = train_processed, 
	       method = "svmRadial", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	earth.model <- train(PHENO ~ ., data = train_processed, 
	       method = "earth", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	lasso.model <- train(PHENO ~ ., data = train_processed, 
	       method = "lasso", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	modelsRan <- list(glm=glm.model, xgbTree=xgbTree.model, xgbDART=xgbDART.model, xgbLinear=xgbLinear.model, glmnet=glmnet.model, svmRadial=svmRadial.model, earth=earth.model, lasso=lasso.model)
}

## now we start making models if the trainSpeed is FURIOUS
if(trainSpeed == "FURIOUS")
{
	
	glm.model <- train(PHENO ~ ., data = train_processed,
	       method = "glm",
	       trControl = fitControl,
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbLinear.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbLinear", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	glmnet.model <- train(PHENO ~ ., data = train_processed, 
	       method = "glmnet", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       ## Specify which metric to optimize
	       metric = "Rsquared", maximize = TRUE)
	
	svmRadial.model <- train(PHENO ~ ., data = train_processed, 
	       method = "svmRadial", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	earth.model <- train(PHENO ~ ., data = train_processed, 
	       method = "earth", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	lasso.model <- train(PHENO ~ ., data = train_processed, 
	       method = "lasso", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	modelsRan <- list(glm=glm.model, xgbLinear=xgbLinear.model, glmnet=glmnet.model, svmRadial=svmRadial.model, earth=earth.model, lasso=lasso.model)
}

## now we start making models if the trainSpeed is BOOSTED
if(trainSpeed == "BOOSTED")
{
	
	xgbTree.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbTree", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbLinear.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbLinear", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	xgbDART.model <- train(PHENO ~ ., data = train_processed, 
	       method = "xgbDART", 
	       trControl = fitControl, 
	       tuneLength = gridSearch,
	       metric = "Rsquared", maximize = TRUE)
	
	modelsRan <- list(xgbTree=xgbTree.model, xgbDART=xgbDART.model, xgbLinear=xgbLinear.model)
}

## shut down multicore
stopCluster(cluster)
registerDoSEQ()

## now compare best models
sink(file = paste(prefix,"_methodPerformance.tab",sep =""), type = c("output"))
methodComparisons <- resamples(modelsRan)
summary(methodComparisons)
sink()
sink(file = paste(prefix,"_methodTimings.tab",sep =""), type = c("output"))
methodComparisons$timings
sink()

## pick best model from model compare then output plots in this case, its picked via ROC, maximizing the mean AUC across resamplings
evalMetric <- as.matrix(methodComparisons, metric = methodComparisons$metric[3]) # default metric here is R2
meanEvalMetric <- as.data.frame(colMeans(evalMetric, na.rm = T))
meanEvalMetric$method <- rownames(meanEvalMetric)
names(meanEvalMetric)[1] <- "meanR2s" 
bestEvalMetric <- subset(meanEvalMetric, meanR2s == max(meanEvalMetric$meanR2s))
bestAlgorithm <- paste(bestEvalMetric[1,2])
write.table(bestAlgorithm, file = paste(prefix,"_bestModel.algorithm",sep = ""), quote = F, row.names = F, col.names = F) # exports "method" option for the best algorithm
bestModel <- get(paste(bestAlgorithm, ".model", sep = ""))
save(bestModel, file = paste(prefix,"_bestModel.RData",sep = ""))
varImpList <- varImp(bestModel, scale = F) # here we get the importance matrix for the best model unscaled
varImpTable <- as.matrix(varImpList$importance)
write.table(varImpTable, file = paste(prefix,"_varImp.tab",sep =""), quote = F, sep = "\t", col.names = F)
train_processed$predicted <- predict(bestModel, train_processed)
trained <- train_processed[,c("PHENO","predicted")]
trained$ID <- ID
write.table(trained, file = paste(prefix,"_trainingSetPredictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
meanDiff <- format(mean(abs(trained$predicted - trained$PHENO)), digits = 3, scientific = T)
regPlot <- ggplot(trained, aes(x=PHENO, y=predicted)) + geom_point(shape=1) + geom_smooth(method=lm, se=T) + ggtitle(paste("mean abs(observed-predicted) = ", meanDiff,sep = "")) + xlab("Observed") + ylab("Predicted") + theme_bw()
ggsave(plot = regPlot, filename = paste(prefix,"_plotRegs.png",sep =""), width = 8, height = 8, units = "in", dpi = 300)
sink()

### sweep the floor, turn everything off and shut the door behind you
q("no")
