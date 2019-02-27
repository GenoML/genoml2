## Parse args and start logging
args <- commandArgs()
print(args)
prefix <- args[6]
sink(file = paste(prefix, "_continuousValdiation.Rout", sep = ""), type = c("output"))
ncores <- as.numeric(args[7])
imputeMissingData <- args[8]
bestModelName <- args[9]

## Check command arguments from user
print(paste("DATA PREEFIX file set as ", prefix, sep = ""))
print(paste("RUNNIN ON HOW MANY CORES ??? ", ncores, " cores, is that enough??? ", sep = ""))
print(paste("ANOTHER LEVEL OF IMPUTATION AND DATA TRANSFORMATION USING ", imputeMissingData, sep = ""))
print(paste("BEST MODEL FROM STEP 2/3 file set as ", bestModelName, sep = ""))

## Load additional packages
packageList <- c("caret","ggplot2","data.table","rBayesianOptimization","plotROC")
lapply(packageList, library, character.only = TRUE)

## Load dataset
train <- fread(paste(prefix,".dataForML", sep = ""), header = T)

## impute missing data and scale
ID <- train$ID
train[,c("ID") := NULL]
preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
train_processed <- predict(preProcValues, train) # here we make the preprocessed values

## find out what the best model from earlier tune and load
loadingModel <- paste(bestModelName, "_bestModel.RData", sep = "")
load(loadingModel) # this loads in "bestModel"

## now summarize and export
train_processed$predicted <- predict(bestModel, train_processed)
trained <- train_processed[,c("PHENO","predicted")]
trained$ID <- ID
write.table(trained, file = paste(prefix,"_validation_predictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
meanDiff <- format(mean(abs(trained$predicted - trained$PHENO)), digits = 3, scientific = T)
regPlot <- ggplot(trained, aes(x=PHENO, y=predicted)) + geom_point(shape=1) + geom_smooth(method=lm, se=T) + ggtitle(paste("mean abs(observed-predicted) = ", meanDiff,sep = "")) + xlab("Observed") + ylab("Predicted") + theme_bw()
ggsave(plot = regPlot, filename = paste(prefix,"_validation_plotRegs.png",sep =""), width = 8, height = 8, units = "in", dpi = 300)
sink()

### sweep the floor, turn everything off and shut the door behind you
q("no")
