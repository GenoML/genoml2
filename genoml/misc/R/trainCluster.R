

## Load additional packages
packageList <- c("caret","ggplot2","data.table","rBayesianOptimization","plotROC","stringr")
lapply(packageList, library, character.only = TRUE)

models = NULL
models[["ALL"]] = c("dnn","lda","glm", "nnet","C5.0","glmnet","nb", "xgbLinear", "earth", "svmRadial", "lasso", 
                    "ridge", "evtree", "xgbTree", "rf", "bayesglm", "xgbDART")
models[["FAST"]] =  c("glm", "glmnet", "xgbLinear", "earth", "svmRadial", "lasso", "xgbTree", "xgbDART")
models[["FURIOUS"]] = c("glm", "glmnet", "xgbLinear", "earth", "svmRadial", "lasso")
models[["BOOSTED"]] = c("xgbLinear", "xgbTree", "xgbDART")

trainCluster_prepareData = function(geno,pheno,cov,split){
  covfile = paste0(cov,".cov")
  phenofile = paste0(pheno,".pheno")
  covdata = read.delim(covfile,stringsAsFactors=F)
  
  
}

mySystem = function(command,verbose=T,strict=T,intern=F){
  if(verbose)
    cat("The command to run is ",command,"\n")
  r = attr(system(command,intern=T),"status")
  if(strict & !is.null(r)){
    if(r > 0)
      stop(paste0("Something went wrong with the command ",command))  
  }
  print(r)
  
}

#' Title
#'
#' @param reduce 
#' @param herit 
#' @param geno 
#' @param pheno 
#' @param cov 
#' @param addit 
#' @param pathtoplink 
#'
#' @return
#' @export
#'
#' @examples
trainCluster_initData = function(geno,
                                 pheno,
                                 cov,
                                 addit="NA",
                                 reduce="PRSICE",
                                 gwas="RISK_noSpain.tab",
                                 herit=NA,
                                 path2GenoML="~/GenoML",
                                 path2plink=paste0(path2GenoML,"/otherPackages/"),
                                 path2gcta64=paste0(path2GenoML,"/otherPackages/"),
                                 cores=1,
                                 phenoScale="DISC",
                                 workPath=path2GenoML,
                                 path2PRSice=path2plink,
                                 path2GWAS=path2GenoML,
                                 path2Genotype=path2GenoML){
  
  ### options passed from list on draftCommandOptions.txt
  prefix=paste0("g-",geno,"-p-",pheno,"-c-",cov,"-a-",addit)
  fprefix = paste0(workPath,"/",prefix)
  
  if(reduce == "PRUNE" || reduce == "DEFAULT"){
    command = paste0(path2plink,"plink --bfile ",path2Genotype,"/",geno," --indep-pairwise 10000 1 0.1 --out ",fprefix,".temp")
    mySystem(command)
    command = paste0(path2plink,"plink --bfile ",path2Genotype,"/",geno," --extract ",fprefix,".temp.prune.in --recode A --out ",
                     fprefix,".reduced_genos")
    #cat("The command",command,"\n")
    mySystem(command)
    command = paste0("cut -f 1 ",fprefix,".temp.prune.in > ",fprefix,".reduced_genos_snpList")
    mySystem(command)
    
  }else if(reduce == "SBLUP" & !is.na(gwas) & !is.na(herit)){
    ### if $reduce = SBLUP, $gwas is not NA and $herit is not NA
    command = paste0("`wc -l < ",path2Genotype,"/",geno,".bim | awk '{print $1}'`")
    cat("The command",command,"\n")
    nsnps = as.numeric(system(command,intern=T))
    
    sbluplambda = nsnps * (1/herit) - 1
    
    #cat("The command",command,"\n")
    mySystem(command)
    command = paste0(path2plink,"plink --bfile ",path2Genotype,"/",geno," --pheno ",workPath,"/",pheno,".pheno --make-bed --out ",
                     workPath,"/",geno,".forSblup")
    #cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0(path2gcta64,"/gcta64 --bfile ",workPath,"/",geno,".forSblup --cojo-file ",path2GWAS,"/",gwas," --cojo-sblup ",sbluplambda,
                     " --cojo-wind 10000 --thread-num ",cores," --out ",fprefix,".temp")
    #cat("The command",command,"\n")
    mySystem(command)
    command = paste0(path2gcta64,"/gcta64 --bfile ",workPath,"/",geno,".forSblup --cojo-file ",path2GWAS,"/",gwas," --cojo-sblup ",sbluplambda,
                     " --cojo-wind 10000 --thread-num ",cores," --out ",fprefix,".temp")
    #cat("The command",command,"\n")
    mySystem(command)
    
    ## load SBLUP results
    sblupdata <- fread(paste(fprefix,".temp.sblup.cojo", sep = ""), header = F)
    
    ## start filters for sign matching and abs > 1 in sblup estimates to get ~25% data
    sblupdata$match <- ifelse(sign(sblupdata$V3) == sign(sblupdata$V4), 1, 0)
    sblupdata <- subset(sblupdata, match == 1 & abs(sblupdata$V4) > 1)
    names(sblupdata) <- c("SNP","effectAllele","gwasBeta","sblupBeta","effectMatch")
    ## export list of SNPs to pull
    fwrite(sblupdata, paste(fprefix,".sblupToPull", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
    
    command = paste0(path2plink,"/plink --bfile ",workPath,"/",geno,".forSblup --extract ",workPath,geno,".sblupToPull --indep-pairwise 10000 1 0.1 --out ",
                     fprefix,".pruning")
    #cat("The command",command,"\n")
    mySystem(command)
    
    
    command = paste0(path2plink,"/plink --bfile ",workPath,geno,".forSblup --extract ",fprefix,".pruning.prune.in --recode A --out ",
                     fprefix,".reduced_genos")
    #cat("The command",command,"\n")
    mySystem(command)
    # exports SNP list for extraction in validataion set
    command = paste0("cut -f 1 ",fprefix,".pruning.prune.in > ",fprefix,".reduced_genos_snpList; rm ",workPath,geno,".forSblup.*")
    #cat("The command",command,"\n")
    mySystem(command)
    
  }else if(reduce == "PRSICE" & !is.na(gwas)){
    ### if $reduce = PRSICE, $phenoScale is DISC, $gwas is not NA, $cov = NA
    
    ifelse(is.na(cov),covstr <- " ",covstr <- paste0(" --cov-file ",path2Genotype,"/",cov,".cov "))
    ifelse(phenoScale == "DISC",binaryTarget <- T,binaryTarget <- F)
    
    command = paste0("Rscript ",path2PRSice,"/PRSice.R --binary-target T --prsice ",path2PRSice,"/PRSice_linux -n ", 
                     cores, " --out ",fprefix,".temp --pheno-file ",path2Genotype,"/",pheno,".pheno -t ",path2Genotype,"/",geno," -b ",
                     path2GWAS,"/",gwas,covstr,
                     " --print-snp --score std --perm 10000 ",
                     " --bar-levels ",
                     "5E-8,4E-8,3E-8,2E-8,1E-8,9E-7,8E-7,7E-7,6E-7,5E-7,4E-7,3E-7,2E-7,1E-7,9E-6,8E-6,7E-6,6E-6,5E-6,4E-6,3E-6,2E-6,1E-6,9E-5,8E-5,7E-5,6E-5,5E-5,4E-5,3E-5,2E-5,1E-5,9E-4,8E-4,7E-4,6E-4,5E-4,4E-4,3E-4,2E-4,1E-4,9E-3,8E-3,7E-3,6E-3,5E-3,4E-3,3E-3,2E-3,1E-3,9E-2,8E-2,7E-2,6E-2,5E-2,4E-2,3E-2,2E-2,1E-2,9E-1,8E-1,7E-1,6E-1,5E-1,4E-1,3E-1,2E-1,1E-1,1 ", 
                     " --fastscore --binary-target ",binaryTarget," --beta --snp SNP --A1 A1 --A2 A2 --stat b --se se --pvalue p")
    #cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0("cut -f 2 ",fprefix,".temp.snp > ",fprefix,"prefix.temp.snpsToPull")
    #cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0("awk 'NR == 2 {print $3}' ",fprefix,".temp.summary")
    cat("The command",command,"\n")
    thresh = as.numeric(system(command,intern=T))
    
    
    command = paste0(path2plink,"/plink --bfile ",path2Genotype,"/",geno," --extract ",
                     fprefix,".temp.snpsToPull --clump ",path2GWAS,"/",gwas, 
                     " --clump-p1 ",thresh," --clump-p2 ",thresh,
                     " --clump-snp-field SNP --clump-field p --clump-r2 0.1 --clump-kb 250 --out ",
                     fprefix,".tempClumps")
    #cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0("cut -f 3 ",fprefix,".tempClumps.clumped > ",fprefix,".temp.snpsToPull2")
    #cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0(path2plink,"/plink --bfile ",path2Genotype,geno," --extract ",fprefix,".temp.snpsToPull2 --recode A --out ",fprefix,".reduced_genos")
    #cat("The command",command,"\n")
    mySystem(command)
    # exports SNP list for extraction in validataion set
    command = paste0("cut -f 1 ",fprefix,".temp.snpsToPull2 > ",fprefix,".reduced_genos_snpList")
    #cat("The command",command,"\n")
    mySystem(command)
    
  }else
    stop("The combination of parameters is not right")
  
  
  trainCluster_merge(geno,pheno,cov,addit="NA",prefix,workPath=workPath)
  
}


#' Title
#'
#' @param prefix 
#' @param outcome 
#' @param skipMLinit 
#' @param ncores 
#' @param trainSpeed 
#' @param cvReps 
#' @param gridSearch 
#' @param imputeMissingData 
#' @param testCov 
#' @param testPheno 
#' @param testAddit 
#' @param testGeno 
#' @param clLogPath 
#' @param workPath 
#' @param caretPath 
#' @param clParams 
#'
#' @return
#' @export
#'
#' @examples
trainCluster_runAtomic = function(prefix,
                                  outcome=c("cont","disc"),
                                  skipMLinit=F,
                                  ncores=1,
                                  trainSpeed="xgbTree",
                                  cvReps=10,
                                  gridSearch=30,
                                  imputeMissingData="median",
                                  testCov=NULL,
                                  testPheno=NULL,
                                  testAddit=NULL,
                                  testGeno=NULL,
                                  clLogPath="~/launch/",
                                  workPath="../",
                                  caretPath="/home/jbotia/caret/pkg/caret/",
                                  clParams=" -l h_rt=96:0:0 -l tmem=3G,h_vmem=3G "){
  detach(package:caret, unload=TRUE)
  library(devtools)
  options(bitmapType='cairo')
  cat("Loading caret from...",caretPath,"\n")
  load_all(caretPath)
  expid = trainSpeed
  ## Parse args and start logging
  prefixout = paste0(prefix,"_",expid)
  cat("Starting ML data initialization\n")
  train <- fread(paste(workPath,prefix,".dataForML", sep = ""), header = T)
  ### set outcome as a factor, check missingness, then impute missing data and scale
  if(outcome == "disc"){
    train$PHENO[train$PHENO == 2] <- "DISEASE"
    train$PHENO[train$PHENO == 1] <- "CONTROL"  
  }
  
  sum(is.na(train))
  ID <- train$ID
  train[,c("ID") := NULL]
  preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
  train_processed <- predict(preProcValues, train) # here we make the preprocessed values
  
  ## a bit more data munging that doesn't matter most of the time, we'll be ignoring this for now due to prefiltering but worth including in your considerations
  #nzv <- nearZeroVar(train_processed[,-1], saveMetrics= TRUE) # checks for near non zero and flags
  #descrCor <- cor(train_processed[,-1]) # builds correlation matrix
  #highlyCorDescr <- findCorrelation(descrCor, cutoff = .75, names = T, verbose = T) # checks correlated features and summarizes, can write function in later to drop
  #fwrite(nzv, paste(workPath,prefixout,".nzv", sep = ""), quote = F, sep = "\t", row.names = F, na = NA) # this exports the near and nonzero predictors
  #write.table(highlyCorDescr, paste(workPath,prefixout,".highCor", sep = ""), quote = F, sep = "\t", row.names = F, col.names = F) # this exports globally correlated predictors
  ## the two files above might want to be used to exclude some data prior to re-analysis
  #comboInfo <- findLinearCombos(train_processed) # finds biased linear combinations
  #fwrite(comboInfo, paste(prefix,".comboInfo", sep = ""), quote = F, sep = "\t", row.names = F, na = NA) # this exports the possible linear combinations in the dataset that may be problematic and should be examined
  
  if(outcome == "cont"){
    ## here we set the cross validation
    fitControl <- trainControl(method = "cv",
                               number = cvReps,
                               search = "random")
    print("This is a continuous run\n")
    ## Check command arguments from user
    print(paste("DATA PREEFIX file set as ", prefix, sep = ""))
    print(paste("RUNNIN ON HOW MANY CORES ??? ", ncores, " cores, is that enough??? ", sep = ""))
    print(paste("RUNNNING CROSS VALIDATION FOR ", cvReps, " REPS", sep = ""))
    print(paste("GRID SEARCH FOR ", gridSearch, " ITERATIONS", sep = ""))
    print(paste("ANOTHER LEVEL OF IMPUTATION AND DATa TRANSFORMATION USING ", imputeMissingData, sep = ""))
    
    model <- train(PHENO ~ ., data = train_processed,
                   method = trainSpeed,
                   trControl = fitControl,
                   tuneLength = gridSearch,
                   metric = "Rsquared", 
                   maximize = TRUE)
    
  }else{
    
    ## Load dataset
    ## Check command arguments from user
    print("This is a discrete run\n")
    print(paste("DATA PREEFIX file set as ", prefix, sep = ""))
    print(paste("RUNNIN ON HOW MANY CORES ??? ", ncores, " cores, is that enough??? ", sep = ""))
    print(paste("RUNNNING CROSS VALIDATION FOR ", cvReps, " REPS", sep = ""))
    print(paste("GRID SEARCH FOR ", gridSearch, " ITERATIONS", sep = ""))
    print(paste("ANOTHER LEVEL OF IMPUTATION AND DATa TRANSFORMATION USING ", imputeMissingData, sep = ""))
    
    
    ## here we set the cross validation
    cat("Setting trainControl object\n")
    fitControl <- trainControl(method = "cv",
                               number = cvReps,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary,
                               search = "random",
                               clLogPath=clLogPath,
                               clParams=clParams)
    
    cat("Calling train and then wait\n")
    model <- train(PHENO ~ ., data = train_processed,
                   method = trainSpeed,
                   trControl = fitControl,
                   tuneLength = gridSearch,
                   metric = "ROC")
    cat("Processing discrete stuff\n")
    ## pick best model from model compare then output plots in this case, its picked via ROC, 
    # maximizing the mean AUC across resamplings
    bestAlgorithm <- trainSpeed
    write.table(bestAlgorithm, file = paste(workPath,prefixout,"_bestModel.algorithm",sep = ""), quote = F, row.names = F, col.names = F) # exports "method" option for the best algorithm
    bestModel <- model
    save(bestModel, file = paste(workPath,prefixout,"_bestModel.RData",sep = ""))
    
    train_processed$predicted <- predict(bestModel, train_processed)
    train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
    train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
    train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
    trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
    trained$ID <- train$ID
    write.table(trained, file = paste(workPath,prefixout,"_trainingSetPredictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
    overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
    ggsave(plot = overlayedRocs, filename = paste(workPath,prefixout,"_plotRoc.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
    densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
    ggsave(plot = densPlot, filename = paste(workPath,prefixout,"_plotDensity.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
    overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
    ggsave(plot = overlayedRocs, filename = paste(workPath,prefixout,"_plotRocNoCI.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
    
    print(str(bestModel))
    varImpList <- varImp(bestModel, scale = F) # here we get the importance matrix for the best model unscaled
    print(varImpList)
    varImpTable <- as.matrix(varImpList$importance)
    print(varImpTable)
    write.table(varImpTable, file = paste(workPath,prefixout,"_varImp.tab",sep =""), quote = F, sep = "\t", col.names = F)
    
    confMat <- confusionMatrix(data = as.factor(trained$predicted), reference = as.factor(trained$PHENO), positive = "DISEASE")
    sink(file = paste(workPath,prefixout,"_confMat.txt",sep =""), type = c("output"))
    print(confMat)
    sink()
    
    trainCluster_merge(testGeno,testPheno,testCov,testAddit,prefix,workPath)
    testprefix <- paste("g",testGeno,"p",testPheno,"c",testCov,"a",testAddit,sep = "-")
    
    print(paste("DATA PREEFIX file for VALIDATION set as ", testprefix, sep = ""))
    print(paste("ANOTHER LEVEL OF IMPUTATION AND DATA TRANSFORMATION USING ", imputeMissingData, sep = ""))
    #Let us check headers
    trainCluster_checkVariantNames(testprefix,prefix)
    ## Load dataset
    train <- fread(paste(workPath,testprefix,".dataForML", sep = ""), header = T)
    sink(file = paste(workPath,testprefix, "_discreteValidation.Rout", sep = ""), type = c("output"))
    ### set outcome as a factor, check missingness, then impute missing data and scale
    train$PHENO[train$PHENO == 2] <- "DISEASE"
    train$PHENO[train$PHENO == 1] <- "CONTROL"
    ID <- train$ID
    train[,c("ID") := NULL]
    preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
    train_processed <- predict(preProcValues, train) # here we make the preprocessed values
    
    train_processed$predicted <- predict(bestModel, train_processed)
    train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
    train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
    train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
    trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
    trained$ID <- ID
    write.table(trained, file = paste(workPath,testprefix,"_validation_predictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
    overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
    ggsave(plot = overlayedRocs, filename = paste(workPath,testprefix,"_validation_plotRoc.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
    densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
    ggsave(plot = densPlot, filename = paste(workPath,testprefix,"_validation_plotDensity.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
    overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
    ggsave(plot = overlayedRocs, filename = paste(workPath,testprefix,"_validation_plotRocNoCI.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
    confMat <- confusionMatrix(data = as.factor(trained$predicted), reference = as.factor(trained$PHENO), positive = "DISEASE")
    sink(file = paste(workPath,testprefix,"_validation_confMat.txt",sep =""), type = c("output"))
    print(confMat)
    sink()
    
    
    
  }
}

trainCluster_launch = function(expid,prefix,
                               outcome=c("cont","disc"),
                               skipMLinit=F,
                               ncores=1,
                               trainSpeed="ALL",
                               cvReps=5,
                               gridSearch=30,
                               imputeMissingData="median"){
  ## Parse args and start logging
  
  if(!skipMLinit){
    cat("Starting ML data initialization\n")
    train <- fread(paste(workPath,prefix,".dataForML", sep = ""), header = T)
    ### set outcome as a factor, check missingness, then impute missing data and scale
    if(outcome == "disc"){
      train$PHENO[train$PHENO == 2] <- "DISEASE"
      train$PHENO[train$PHENO == 1] <- "CONTROL"  
    }
    
    sum(is.na(train))
    ID <- train$ID
    train[,c("ID") := NULL]
    preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
    train_processed <- predict(preProcValues, train) # here we make the preprocessed values
    
    ## a bit more data munging that doesn't matter most of the time, we'll be ignoring this for now due to prefiltering but worth including in your considerations
    nzv <- nearZeroVar(train_processed[,-1], saveMetrics= TRUE) # checks for near non zero and flags
    descrCor <- cor(train_processed[,-1]) # builds correlation matrix
    highlyCorDescr <- findCorrelation(descrCor, cutoff = .75, names = T, verbose = T) # checks correlated features and summarizes, can write function in later to drop
    fwrite(nzv, paste(workPath,prefix,".nzv", sep = ""), quote = F, sep = "\t", row.names = F, na = NA) # this exports the near and nonzero predictors
    write.table(highlyCorDescr, paste(workPath,prefix,".highCor", sep = ""), quote = F, sep = "\t", row.names = F, col.names = F) # this exports globally correlated predictors
    ## the two files above might want to be used to exclude some data prior to re-analysis
    #comboInfo <- findLinearCombos(train_processed) # finds biased linear combinations
    #fwrite(comboInfo, paste(prefix,".comboInfo", sep = ""), quote = F, sep = "\t", row.names = F, na = NA) # this exports the possible linear combinations in the dataset that may be problematic and should be examined
    
    if(outcome == "cont")
      ## here we set the cross validation
      fitControl <- trainControl(method = "cv",
                                 number = cvReps,
                                 search = "random")
    else
      ## here we set the cross validation
      fitControl <- trainControl(method = "cv",
                                 number = cvReps,
                                 classProbs = TRUE,
                                 summaryFunction = twoClassSummary,
                                 search = "random")
    token = list(train_processed=train_processed,fitControl=fitControl)
    saveRDS(token,paste(workPath,prefix,".train_processed.rds", sep = ""))
    
    
  }else
    cat("Skipping ML data initialization\n")
  
  
  
  for(model in models[[trainSpeed]]){
    newprefix = paste0(prefix,"_",expid,"_",model)
    cat("We launch now ",model," algorithm\n")
    command = paste0("echo \"cd /SAN/neuroscience/WT_BRAINEAC/ml/nalls/otherPackages/; ")
    if(outcome == "cont")
      command = paste0(command,"Rscript -e \\\"source(\\\\\\\"trainCluster.R\\\\\\\"); trainContCluster_run(",
                       "prefix=\\\\\\\"",prefix,"\\\\\\\"",
                       ",method=","\\\\\\\"",model,"\\\\\\\"",
                       ",ncores=",ncores,
                       ",cvReps=",cvReps,
                       ",gridSearch=",gridSearch,
                       ",imputeMissingData=","\\\\\\\"",imputeMissingData,"\\\\\\\")",
                       "\\\"\" | qsub -S /bin/bash -cwd -N ",paste0(expid,".",model),
                       " -l h_rt=24:0:0 -l tmem=7.9G,h_vmem=7.9G",
                       " -o /home/jbotia/launch/",newprefix,
                       ".log -e /home/jbotia/launch/",newprefix,".e")
    else
      command = paste0(command,"Rscript -e \\\"source(\\\\\\\"trainCluster.R\\\\\\\"); trainDiscCluster_run(",
                       "prefix=\\\\\\\"",prefix,"\\\\\\\"",
                       ",method=","\\\\\\\"",model,"\\\\\\\"",
                       ",ncores=",ncores,
                       ",cvReps=",cvReps,
                       ",gridSearch=",gridSearch,
                       ",imputeMissingData=","\\\\\\\"",imputeMissingData,"\\\\\\\")",
                       "\\\"\" | qsub -S /bin/bash -cwd -N ",paste0(expid,".",model),
                       " -l h_rt=24:0:0 -l tmem=7.9G,h_vmem=7.9G",
                       " -o /home/jbotia/launch/",newprefix,
                       ".log -e /home/jbotia/launch/",newprefix,".e")
    cat("The command",command,"\n")
    system(command)
  }
}

trainDiscCluster_run = function(prefix,method,ncores,cvReps,gridSearch,imputeMissingData){
  ## Load dataset
  ## Check command arguments from user
  print("This is a discrete run\n")
  print(paste("DATA PREEFIX file set as ", prefix, sep = ""))
  print(paste("RUNNIN ON HOW MANY CORES ??? ", ncores, " cores, is that enough??? ", sep = ""))
  print(paste("RUNNNING CROSS VALIDATION FOR ", cvReps, " REPS", sep = ""))
  print(paste("GRID SEARCH FOR ", gridSearch, " ITERATIONS", sep = ""))
  print(paste("ANOTHER LEVEL OF IMPUTATION AND DATa TRANSFORMATION USING ", imputeMissingData, sep = ""))
  
  token = readRDS(paste(workPath,prefix,".train_processed.rds", sep = ""))
  train_processed = token$train_processed
  fitControl = token$fitControl
  
  model <- train(PHENO ~ ., data = train_processed,
                 method = method,
                 trControl = fitControl,
                 tuneLength = gridSearch,
                 metric = "ROC")
  
  traintokenfile = paste0(workPath,prefix,"_",method,".rds")
  saveRDS(model,traintokenfile)
  
}

trainContCluster_run = function(prefix,method,ncores,cvReps,gridSearch,imputeMissingData){
  
  print("This is a continuous run\n")
  ## Check command arguments from user
  print(paste("DATA PREEFIX file set as ", prefix, sep = ""))
  print(paste("RUNNIN ON HOW MANY CORES ??? ", ncores, " cores, is that enough??? ", sep = ""))
  print(paste("RUNNNING CROSS VALIDATION FOR ", cvReps, " REPS", sep = ""))
  print(paste("GRID SEARCH FOR ", gridSearch, " ITERATIONS", sep = ""))
  print(paste("ANOTHER LEVEL OF IMPUTATION AND DATa TRANSFORMATION USING ", imputeMissingData, sep = ""))
  
  
  token = readRDS(paste(workPath,prefix,".train_processed.rds", sep = ""))
  train_processed = token$train_processed
  fitControl = token$fitControl
  
  model <- train(PHENO ~ ., data = train_processed,
                 method = method,
                 trControl = fitControl,
                 tuneLength = gridSearch,
                 metric = "Rsquared", 
                 maximize = TRUE)
  
  traintokenfile = paste0(workPath,prefix,"_",method,".rds")
  saveRDS(model,traintokenfile)
}


trainCluster_resume = function(prefix,trainSpeed,outcome=c("cont","disc"),fromAtomic=F)
{
  #sink(file = paste(prefix, "_continuousTraining.Rout", sep = ""))
  ##Juan's add
  options(bitmapType='cairo')
  
  if(!fromAtomic)
    train_processed = readRDS(paste(workPath,prefix,".train_processed.rds", sep = ""))$train_processed
  
  cat("Resuming ",prefix,"after the cluster run with speed",trainSpeed,"\n")
  methods = models[[trainSpeed]]
  modelsRan = NULL
  for(method in methods){
    if(fromAtomic){
      #g-CALLS_B_Train-p-pheno_BTrain-c-covs_BTrain-a-NA_xgbTree_bestModel.RData
      filein = paste0(workPath,prefix,"_",method,"_bestModel.RData")
      if(file.exists(filein)){
        load(filein)
        modelsRan[[method]] =  bestModel
        cat("Just readed ",filein,"\n")
      }else
        cat("Model Atomic file",filein,"for method",method," is missing\n")  
      
    }else{
      
      filein = paste0(workPath,prefix,"_",method,".rds")
      if(file.exists(filein)){
        modelsRan[[method]] =  readRDS(filein)
        cat("Just readed ",filein,"\n")
      }
      
      else
        cat("Model file",filein,"for method",method," is missing\n")
    }
  }
  if(!is.null(modelsRan)){
    ## now compare best models
    sink(file = paste(workPath,prefix,"_methodPerformance.tab",sep =""), type = c("output"))
    methodComparisons <- resamples(modelsRan)
    print(summary(methodComparisons))
    sink()
    sink(file = paste(workPath,prefix,"_methodTimings.tab",sep =""), type = c("output"))
    print(methodComparisons$timings)
    sink()
    
    if(outcome == "cont"){
      cat("Processing continuous stuff\n")
      
      ## pick best model from model compare then output plots in this case, its picked via ROC, maximizing the mean AUC across resamplings
      evalMetric <- as.matrix(methodComparisons, metric = methodComparisons$metric[3]) # default metric here is R2
      meanEvalMetric <- as.data.frame(colMeans(evalMetric, na.rm = T))
      meanEvalMetric$method <- rownames(meanEvalMetric)
      names(meanEvalMetric)[1] <- "meanR2s" 
      bestEvalMetric <- subset(meanEvalMetric, meanR2s == max(meanEvalMetric$meanR2s))
      bestAlgorithm <- paste(bestEvalMetric[1,2])
      write.table(bestAlgorithm, file = paste(workPath,prefix,"_bestModel.algorithm",sep = ""), quote = F, row.names = F, col.names = F) # exports "method" option for the best algorithm
      bestModel <- modelsRan[[bestAlgorithm]]
      save(bestModel, file = paste(workPath,prefix,"_bestModel.RData",sep = ""))
      varImpList <- varImp(bestModel, scale = F) # here we get the importance matrix for the best model unscaled
      varImpTable <- as.matrix(varImpList$importance)
      write.table(varImpTable, file = paste(workPath,prefix,"_varImp.tab",sep =""), quote = F, sep = "\t", col.names = F)
      train_processed$predicted <- predict(bestModel, train_processed)
      trained <- train_processed[,c("PHENO","predicted")]
      train <- fread(paste(workPath,prefix,".dataForML", sep = ""), header = T)
      trained$ID <- train$ID
      write.table(trained, file = paste(workPath,prefix,"_trainingSetPredictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
      meanDiff <- format(mean(abs(trained$predicted - trained$PHENO)), digits = 3, scientific = T)
      regPlot <- ggplot(trained, aes(x=PHENO, y=predicted)) + geom_point(shape=1) + geom_smooth(method=lm, se=T) + 
        ggtitle(paste("mean abs(observed-predicted) = ", meanDiff,sep = "")) + xlab("Observed") + ylab("Predicted") + theme_bw()
      ggsave(plot = regPlot, filename = paste(workPath,prefix,"_plotRegs.png",sep =""), width = 8, height = 8, units = "in", dpi = 300)
      
    }else{
      cat("Processing discrete stuff\n")
      ## pick best model from model compare then output plots in this case, its picked via ROC, 
      # maximizing the mean AUC across resamplings
      ROCs <- as.matrix(methodComparisons, metric = methodComparisons$metric[1])
      meanROCs <- as.data.frame(colMeans(ROCs, na.rm = T))
      meanROCs$method <- rownames(meanROCs)
      names(meanROCs)[1] <- "meanROC" 
      bestFromROC <- subset(meanROCs, meanROC == max(meanROCs$meanROC))
      bestAlgorithm <- bestFromROC[1,2]
      write.table(bestAlgorithm, file = paste(workPath,prefix,"_bestModel.algorithm",sep = ""), quote = F, row.names = F, col.names = F) # exports "method" option for the best algorithm
      bestModel <- modelsRan[[bestAlgorithm]]
      save(bestModel, file = paste(workPath,prefix,"_bestModel.RData",sep = ""))
      
      if(!fromAtomic){
        train_processed$predicted <- predict(bestModel, train_processed)
        train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
        train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
        train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
        trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
        
        train <- fread(paste(workPath,prefix,".dataForML", sep = ""), header = T)
        trained$ID <- train$ID
        write.table(trained, file = paste(workPath,prefix,"_trainingSetPredictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
        overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
        ggsave(plot = overlayedRocs, filename = paste(workPath,prefix,"_plotRoc.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
        densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
        ggsave(plot = densPlot, filename = paste(workPath,prefix,"_plotDensity.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
        overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
        ggsave(plot = overlayedRocs, filename = paste(workPath,prefix,"_plotRocNoCI.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
        
        varImpList <- varImp(bestModel, scale = F) # here we get the importance matrix for the best model unscaled
        varImpTable <- as.matrix(varImpList$importance)
        write.table(varImpTable, file = paste(workPath,prefix,"_varImp.tab",sep =""), quote = F, sep = "\t", col.names = F)
        
        confMat <- confusionMatrix(data = as.factor(trained$predicted), reference = as.factor(trained$PHENO), positive = "DISEASE")
        sink(file = paste(workPath,prefix,"_confMat.txt",sep =""), type = c("output"))
        print(confMat)
        sink()
      }
      
      
    }
    ### sweep the floor, turn everything off and shut the door behind you
  }
}


trainCluster_variantScale = function(prefix,gwas,geno){
  library(data.table)
  
  #sink(file = paste(workPath,prefix, "_alleleDosageScaling.Rout", sep = ""))
  
  ## now read in the data
  cat("Reading ",paste(workPath,geno, ".bim", sep = ""),"\n")
  mapFile <- fread(file = paste(workPath,geno, ".bim", sep = ""))
  cat("Reading ",paste(workPath,prefix, ".temp.snpsToPull2", sep = ""),"\n")
  snpList <- fread(file = paste(workPath,prefix, ".temp.snpsToPull2", sep = ""))
  cat("Reading ",paste0(workPath,gwas),"\n")
  gwasResults <- fread(file = paste0(workPath,gwas))
  cat("Reading ",paste(workPath,prefix, ".reduced_genos.raw", sep = ""),"\n")
  genoDoses <- fread(file = paste(workPath,prefix, ".reduced_genos.raw", sep = ""))
  cat("Done\n")
  ## now make some tweaks to the GWAS
  gwasResults$mafWeight <- ifelse(gwasResults$freq < 0.5, gwasResults$b, -1*gwasResults$b)
  cat("GWAS modified\n")
  print(head(gwasResults))
  
  ## build the index file
  mapReduced <- mapFile[,c("V2","V5")]
  names(mapReduced) <- c("SNP","minorAllele")
  snpsReduced <- snpList[,c("SNP","CHR","BP","P")]
  gwasReduced <- gwasResults[,c("SNP","mafWeight")]
  cat("Merging\n")
  varIndexTemp <-merge(mapReduced, snpsReduced, by = "SNP")
  print(head(varIndexTemp))
  cat("Done\n")
  varIndex <- merge(varIndexTemp, gwasReduced, by = "SNP")
  print(head(varIndex))
  cat("Done\n")
  fwrite(varIndex,file = paste(workPath,prefix, "variantWeightings", sep = "."), quote = F, sep = "\t", row.names = F, na = NA)
  varIndex$rawSnpName <- paste(varIndex$SNP, varIndex$minorAllele, sep = "_")
  
  ## now recode the *.raw file with new weights for minor alleles
  for (i in 1:length(varIndex$rawSnpName)) 
  {
    thisVar <- varIndex$rawSnpName[i]
    thisVarWeight <- varIndex$mafWeight[i]
    thisVarWeighted <- genoDoses[[thisVar]]*thisVarWeight
    genoDoses[[thisVar]] <- thisVarWeighted
  }
  
  ## here we just output the weighted file over the old one, not the best idea but...
  fwrite(genoDoses, file = paste(workPath,prefix, ".reduced_genos.raw", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
  
  ## shut'em down
  #sink()
}


#Let us check headers
trainCluster_checkVariantNames = function(testprefix,oldprefix){
  
  traindata = paste0(workPath,oldprefix,".dataForML")
  testdata = paste0(workPath,testprefix,".dataForML")
  
  cat("Checking possible change of main allele between train ",traindata,
      " and test data ",testdata,"\n")
  train = fread(traindata, header = T)
  test = fread(testdata,header=T)
  snps = grep(":",colnames(train))
  snps = colnames(train)[snps]
  sentinel = F
  for(snp in snps){
    if(!(snp %in% colnames(test))){
      cat(snp," from traning data is not at test data\n")
      thesplit = str_split(snp,"_")
      commonpart = thesplit[[1]][[1]]
      letter = thesplit[[1]][[2]]
      testpos = grep(commonpart,colnames(test))
      wrongsnp = colnames(test)[testpos]
      test[,testpos] = 2 - test[,..testpos]
      
      cat("Converting ",wrongsnp," into ",paste0(commonpart,"_",letter),"\n")
      colnames(test)[testpos] = paste0(commonpart,"_",letter)
      sentinel = T
    }
  }
  if(sentinel){
    cat("Saving ",testdata," again to disk\n")
    fwrite(test, file = testdata, quote = F, sep = "\t", row.names = F, na = NA)  
  }
  else
    cat("No need to change any variant name, identical column nanes for test/train data\n")
  
}


trainCluster_validate = function(geno,
                                 pheno,
                                 cov,
                                 addit="NA",
                                 outcome,
                                 oldprefix,
                                 useAllModels=F,
                                 imputeMissingData="median",
                                 trainSpeed="ALL"){
  
  trainCluster_merge(geno,pheno,cov,addit,oldprefix)
  testprefix <- paste("g",geno,"p",pheno,"c",cov,"a",addit, sep = "-")
  
  ##Juan's add
  options(bitmapType='cairo')
  print(paste("DATA PREEFIX file set as ", testprefix, sep = ""))
  print(paste("ANOTHER LEVEL OF IMPUTATION AND DATA TRANSFORMATION USING ", imputeMissingData, sep = ""))
  
  ## Load additional packages
  packageList <- c("caret","ggplot2","data.table","rBayesianOptimization","plotROC")
  lapply(packageList, library, character.only = TRUE)
  
  #Let us check headers
  trainCluster_checkVariantNames(testprefix,oldprefix)
  
  ## Load dataset
  train <- fread(paste(workPath,testprefix,".dataForML", sep = ""), header = T)
  
  
  if(outcome == "disc"){
    sink(file = paste(workPath,testprefix, "_discreteValidation.Rout", sep = ""), type = c("output"))
    ### set outcome as a factor, check missingness, then impute missing data and scale
    train$PHENO[train$PHENO == 2] <- "DISEASE"
    train$PHENO[train$PHENO == 1] <- "CONTROL"
    ID <- train$ID
    train[,c("ID") := NULL]
    preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
    train_processed <- predict(preProcValues, train) # here we make the preprocessed values
    
    if(!useAllModels){
      m = colnames(read.delim(paste0(workPath,oldprefix,"_bestModel.algorithm"),stringsAsFactors=F))[1]
      cat("The best model is ",m,'\n')
      ## find out what the best model from earlier tune and load
      oldname = paste0(workPath,oldprefix,"_bestModel.RData")
      newname = paste0(workPath,m,"_bestModel.RData")
      if(file.exists(newname) & !file.exists(oldname)){
        cat("WARNING: File ", newname," exists already\n")
      }else{
        cat("Changing name of model from ",oldname," to ",newname,"\n")
        file.copy(from=oldname,to=newname,overwrite=T)
      }
      load(newname) # this loads in "bestModel"
      ## now summarize and export
      train_processed$predicted <- predict(bestModel, train_processed)
      train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
      train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
      train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
      trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
      trained$ID <- ID
      write.table(trained, file = paste(workPath,testprefix,"_validation_predictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
      overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
      ggsave(plot = overlayedRocs, filename = paste(workPath,testprefix,"_validation_plotRoc.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
      densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
      ggsave(plot = densPlot, filename = paste(workPath,testprefix,"_validation_plotDensity.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
      overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
      ggsave(plot = overlayedRocs, filename = paste(workPath,testprefix,"_validation_plotRocNoCI.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
      confMat <- confusionMatrix(data = as.factor(trained$predicted), reference = as.factor(trained$PHENO), positive = "DISEASE")
      sink(file = paste(workPath,testprefix,"_validation_confMat.txt",sep =""), type = c("output"))
      print(confMat)
      sink()
    }else{ 
      #We will use all models
      localmodels = 
        for(localmodel in models[[trainSpeed]]){
          filein = paste0(workPath,oldprefix,"_",localmodel,"_bestModel.RData")
          
          if(file.exists(filein)){
            cat("Assessing ",filein,"\n")
            load(filein)
            ## now summarize and export
            train_processed$predicted <- predict(bestModel, train_processed)
            train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
            train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
            train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
            trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
            trained$ID <- ID
            write.table(trained, file = paste(workPath,testprefix,"_",localmodel,"_validation_predictions.tab",sep =""), 
                        quote = F, sep = "\t", row.names = F)
            overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + 
              geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
            ggsave(plot = overlayedRocs, filename = paste(workPath,testprefix,"_",localmodel,"_validation_plotRoc.png",sep =""), 
                   width = 8, height = 5, units = "in", dpi = 300)
            densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
            ggsave(plot = densPlot, filename = paste(workPath,testprefix,"_",localmodel,"_validation_plotDensity.png",sep =""), 
                   width = 8, height = 5, units = "in", dpi = 300)
            overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + 
              style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
            ggsave(plot = overlayedRocs, filename = paste(workPath,testprefix,"_",localmodel,"_validation_plotRocNoCI.png",sep =""), 
                   width = 8, height = 5, units = "in", dpi = 300)
            confMat <- confusionMatrix(data = as.factor(trained$predicted), 
                                       reference = as.factor(trained$PHENO), positive = "DISEASE")
            sink(file = paste(workPath,testprefix,"_",localmodel,"_validation_confMat.txt",sep =""), type = c("output"))
            print(confMat)
            sink()  
          }else{
            cat("File",filein,"not found, skipping the model\n")
          }
          
          
        }
      
    }
    sink()
    
  }else{
    sink(file = paste(workPath,testprefix, "_continuousValdiation.Rout", sep = ""), type = c("output"))
    
    
    ## Load additional packages
    packageList <- c("caret","ggplot2","data.table","rBayesianOptimization","plotROC")
    lapply(packageList, library, character.only = TRUE)
    
    ## Load dataset
    train <- fread(paste(workPath,testprefix,".dataForML", sep = ""), header = T)
    
    ## impute missing data and scale
    ID <- train$ID
    train[,c("ID") := NULL]
    preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
    train_processed <- predict(preProcValues, train) # here we make the preprocessed values
    m = read.delim(paste0(workPath,oldprefix,"_bestModel.algorithm"),stringsAsFactors=F)
    ## find out what the best model from earlier tune and load
    oldname = paste0(workPath,oldprefix,"_bestModel.RData")
    newname = paste0(workPath,m,"_bestModel.RData")
    if(file.exists(newname) & !file.exists(oldname)){
      cat("WARNING: File ", newname," exists already\n")
    }else{
      cat("Changing name of model from ",oldname," to ",newname,"\n")
      file.copy(from=oldname,to=newname,overwrite=T)
    }
    load(newname) # this loads in "bestModel"
    
    ## now summarize and export
    train_processed$predicted <- predict(bestModel, train_processed)
    trained <- train_processed[,c("PHENO","predicted")]
    trained$ID <- ID
    write.table(trained, file = paste(workPath,testprefix,"_validation_predictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
    meanDiff <- format(mean(abs(trained$predicted - trained$PHENO)), digits = 3, scientific = T)
    regPlot <- ggplot(trained, aes(x=PHENO, y=predicted)) + geom_point(shape=1) + geom_smooth(method=lm, se=T) + ggtitle(paste("mean abs(observed-predicted) = ", meanDiff,sep = "")) + xlab("Observed") + ylab("Predicted") + theme_bw()
    ggsave(plot = regPlot, filename = paste(workPath,testprefix,"_validation_plotRegs.png",sep =""), width = 8, height = 8, units = "in", dpi = 300)
    sink()
  }
}

trainCluster_merge = function(geno,pheno,cov,addit,previous,workPath){
  
  prefix <- paste("g",geno,"p",pheno,"c",cov,"a",addit, sep = "-")
  
  command = paste0("plink --bfile ",workPath,"/",geno," --keep ",workPath,"/",cov,".cov" ,
                   " --extract ",workPath,"/",previous,
                   ".reduced_genos_snpList --recode A --out ",workPath,"/", prefix,".reduced_genos")
  
  cat("Running command ",command,"\n")
  system(command)
  
  ## Merges in R start by the commands below acessing additional scripts
  #Rscript $pathToGenoML/otherPackages/mergeForGenoML.R $geno $pheno $cov $addit
  ## if phenoScale = CNT
  #Rscript $pathToGenoML/otherPackages/validateCont.R $prefix $cores $imputeMissingData $bestModelName
  
  print(prefix)
  #sink(file = paste(prefix, "_mergeInput.Rout", sep = ""), type = c("output"))
  print(paste("GENO PREEFIX file set as ", geno, sep = ""))
  print(paste("PHENO PREEFIX file set as ", pheno, sep = ""))
  print(paste("COV PREEFIX file set as ", cov, sep = "")) # if not specified load NA
  print(paste("ADDIT PREEFIX file set as ", addit, sep = "")) # if not specified load NA
  print(paste("ALL RESULTS TAGGED WITH PREFIX -> ", prefix, sep = ""))
  
  ## load packages
  library("data.table")
  ### now decide what to merge
  genoPheno <- 2
  addCov <- ifelse(cov == "NA", 0, 1)
  addAddit <- ifelse(addit == "NA", 0, 1)
  nFiles <- genoPheno + addCov + addAddit # this specifies the number of files to merge
  print(paste("MERGING ", nFiles," FILES", sep = ""))
  genotypeInput <- paste(workPath,prefix, ".reduced_genos.raw", sep = "")
  phenoInput <- paste(workPath,pheno, ".pheno", sep = "")
  covInput <- paste(workPath,cov, ".cov", sep = "")
  additInput <- paste(workPath,addit, ".addit", sep = "")
  
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
    fwrite(temp, file = paste(workPath,prefix,".dataForML", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
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
    fwrite(temp3, file = paste(workPath,prefix,".dataForML", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
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
    fwrite(temp2, file = paste(workPath,prefix,".dataForML", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
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
    fwrite(temp2, file = paste(workPath,prefix,".dataForML", sep = ""), quote = F, sep = "\t", row.names = F, na = NA)
    print("First 100 variable names for your file below, the rest are likely just more genotypes...")
    print(head(names(temp2), n = 100))
    print("... and the last 100 variable names for your file below...")
    print(tail(names(temp2), n = 100))
    print(paste("Your final file has ", length(temp2$ID)," samples, and ", length(names(temp2))," predictors for analysis", sep = ""))	
  }
  
  ### save the log file
  sink()
}

