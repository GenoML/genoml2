

#' Title
#'
#' @param geno
#' @param covs
#' @param predictor
#' @param pheno
#'
#' @return
#' @export
#'
#' @examples
getHandlerToGenotypeData = function(geno,covs,predictor,id,fid,pheno){
  
  stopifnot(file.exists(paste0(geno,".bed")))
  stopifnot(file.exists(paste0(geno,".bim")))
  stopifnot(file.exists(paste0(geno,".fam")))
  stopifnot(file.exists(paste0(covs,".cov")))
  cdata = read.table(paste0(covs,".cov"),header = TRUE,stringsAsFactors=F)
  #cat("Covariates are",paste0(colnames(cdata),collapse=", "),"\n")
  phenofile = paste0(pheno, ".pheno")
  if(!file.exists(phenofile))
    stopifnot(predictor %in% colnames(cdata))
  stopifnot(id %in% colnames(cdata))
  stopifnot(fid %in% colnames(cdata))
  
  #Generating the phenotype file
  if(file.exists(phenofile)){
    phenoDat = read.table(phenofile,header = TRUE,stringsAsFactors=F)
  }else{
    phenoDat = cdata[,c(fid,id,predictor)]
    write.table(phenoDat,phenofile,
                sep = " ", row.names = FALSE, quote = FALSE)
    
  }
  
  #Removing the phenotype from the covariates
  cdata = cdata[,colnames(cdata) != predictor]
  
  genoHandler = NULL
  genoHandler$geno = geno
  genoHandler$pheno = pheno
  genoHandler$covs = covs
  genoHandler$id = id
  genoHandler$fid = fid
  genoHandler$covsDat =  cdata
  genoHandler$phenoDat = phenoDat
  genoHandler$Class = predictor
  attr(genoHandler,"class") = "genohandler"
  return(genoHandler)
}

#' Title
#'
#' @param handler 
#' @param type 
#' @param index 
#'
#' @return
#' @export
#'
#' @examples
getHandlerFromFold = function(handler,type="train",index=1){
  key = paste0(type,"FoldFiles")
  if(type == "test"){
    stopifnot(!is.null(handler$testFolds))
    stopifnot(length(handler$testFolds) >= index)
  }
  if(type == "train"){
    stopifnot(!is.null(handler$trainFolds))
    stopifnot(length(handler$testFolds) >= index)
  }
  stopifnot(!is.null(handler[[key]]))
  files = handler[[key]][[index]]  
  handler = getHandlerToGenotypeData(geno=files$genoFile,
                                     covs=gsub(".cov","",files$covsFile),
                                     predictor=handler$Class,
                                     id=handler$id,
                                     fid=handler$fid,
                                     pheno=gsub(".pheno","",files$phenoFile))
  return(handler)
}

#' Title
#'
#' @param h 
#' @param level 
#'
#' @return
#' @export
#'
#' @examples
verifyHandler = function(h,level=1){
  if(level >= 1){
    stopifnot(file.exists(paste0(h$geno,".bed")))
    stopifnot(file.exists(paste0(h$geno,".bim")))
    stopifnot(file.exists(paste0(h$geno,".fam")))
    stopifnot(file.exists(paste0(h$covs,".cov")))
    cdata = read.table(paste0(h$covs,".cov"),header = TRUE)
    stopifnot(h$id %in% colnames(cdata))
    stopifnot(file.exists(paste0(h$pheno,".pheno")))
  }
  if(level >= 2){
    stopifnot(!is.null(h$trainFolds))
    stopifnot(!is.null(h$testFolds))
  }
}

cleanHandler = function(h,level=1){
  file.remove(paste0(h$geno,".bed"))
  file.remove(paste0(h$geno,".bim"))
  file.remove(paste0(h$geno,".fam"))
  file.remove(paste0(h$covs,".cov"))
  file.remove(paste0(h$pheno,".pheno"))
  if(!is.null(h$nfolds)){
    for(i in 1:h$nfolds){
      cleanHandler(getHandlerFromFold(handler=h,type="train",index=i))
      cleanHandler(getHandlerFromFold(handler=h,type="test",index=i))
    }
  }
}

#' Title
#'
#' @param handler 
#'
#' @return
#' @export
#'
#' @examples
print.genohandler = function(handler){
  cat("A verified GenoML handler with the following features\n")
  cat("Genotype files (bed/bim/fam) prefix",handler$geno,"\n")
  cat("Covariates file",paste0(handler$covs,".cov"),"\n")
  cat("Class attribute in the covariates",handler$Class,"\n")
  cat(nrow(handler$covsDat),"examples and columns",paste0(colnames(handler$covsDat),collapse=", "),"\n")
  cat("Predictor values count\n")
  print(table(handler$phenoDat[,handler$Class]))
  if(!is.null(handler$trainFolds)){
    if(handler$plan == "k-fold cv"){
      cat("K-fold evaluation with k =",
          length(handler$trainFolds),
          "and test split proportion",
          signif(length(handler$testFolds[[1]])/length(handler$trainFolds[[1]]),2),"\n")
    }else
      cat("Hold out cross-validation with test split proportion",
          signif(length(handler$testFolds[[1]])/length(handler$trainFolds[[1]]),2),"\n")
    
    if(is.null(handler$trainFoldFiles) & is.null(handler$testFoldFiles))
      cat("No data was generated from the data partitions created\n")
    else{
      if(!is.null(handler$trainFoldFiles)){
        cat(length(handler$trainFoldFiles),"sets of files were created for training. See below for the 1st one\n")
        print(handler$trainFoldFiles[[1]])
      }
      if(!is.null(handler$testFoldFiles)){
        cat(length(handler$testFoldFiles),"sets of files were created for test. See below for the 1st one\n")
        print(handler$testFoldFiles[[1]])
      }
    }
  }
  
  
}



#' Title
#'
#' @param genoHandler A handler obtained by calling to `getHandlerToGenotypeData()`
#' @param how We can get a sample with "holdout" and "k-fold cv"
#' @param k The number of folds when we use cross-validation
#' @param p The proportion of data that does to training
#'
#' @return The same handler, with two new fields, `trainFolds` and `testFolds` with the indexes
#' @export
#'
#' @examples
getSamplesFromHandler = function(genoHandler,how="k-fold cv",k=10,p=0.75){
  
  stopifnot(how == "k-fold cv" | how == "holdout")
  genoHandler$plan = how
  
  library(caret)
  pred <- genoHandler$Class #accessing the column given by the user
  
  if(how == "holdout"){
    hoFold = createDataPartition(y=genoHandler$phenoDat[[h$Class]],p=p)
    genoHandler$folds = hoFold
    genoHandler$nfolds = 1
    genoHandler$trainFolds = NULL
    genoHandler$testFolds = NULL
    genoHandler$trainFolds$train1 = hoFold$Resample1
    genoHandler$testFolds$test1 = (1:length(genoHandler$phenoDat[[h$Class]]))[-hoFold$Resample1]
  }else{
    
    folds <- createFolds(y=genoHandler$covsDat[[pred]], k=k, list = TRUE, returnTrain = FALSE)
    genoHandler$folds = folds
    genoHandler$nfolds = length(folds)
    
    trainFolds <- list() #list with the indexes used for train in each one of the k cross-validations
    testFolds <- list()  #list with the indexes saved for test in each one of the k cross-validations
    for (i in 1:genoHandler$nfolds){
      trainFolds[[paste0("train",i)]] = NULL
      trainFolds[[paste0("train",i)]] = (1:length(genoHandler$phenoDat[[h$Class]]))[-folds[[names(folds[i])]]]
      testFolds[[paste0("test",i)]] = NULL
      testFolds[[paste0("test",i)]] = folds[[names(folds[i])]]
    }
    genoHandler$trainFolds = trainFolds
    genoHandler$testFolds = testFolds
  }
  return(genoHandler)
}



#' Title Creating your genotipe data
#'
#' @param genoHandler 
#' @param workPath 
#' @param path2plink 
#' @param onlyFold 
#' @param which.to.create 
#'
#' @return
#' @export
#'
#' @examples
genData = function(genoHandler,
                   workPath = "~/genoml-core-master",
                   path2plink="~/genoml-core-master/otherPackages/",
                   onlyFold=-1,
                   onlyFiles=F,
                   which.to.create=c("train","test")){
  verifyHandler(genoHandler)
  stopifnot(dir.exists(workPath))
  if(path2plink != "")
    stopifnot(dir.exists(path2plink))
  
  if(onlyFold > 0)
    indexes = onlyFold
  else
    indexes = 1:genoHandler$nfolds
  for(tt in which.to.create){
    key = paste0(tt,"FoldFiles")
    genoHandler[[key]] = NULL
    
  }
  
  for (i in indexes){
    foldFiles = NULL
    #generation of a new directory for each iteration
    for(tt in which.to.create){
      key = paste0(tt,"FoldFiles")
      dir.create(file.path(workPath, paste0("Fold",tt,i)), showWarnings = FALSE)
      dirFold = paste0(workPath,"/", paste0("Fold",tt,i))
      
      splitIdx <- paste0(tt,i)
      covsFoldDT <- genoHandler$covsDat[genoHandler[[paste0(tt,"Folds")]][[splitIdx]],] #dataframe with the covariates of the individuals given by this fold
      
      covsFold <- paste0("COVS_",tt,i)
      covsFile = file = paste0(dirFold, "/" ,covsFold,".cov")
      write.table(covsFoldDT, covsFile, sep = " ", row.names = FALSE, quote = FALSE) #file with the covs
      foldFiles$covsFile = covsFile
      
      genoFold = paste0(basename(genoHandler$geno),tt,".",i)
      phenoFold = paste0(basename(genoHandler$pheno),i)
      
      #generation of the geno file using plink
      idsfile = paste0(dirFold, "/",basename(genoHandler$covs),tt,".ids")
      write.table(genoHandler$covsDat[genoHandler[[paste0(tt,"Folds")]][[splitIdx]],c(genoHandler$fid,genoHandler$id)],idsfile,
                  col.names=F,sep = " ", row.names = FALSE, quote = FALSE)
      foldFiles$idsFile = idsfile
      
      #Generating the phenotype files
      phenoFoldDT <- genoHandler$phenoDat[genoHandler[[paste0(tt,"Folds")]][[splitIdx]],] #dataframe with the covariates of the individuals given by this fold
      phenofile = paste0(dirFold, "/", phenoFold,tt, ".pheno")
      write.table(phenoFoldDT,phenofile,
                  sep = " ", row.names = FALSE, quote = FALSE)
      foldFiles$phenoFile = phenofile
      
      #Generating the genotype files
      genofile = paste0(dirFold, "/", genoFold)
      command = paste0(path2plink,"plink --bfile ", genoHandler$geno, " --keep ", idsfile, " --make-bed --out ", genofile)
      r = mySystem(command)
      foldFiles$genoFile = genofile
      genoHandler[[key]][[1]] = foldFiles
    }
  }
  return(genoHandler)
}


#' Title
#'
#' @param genoHandler
#' @param how
#' @param workPath
#' @param path2plink
#'
#' @return
#' @export
#'
#' @examples
mostRelevantSNPs = function(handler,
                            how="PRSICE",
                            phenoScale="DISC",
                            SNPcolumnatGWAS="SNP",
                            clumpField="p",
                            path2plink,...){
  
  verifyHandler(handler)
  if(path2plink != "")
    stopifnot(dir.exists(path2plink))
  
  #call to the function that reproduces step1 of the pipeline and gets a dataForML por each one of the nfolds iterations
  trainCluster_initData(handler=handler,
                        reduce = how,
                        path2plink=path2plink,
                        path2gcta64=path2plink,
                        phenoScale=phenoScale,
                        SNPcolumnatGWAS=SNPcolumnatGWAS,
                        clumpField=clumpField,...)
  
}

trainCluster_initData = function(handler,
                                 addit="NA",
                                 reduce="PRSICE",
                                 gwas="RISK_noSpain.tab",
                                 SNPcolumnatGWAS="SNP",
                                 clumpField="p",
                                 herit=NA,
                                 path2plink,
                                 path2gcta64,
                                 cores=1,
                                 phenoScale="DISC",
                                 path2PRSice=path2plink,
                                 PRSiceexe="PRSice_linux",
                                 path2GWAS,...){
  
  geno = basename(handler$geno)
  pheno = basename(handler$pheno)
  workPath=paste0(dirname(handler$geno),"/")
  path2Genotype=paste0(dirname(handler$geno),"/")
  cov = basename(handler$covs)
  
  ### options passed from list on draftCommandOptions.txt
  prefix=paste0("g-",geno,"-p-",pheno,"-c-",cov,"-a-",addit)
  fprefix = paste0(workPath,"/",prefix)  ##CAMBIAR
  
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
    
    command = paste0(path2plink,"plink --bfile ",workPath,"/",geno,".forSblup --extract ",workPath,geno,".sblupToPull --indep-pairwise 10000 1 0.1 --out ",
                     fprefix,".pruning")
    #cat("The command",command,"\n")
    mySystem(command)
    
    
    command = paste0(path2plink,"plink --bfile ",workPath,geno,".forSblup --extract ",fprefix,".pruning.prune.in --recode A --out ",
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
    
    command = genPRSiceCommand(geno,
                               pheno,
                               covstr,
                               path2PRSice,
                               PRSiceexe,
                               cores,
                               fprefix,
                               path2Genotype,
                               path2GWAS,
                               gwas,
                               binaryTarget,
                               ...)
    
    cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0("cut -f 2 ",fprefix,".temp.snp > ",fprefix,".temp.snpsToPull")
    #cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0("awk 'NR == 2 {print $3}' ",fprefix,".temp.summary")
    cat("The command",command,"\n")
    thresh = as.numeric(system(command,intern=T))
    
    command = clumpCommand(geno,gwas,thresh,fprefix,path2plink,path2Genotype,path2GWAS,SNPcolumnatGWAS,clumpField)
    #cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0("cut -f 3 ",fprefix,".tempClumps.clumped > ",fprefix,".temp.snpsToPull2")
    #cat("The command",command,"\n")
    mySystem(command)
    
    command = paste0(path2plink,"plink --bfile ",path2Genotype,geno," --extract ",fprefix,".temp.snpsToPull2 --recode A --out ",fprefix,".reduced_genos")
    #cat("The command",command,"\n")
    mySystem(command)
    # exports SNP list for extraction in validataion set
    command = paste0("cut -f 1 ",fprefix,".temp.snpsToPull2 > ",fprefix,".reduced_genos_snpList")
    #cat("The command",command,"\n")
    mySystem(command)
    
    return(handler)
    
  }else
    stop("The combination of parameters is not right")
}







#' Title
#'
#' @param handler 
#' @param addit 
#' @param path2plink 
#'
#' @return
#' @export
#'
#' @examples
fromSNPs2MLdata = function(handler,addit,path2plink){
  
  if(!is.null(handler$nfolds)){
    modes = c("train","test")
  }else
    modes = "train"
  
  for(mode in modes){
    if(mode == "train" & length(modes) == 1){
      geno = basename(handler$geno)
      pheno = basename(handler$pheno)
      workPath=paste0(dirname(handler$geno),"/")
      path2Genotype=paste0(dirname(handler$geno),"/")
      cov = basename(handler$covs)
      prefix <- paste("g",geno,"p",pheno,"c",cov,"a",addit, sep = "-")
      
    }else if(mode == "train" & length(modes) > 1){
      geno = basename(handler$trainFoldFiles[[1]]$genoFile)
      pheno = basename(gsub(".pheno","",handler$trainFoldFiles[[1]]$phenoFile))
      workPath=paste0(dirname(handler$trainFoldFiles[[1]]$genoFile),"/")
      path2Genotype=workPath
      cov = basename(gsub(".cov","",handler$trainFoldFiles[[1]]$covsFile))
      prefix <- paste("g",geno,"p",pheno,"c",cov,"a",addit, sep = "-")
      
    }else if(mode == "test" & length(modes) > 1){
      geno = basename(handler$testFoldFiles[[1]]$genoFile)
      pheno = basename(gsub(".pheno","",handler$testFoldFiles[[1]]$phenoFile))
      workPath=paste0(dirname(handler$testFoldFiles[[1]]$genoFile),"/")
      path2Genotype=workPath
      cov = basename(gsub(".cov","",handler$testFoldFiles[[1]]$covsFile))
      
      #We need the previous prefix
      genotrn = basename(handler$trainFoldFiles[[1]]$genoFile)
      phenotrn = basename(gsub(".pheno","",handler$trainFoldFiles[[1]]$phenoFile))
      covtrn = basename(gsub(".cov","",handler$trainFoldFiles[[1]]$covsFile))
      workPathtrn=paste0(dirname(handler$trainFoldFiles[[1]]$genoFile),"/")
      previous <- paste("g",genotrn,"p",phenotrn,"c",covtrn,"a",addit, sep = "-")
      prefix <- paste("g",geno,"p",pheno,"c",cov,"a",addit, sep = "-")
      command = paste0(path2plink,"plink --bfile ",workPath,"/",geno," --keep ",workPath,"/",cov,".cov" ,
                       " --extract ",workPathtrn,"/",previous,
                       ".reduced_genos_snpList --recode A --out ",workPath,"/", prefix,".reduced_genos")
      
      cat("Running command ",command,"\n")
      system(command)
    }else
      stop()
    
    
    
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
      fname = paste(workPath,prefix,".dataForML", sep = "")
      fwrite(temp, file = fname, quote = F, sep = "\t", row.names = F, na = NA)
      print("First 100 variable names for your file below, the rest are likely just more genotypes...")
      print(head(names(temp), n = 100))
      print("... and the last 100 variable names for your file below...")
      print(tail(names(temp), n = 100))
      print(paste("Your final file has ", length(temp$ID)," samples, and ", length(names(temp))," predictors for analysis", sep = ""))
      handler[[paste0(mode,"mldata")]] = fname
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
      fname = paste(workPath,prefix,".dataForML", sep = "")
      fwrite(temp3, file = fname, quote = F, sep = "\t", row.names = F, na = NA)
      print("First 100 variable names for your file below, the rest are likely just more genotypes...")
      print(head(names(temp3), n = 100))
      print("... and the last 100 variable names for your file below...")
      print(tail(names(temp3), n = 100))
      print(paste("Your final file has ", length(temp3$ID)," samples, and ", length(names(temp3))," predictors for analysis", sep = ""))
      
      handler[[paste0(mode,"mldata")]] = fname
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
      fname = paste(workPath,prefix,".dataForML", sep = "")
      fwrite(temp2, file = fname, quote = F, sep = "\t", row.names = F, na = NA)
      print("First 100 variable names for your file below, the rest are likely just more genotypes...")
      print(head(names(temp2), n = 100))
      print("... and the last 100 variable names for your file below...")
      print(tail(names(temp2), n = 100))
      print(paste("Your final file has ", length(temp2$ID)," samples, and ", length(names(temp2))," predictors for analysis", sep = ""))
      handler[[paste0(mode,"mldata")]] = fname
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
      fname = paste(workPath,prefix,".dataForML", sep = "")
      fwrite(temp2, file = fname, quote = F, sep = "\t", row.names = F, na = NA)
      print("First 100 variable names for your file below, the rest are likely just more genotypes...")
      print(head(names(temp2), n = 100))
      print("... and the last 100 variable names for your file below...")
      print(tail(names(temp2), n = 100))
      print(paste("Your final file has ", length(temp2$ID)," samples, and ", length(names(temp2))," predictors for analysis", sep = ""))
      handler[[paste0(mode,"mldata")]] = fname
    }
    
    
  }
  return(handler)
}

#' Title
#'
#' @param prefix 
#' @param outcome 
#' @param skipMLinit 
#' @param ncores 
#' @param learningAlgorithm
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
genoMLtrainAndTest = function(handler,
                              outcome=c("cont","disc"),
                              skipMLinit=F,
                              ncores=1,
                              learningAlgorithm="xgbTree",
                              cvReps=10,
                              gridSearch=30,
                              imputeMissingData="median",
                              useCluster=T,
                              clLogPath="~/launch/",
                              caretPath="/home/jbotia/caret/pkg/caret/",
                              clParams=" -l h_rt=96:0:0 -l tmem=3G,h_vmem=3G "){
  stopifnot(handler$nfolds == 1)
  detach(package:caret, unload=TRUE)
  if(useCluster){
    library(devtools)
    cat("Loading caret from...",caretPath,"\n")
    load_all(caretPath)
    
  }else
    library(caret)
  library(plotROC)
  prefix = gsub(".dataForML","",basename(handler$trainmldata))
  workPath = dirname(handler$trainmldata)
  options(bitmapType='cairo')
  expid = learningAlgorithm
  ## Parse args and start logging
  prefixout = paste0(prefix,"_",expid)
  cat("Starting ML data initialization\n")
  train <- fread(handler$trainmldata, header = T)
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
  
  if(outcome == "cont")
    stop("Continuous mode not implemented")
  
  
  
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
  if(useCluster){
    clusterwd = paste0(workPath,"/cluster/")
    if(!dir.exists(clusterwd))
      dir.create(clusterwd)
    fitControl <- trainControl(method = "cv",
                               number = cvReps,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary,
                               search = "random",
                               clLogPath=clusterwd,
                               clParams=clParams)
    
  }else
    fitControl <- trainControl(method = "cv",
                               number = cvReps,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary,
                               search = "random")
  
  cat("Calling train and then wait\n")
  model <- train(PHENO ~ ., 
                 data = train_processed,
                 workPath=workPath,
                 method = learningAlgorithm,
                 trControl = fitControl,
                 tuneLength = gridSearch,
                 metric = "ROC")
  
  cat("Processing discrete stuff\n")
  ## pick best model from model compare then output plots in this case, its picked via ROC, 
  # maximizing the mean AUC across resamplings
  bestAlgorithm <- learningAlgorithm
  write.table(bestAlgorithm, file = paste(workPath,prefixout,"_bestModel.algorithm",sep = ""), 
              quote = F, row.names = F, col.names = F) # exports "method" option for the best algorithm
  handler$bestAlgorithm = bestAlgorithm
  handler$bestModel = model
  bestModel <- model
  
  train_processed$predicted <- predict(bestModel, train_processed)
  train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
  train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
  train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
  trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
  trained$ID <- train$ID
  write.table(trained, file = paste(workPath,prefixout,"_trainingSetPredictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
  handler$trainingPredictions = trained
  overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
  ggsave(plot = overlayedRocs, filename = paste(workPath,prefixout,"_plotRoc.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
  densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
  ggsave(plot = densPlot, filename = paste(workPath,prefixout,"_plotDensity.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
  overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
  ggsave(plot = overlayedRocs, filename = paste(workPath,prefixout,"_plotRocNoCI.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
  
  # print("entering varImp")
  # varImpList <- varImp(bestModel, scale = F) # here we get the importance matrix for the best model unscaled
  # print(varImpList)
  # if(!is.null(varImpList)){
  #   print(varImpList)
  #   varImpTable <- as.matrix(varImpList$importance)
  #   print(varImpTable)
  #   write.table(varImpTable, file = paste(workPath,prefixout,"_varImp.tab",sep =""), quote = F, sep = "\t", col.names = F)
  # }
  # handler$varImportance = varImpTable
  
  confMat <- confusionMatrix(data = as.factor(trained$predicted), reference = as.factor(trained$PHENO), positive = "DISEASE")
  handler$confMat = confMat
  testGeno = basename(handler$testFoldFiles[[1]]$genoFile)
  testPheno = basename(gsub(".pheno","",handler$testFoldFiles[[1]]$phenoFile))
  workPath=paste0(dirname(handler$testFoldFiles[[1]]$genoFile),"/")
  testCov = basename(gsub(".cov","",handler$testFoldFiles[[1]]$covsFile))
  testAddit = "NA"
  testprefix <- paste("g",testGeno,"p",testPheno,"c",testCov,"a",testAddit,sep = "-")
  
  print(paste("DATA PREEFIX file for VALIDATION set as ", testprefix, sep = ""))
  print(paste("ANOTHER LEVEL OF IMPUTATION AND DATA TRANSFORMATION USING ", imputeMissingData, sep = ""))
  #Let us check headers
  checkVariantNames(handler$trainmldata,handler$testmldata)
  ## Load dataset
  train <- fread(handler$testmldata, header = T)
  
  ### set outcome as a factor, check missingness, then impute missing data and scale
  train$PHENO[train$PHENO == 2] <- "DISEASE"
  train$PHENO[train$PHENO == 1] <- "CONTROL"
  ID <- train$ID
  train[,c("ID") := NULL]
  preProcValues <- preProcess(train[,-1], method = c(paste(imputeMissingData,"Impute", sep = ""))) 
  # note here we pick impute method (KNN or median),  we can also exclude near zero variance predictors and correlated predictors
  train_processed <- predict(preProcValues, train) # here we make the preprocessed values
  
  train_processed$predicted <- predict(bestModel, train_processed)
  train_processed$probDisease <- predict(bestModel, train_processed, type = "prob")[2]
  train_processed$diseaseBinomial <- ifelse(train_processed$PHENO == "DISEASE", 1, 0)
  train_processed$predictedBinomial <- ifelse(train_processed$predicted == "DISEASE", 1, 0)
  trained <- train_processed[,c("PHENO","predicted","probDisease","diseaseBinomial","predictedBinomial")]
  trained$ID <- ID
  handler$evalPredictions = trained
  write.table(trained, file = paste(workPath,testprefix,"_validation_predictions.tab",sep =""), quote = F, sep = "\t", row.names = F)
  overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + geom_rocci() + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
  ggsave(plot = overlayedRocs, filename = paste(workPath,testprefix,"_validation_plotRoc.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
  densPlot <- ggplot(trained, aes(probDisease, fill = PHENO, color = PHENO)) + geom_density(alpha = 0.5) + theme_bw()
  ggsave(plot = densPlot, filename = paste(workPath,testprefix,"_validation_plotDensity.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
  overlayedRocs <- ggplot(trained, aes(d = diseaseBinomial, m = probDisease)) + geom_roc(labels = FALSE) + style_roc(theme = theme_gray) + theme_bw() + scale_fill_brewer(palette="Spectral")
  ggsave(plot = overlayedRocs, filename = paste(workPath,testprefix,"_validation_plotRocNoCI.png",sep =""), width = 8, height = 5, units = "in", dpi = 300)
  confMat <- confusionMatrix(data = as.factor(trained$predicted), reference = as.factor(trained$PHENO), positive = "DISEASE")
  print(confMat)
  handler$evalConfMat = confMat
  return(handler)
  
}



genPRSiceCommand = function(geno,
                            pheno,
                            covstr,
                            path2PRSice,
                            PRSiceexe,
                            cores,
                            fprefix,
                            path2Genotype,
                            path2GWAS,
                            gwas,
                            binaryTarget,
                            barLevels="5E-8,4E-8,3E-8,2E-8,1E-8,9E-7,8E-7,7E-7,6E-7,5E-7,4E-7,3E-7,2E-7,1E-7,9E-6,8E-6,7E-6,6E-6,5E-6,4E-6,3E-6,2E-6,1E-6,9E-5,8E-5,7E-5,6E-5,5E-5,4E-5,3E-5,2E-5,1E-5,9E-4,8E-4,7E-4,6E-4,5E-4,4E-4,3E-4,2E-4,1E-4,9E-3,8E-3,7E-3,6E-3,5E-3,4E-3,3E-3,2E-3,1E-3,9E-2,8E-2,7E-2,6E-2,5E-2,4E-2,3E-2,2E-2,1E-2,9E-1,8E-1,7E-1,6E-1,5E-1,4E-1,3E-1,2E-1,1E-1,1 ",
                            gwasDef=" --beta --snp SNP --A1 A1 --A2 A2 --stat b --se se --pvalue p"){
  
  return(paste0("Rscript ",path2PRSice,"PRSice.R --binary-target T --prsice ",path2PRSice,PRSiceexe,
                " -n ",
                cores, " --out ",fprefix,".temp --pheno-file ",path2Genotype,"/",pheno,".pheno -t ",
                path2Genotype,"/",geno," -b ",
                path2GWAS,"/",gwas,covstr,
                " --print-snp --score std --perm 10000 ",
                " --bar-levels ",barLevels,
                " --fastscore --binary-target ",binaryTarget,
                gwasDef))
  
}



clumpCommand = function(geno,gwas,thresh,fprefix,path2plink,path2Genotype,path2GWAS,SNPcolumnatGWAS,clumpField){
  return(paste0(path2plink,"plink --bfile ",path2Genotype,"/",geno," --extract ",
                fprefix,".temp.snpsToPull --clump ",path2GWAS,"/",gwas,
                " --clump-p1 ",thresh," --clump-p2 ",thresh,
                " --clump-snp-field ",SNPcolumnatGWAS," --clump-field ",clumpField," --clump-r2 0.1 --clump-kb 250 --out ",
                fprefix,".tempClumps"))
  
  
}

checkVariantNames = function(traindata,testdata){
  
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
      thesplit = stringr::str_split(snp,"_")
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















