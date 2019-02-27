list.of.packages <- c(    "caret", "lattice", "ggplot2", "rBayesianOptimization", "plotROC", "doParallel", "randomForest", "xgboost", "e1071")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "https://cloud.r-project.org/")
