# Importing the necessary packages 
import argparse
import sys
import sklearn
import h5py
import pandas as pd
import numpy as np
import time

# Import the necessary internal GenoML packages 
from genoml.discrete.supervised import train

def dstrain(prefix, rank_features, prob_hist, auc):
    print("")
    print("Here is some basic info on the command you are about to run.")
    print("Python Version info...")
    print(sys.version) 

    # Print out chosen CLI arguments 
    print("CLI argument info...")
    print(f"Are you ranking features, even though it is pretty slow? Right now, GenoML runs general recursive feature ranking. You chose to {rank_features} this part.")
    print(f"Working with dataset {prefix} from previous data munging efforts.")
    print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to Python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
    print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")
    print("")

    # Specify prefix and dataframe variables to be passed into class
    run_prefix = prefix
    infile_h5 = run_prefix + ".dataForML.h5"
    df = pd.read_hdf(infile_h5, key = "dataForML")

    model = train(df, run_prefix)
    model.summary()
    
    # Give user context prior to competing algorithms 
    # Explains to users how we are splitting their data 70:30 
    print("")
    print("Now let's compete these algorithms!")
    print("We'll update you as each algorithm runs, then summarize at the end.")
    print("Here we test each algorithm under default settings using the same training and test datasets derived from a 70% training and 30% testing split of your data.")
    print("For each algorithm, we will output the following metrics...")
    print("Algorithm name, hoping that's pretty self-explanatory. Plenty of resources on these common ML algorithms at https://scikit-learn.org and https://xgboost.readthedocs.io/.")
    print("AUC_percent, this is the area under the curve from receiver operating characteristic analyses. This is the most common metric of classifier performance in biomedical literature, we express this as a percent. We calculate AUC based on the predicted probability of being a case.")
    print("Accuracy_percent, this is the simple accuracy of the classifier, how many predictions were correct from best classification cutoff (python default).")
    print("Balanced_Accuracy_Percent, consider this as the accuracy resampled to a 1:1 mix of cases and controls. Imbalanced datasets can give funny results for simple accuracy.")
    print("Log_Loss, this is essentially the inverse of the likelihood function for a correct prediction, you want to minimize this.")
    print("Sensitivity, proportion of cases correctly identified.")
    print("Specificity, proportion of controls correctly identified.")
    print("PPV, this is the positive predictive value, the probability that subjects with a positive result actually have the disease.")
    print("NPV, this is the negative predictive value, the probability that subjects with a negative result don't have the disease.")
    print("We also log the runtimes per algorithm.")
    print("")
    print("Algorithm summaries incoming...")
    print("")

    # Compete the algorithms 
    model.compete()
    
    # Output the results of the log
    model.results()

    # Export the results 
    model.export_model()

    if(rank_features == "run"):
        model.feature_ranking()

    # Export the AUC     
    model.AUC(save=True)

    # Export the probability histograms
    model.export_prob_hist()

    # Save out the proper algorithm
    model.save_results(prefix, algorithmResults = True, bestAlgorithm = True, featureRankings = True)

    print("Thank you for training with GenoML!")