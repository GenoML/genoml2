# Importing the necessary packages 
import argparse
import sys
import sklearn
import h5py
import pandas as pd
import numpy as np
import time

# Import the necessary internal GenoML packages 
from genoml.continuous.supervised import train

def cstrain(prefix, rank_features, export_predictions):
    # Print out simple info for users 
    print("")
    print("Here is some basic info on the command you are about to run.")
    print("Python version info...")
    print(sys.version)

    # Print out chosen CLI arguments     
    print("CLI argument info...")
    print(f"Are you ranking features, even though it is pretty slow? Right now, GenoML runs general recursive feature ranking. You chose to {rank_features} this part.")
    print(f"Working with dataset {prefix} from previous data munging efforts.")
    print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
    print("")

    # Specify prefix and dataframe variables to be passed into class
    run_prefix = prefix
    infile_h5 = run_prefix + ".dataForML.h5"
    df = pd.read_hdf(infile_h5, key = "dataForML")

    # Train the model and output a summary
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
    print("explained_variance_score, this is the variance explained by the model per algorithm (scale from 0 to 1 with 1 being completely explained).")
    print("mean_squared_error, this is the mean squared error from regression loss.")
    print("median_absolute_error, median absolute error from regression loss.")
    print("r2_score, standard r2 metric from linear regression (coefficient of determination), remember, this can be negative if your model is really bad.")
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
    
    model.export_predictions()
    
    # Save out the proper algorithm
    model.save_results(prefix, algorithmResults = True, bestAlgorithm = True, featureRankings = True)

    print("Thank you for training with GenoML!")