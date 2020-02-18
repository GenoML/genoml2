# Importing the necessary packages 
import argparse
import sys
import sklearn
import h5py
import pandas as pd
import numpy as np
import time

# Import the necessary internal GenoML packages 
from genoml.continuous.supervised import tune

def cstune(run_prefix, max_iter, cv_count):
    # TUNING 
    # Create a dialogue with the user 
    print("Here is some basic info on the command you are about to run.")
    print("CLI argument info...")
    print(f"Working with the dataset and best model corresponding to prefix {run_prefix} the timestamp from the merge is the prefix in most cases.")
    print(f"Your maximum number of tuning iterations is {max_iter} and if you are concerned about runtime, make this number smaller.")
    print(f"You are running {cv_count} rounds of cross-validation, and again... if you are concerned about runtime, make this number smaller.")
    print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
    print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")

    print("")
    
    infile_h5 = run_prefix + ".dataForML.h5"
    df = pd.read_hdf(infile_h5, key = "dataForML")

    y_tune = df.PHENO
    X_tune = df.drop(columns=['PHENO'])
    IDs_tune = X_tune.ID
    X_tune = X_tune.drop(columns=['ID'])


    best_algo_name_in = run_prefix + '.best_algorithm.txt'
    best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
    best_algo = str(best_algo_df.iloc[0,0])
    

    # Communicate to the user the best identified algorithm 
    print(f"From previous analyses in the training phase, we've determined that the best algorithm for this application is {best_algo}... so let's tune it up and see what gains we can make!")

    # Tuning 
    ## This calls on the functions made in the tune class (tuning.py) at the genoml.continuous.supervised
    model_tune = tune(df, run_prefix, max_iter, cv_count)