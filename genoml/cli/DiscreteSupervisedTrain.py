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

def dstrain(prefix, rank_features):
    print("")
    print("Here is some basic info on the command you are about to run.")
    print("Python Version info...")
    print(sys.version) 

    print("CLI argument info...")
    print(f"Are you ranking features, even though it is pretty slow? Right now, GenoML runs general recursive feature ranking. You chose to {rank_features} this part.")
    print(f"Working with dataset {prefix} from previous data munging efforts.")
    print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to Python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
    print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")
    print("")

    run_prefix = prefix
    infile_h5 = run_prefix + ".dataForML.h5"
    df = pd.read_hdf(infile_h5, key = "dataForML")

    model = train(df)
    model.summary()
    
    # 
    log = model.compete()
    
    # Output the results of the log
    best = model.results()

    # Save out the proper algorithm
    model.save_results("./", algorithmResults = True)