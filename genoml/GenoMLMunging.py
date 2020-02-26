# Import the necessary packages
import os
import sys
import argparse
import math
import time
import h5py
import joblib
import subprocess
import numpy as np
import pandas as pd
from sys import platform

# Importing additional packages necessary for VIF
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from joblib import Parallel, delayed

# Import the necessary internal GenoML packages 
from genoml.preprocessing import utils, munging, vif

# Check the platform to load the right PLINK to path
    # This will load a PLINK v1.9 
def get_platform():
    platforms = {
        "linux" : "linux",
        "linux1" : "linux",
        "linux2" : "linux",
        "darwin" : "mac"
    }
    if sys.platform not in platforms:
        return "GenoML is not supported on this platform. Please try Mac or Linux"
    elif platforms[sys.platform] == "mac":
        filename_mac = "plink/mac/"
        directory_mac = os.getcwd() + "/" + filename_mac
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + directory_mac
    else:
        filename_linux = "plink/linux/"
        directory_linux = os.getcwd() + "/" + filename_linux
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + directory_linux
    return print("PLINK has successfully been loaded to your path!")


def main():
    # Run and get the proper PLINK to path
    get_platform()

    # Create the arguments
    parser = argparse.ArgumentParser(
        description="Arguments for building a training dataset for GenoML.")
    parser.add_argument("--prefix", type=str, default="GenoML_data",
                        help="Prefix for your training data build.")
    parser.add_argument("--geno", type=str, default="nope",
                        help="Genotype: (string file path). Path to PLINK format genotype file, everything before the *.bed/bim/fam [default: nope].")
    parser.add_argument("--addit", type=str, default="nope",
                        help="Additional: (string file path). Path to CSV format feature file [default: nope].")
    parser.add_argument("--pheno", type=str, default="lost",
                        help="Phenotype: (string file path). Path to CSV phenotype file [default: lost].")
    parser.add_argument("--gwas", type=str, default="nope",
                        help="GWAS summary stats: (string file path). Path to CSV format external GWAS summary statistics containing at least the columns SNP and P in the header [default: nope].")
    parser.add_argument("--p", type=float, default=0.001,
                        help="P threshold for GWAS: (some value between 0-1). P value to filter your SNP data on [default: 0.001].")
    parser.add_argument("--vif", type=int, default=0,
                        help="Variance Inflation Factor (VIF): (integer). This is the VIF threshold for pruning non-genotype features. We recommend a value of 5-10. The default of 0 means no VIF filtering will be done. [default: 0].")
    parser.add_argument("--iter", type=int, default=0,
                        help="Iterator: (integer). How many iterations of VIF pruning of features do you want to run. To save time VIF is run in randomly assorted chunks of 1000 features per iteration. The default of 1 means only one pass through the data. [default: 1].")
    parser.add_argument("--impute", type=str, default="median",
                        help="Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].")

    # Process the arguments
    args = parser.parse_args()
    run_prefix = args.prefix
    utils.print_config(args)
    pheno_path, addit_path, gwas_path, geno_path, pheno_df, addit_df, gwas_df, impute_type, vif_threshold, iteration, p_gwas = utils.parse_args(args)
    # Run the munging script in genoml.preprocessing 
    munger = munging(pheno_df, addit_df, gwas_df, impute_type, pheno_path, addit_path, gwas_path, geno_path, p_gwas, run_prefix, args)

    # Process the PLINK inputs (for pruning)
    df = munger.plink_inputs()

    # If the imputation flags are set, run to impute based on user input 

    # Run the VIF calculation 
    if(args.iter > 0):
        vif_calc = vif(args.iter, args.vif, df, 100, run_prefix)
        vif_calc.vif_calculations()
    
    # Thank the user
    print("Thank you for munging with GenoML!")