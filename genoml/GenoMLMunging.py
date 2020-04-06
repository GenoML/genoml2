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
import genoml.dependencies
from genoml.preprocessing import munging, vif, featureselection


def main():
    genoml.dependencies.check_dependencies()

    # Create the arguments
    parser = argparse.ArgumentParser(
        description="Arguments for building a training dataset for GenoML.")
    parser.add_argument("--prefix", type=str, default="GenoML_data",
                        help="Prefix for your training data build.", required=True)
    parser.add_argument("--datatype", type=str, default="d", choices=['d', 'c'],
                        help="Type of data. Choices are d for discrete, c for continuous", required=True)
    parser.add_argument("--pheno", type=str, default="lost",
                        help="Phenotype: (string file path). Path to CSV phenotype file [default: lost].",
                        required=True)
    parser.add_argument("--geno", type=str, default=None,
                        help="Genotype: (string file path). Path to PLINK format genotype file, everything before the *.bed/bim/fam [default: None].")
    parser.add_argument("--addit", type=str, default=None,
                        help="Additional: (string file path). Path to CSV format feature file [default: None].")
    parser.add_argument("--gwas", type=str, default=None,
                        help="GWAS summary stats: (string file path). Path to CSV format external GWAS summary statistics containing at least the columns SNP and P in the header [default: nope].")
    parser.add_argument("--p", type=float, default=0.001,
                        help="P threshold for GWAS: (some value between 0-1). P value to filter your SNP data on [default: 0.001].")
    parser.add_argument("--vif", type=int, default=0,
                        help="Variance Inflation Factor (VIF): (integer). This is the VIF threshold for pruning non-genotype features. We recommend a value of 5-10. The default of 0 means no VIF filtering will be done. [default: 0].")
    parser.add_argument("--iter", type=int, default=0,
                        help="Iterator: (integer). How many iterations of VIF pruning of features do you want to run. To save time VIF is run in randomly assorted chunks of 1000 features per iteration. The default of 1 means only one pass through the data. [default: 1].")
    parser.add_argument("--impute", type=str, default="median",
                        help="Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].",
                        choices=["median", "mean"])
    parser.add_argument('--featureSelection', type=int, default=0,
                        help='Run a quick tree-based feature selection routine prior to anything else, here you input the integer number of estimators needed, we suggest >= 50. The default of 0 will skip this functionality. This will also output a reduced dataset for analyses in addition to feature ranks. [default: 0]')

    args = parser.parse_args()

    # Print configurations
    print("")
    print("Here is some basic info on the command you are about to run.")
    print("Python version info...")
    print(sys.version)
    print("CLI argument info...")
    print(
        f"The output prefix for this run is {args.prefix} and will be appended to later runs of GenoML.")
    print(f"Working with genotype data? {args.geno}")
    print(f"Working with additional predictors? {args.addit}")
    print(f"Where is your phenotype file? {args.pheno}")
    print(f"Any use for an external set of GWAS summary stats? {args.gwas}")
    print(
        f"If you plan on using external GWAS summary stats for SNP filtering, we'll only keep SNPs at what P value? {args.p}")
    print(f"How strong is your VIF filter? {args.vif}")
    print(f"How many iterations of VIF filtering are you doing? {args.iter}")
    print(
        f"The imputation method you picked is using the column {args.impute} to fill in any remaining NAs.")
    print(
        "Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: os, sys, argparse, numpy, pandas, joblib, math and time. We also use PLINK v1.9 from https://www.cog-genomics.org/plink/1.9/.")
    print("")

    # Process the arguments
    run_prefix = args.prefix
    dataType = args.datatype
    n_est = args.featureSelection

    # Run the munging script in genoml.preprocessing
    munger = munging(pheno_path=args.pheno, run_prefix=args.prefix, impute_type=args.impute,
                     p_gwas=args.p, addit_path=args.addit, gwas_path=args.gwas, geno_path=args.geno)

    # Process the PLINK inputs (for pruning)
    df = munger.plink_inputs()

    # Run the feature selection using extraTrees
    if (args.featureSelection > 0):
        featureSelection_df = featureselection(run_prefix, df, dataType, n_est)
        df = featureSelection_df.rank()
        featureSelection_df.export_data()

    # Run the VIF calculation
    if (args.iter > 0):
        vif_calc = vif(args.iter, args.vif, df, 100, run_prefix)
        vif_calc.vif_calculations()

    # Thank the user
    print("Thank you for munging with GenoML!")
