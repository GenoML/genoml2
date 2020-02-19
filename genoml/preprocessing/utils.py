import pandas as pd
import sys

def print_config(args):
    # Print out the formatted arguments
    print("")
    print("Here is some basic info on the command you are about to run.")
    print("Python version info...")
    print(sys.version)
    print("CLI argument info...")
    print(f"The output prefix for this run is {args.prefix} and will be appended to later runs of GenoML.")
    print(f"Working with genotype data? {args.geno}")
    print(f"Working with additional predictors? {args.addit}")
    print(f"Where is your phenotype file? {args.pheno}")
    print(f"Any use for an external set of GWAS summary stats? {args.gwas}")
    print(f"If you plan on using external GWAS summary stats for SNP filtering, we'll only keep SNPs at what P value? {args.p}")
    print(f"How strong is your VIF filter? {args.vif}")
    print(f"How many iterations of VIF filtering are you doing? {args.iter}")
    print(f"The imputation method you picked is using the column {args.impute} to fill in any remaining NAs.")
    print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: os, sys, argparse, numpy, pandas, joblib, math and time. We also use PLINKv1.9 from https://www.cog-genomics.org/plink/1.9/.")
    print("")


def parse_args(args):

    # Path variable initializations
    pheno_path = args.pheno
    addit_path = args.addit
    gwas_path = args.gwas
    geno_path = args.geno

    # Dataframe initializations
    pheno_df = None
    addit_df = None
    gwas_df = None

    # Other argument initializations 
    impute_type = args.impute
    vif_threshold = args.vif 
    iteration = args.iter
    p_gwas = args.p

    # Dataframe loading based on arguments
    if (pheno_path == "lost"):
        print("Looks like you lost your phenotype file. Just give up because you are currently don't have anything to predict.")
    else:
        pheno_df = pd.read_csv(pheno_path, engine='c')

    if (addit_path == "nope"):
        print("No additional features as predictors? No problem, we'll stick to genotypes.")
    else:
        addit_df = pd.read_csv(addit_path, engine='c')

    if (gwas_path == "nope"):
        print("So you don't want to filter on P values from external GWAS? No worries, we don't usually either (if the dataset is large enough).")
    else:
        gwas_df = pd.read_csv(gwas_path, engine='c')

    if (geno_path == "nope"):
        print("So no genotypes? Okay, we'll just use additional features provided for the predictions.")
    else:
        print("Pruning your data and exporting a reduced set of genotypes.")

    # Return populated variables
    return pheno_path, addit_path, gwas_path, geno_path, pheno_df, addit_df, gwas_df, impute_type, vif_threshold, iteration, p_gwas
