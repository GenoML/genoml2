# Import the necessary packages 
import argparse

# Import the necessary internal GenoML packages 
from genoml.cli import dstrain
from genoml.cli import dstune
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", choices=["discrete", "continuous"])
    parser.add_argument("method", choices=["supervised", "unsupervised"])
    parser.add_argument("mode", choices=["train", "tune"])

    #Global
    parser.add_argument("--prefix", type=str, default="GenoML_data", help="Prefix for your training data build.")

    #Mung
    parser.add_argument("--geno", type=str, default="nope", help="Genotype: (string file path). Path to PLINK format genotype file, everything before the *.bed/bim/fam [default: nope].")
    parser.add_argument("--addit", type=str, default="nope", help="Additional: (string file path). Path to CSV format feature file [default: nope].")
    parser.add_argument("--pheno", type=str, default="lost", help="Phenotype: (string file path). Path to CSV phenotype file [default: lost].")
    parser.add_argument("--gwas", type=str, default="nope", help="GWAS summary stats: (string file path). Path to CSV format external GWAS summary statistics containing at least the columns SNP and P in the header [default: nope].")
    parser.add_argument("--p", type=float, default=0.001, help="P threshold for GWAS: (some value between 0-1). P value to filter your SNP data on [default: 0.001].")
    parser.add_argument("--vif", type=int, default=0, help="Variance Inflation Factor (VIF): (integer). This is the VIF threshold for pruning non-genotype features. We recommend a value of 5-10. The default of 0 means no VIF filtering will be done. [default: 0].")
    parser.add_argument("--iter", type=int, default=0, help="Iterator: (integer). How many iterations of VIF pruning of features do you want to run. To save time VIF is run in randomly assorted chunks of 1000 features per iteration. The default of 1 means only one pass through the data. [default: 1].")
    parser.add_argument("--impute", type=str, default="median", help="Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].")

    # TRAINING
    parser.add_argument('--rank_features', type=str, default='skip', choices=['skip','run'], help='Export feature rankings: (skip, run). Exports feature rankings but can be quite slow with huge numbers of features [default: skip].')

    # TUNING
    parser.add_argument('--max_tune', type=int, default=50, help='Max number of tuning iterations: (integer likely greater than 10). This governs the length of tuning process, run speed and the maximum number of possible combinations of tuning parameters [default: 50].')
    parser.add_argument('--n_cv', type=int, default=5, help='Number of cross validations: (integer likely greater than 3). Here we set the number of cross-validation runs for the algorithms [default: 5].')

    args = parser.parse_args()

    # DICTIONARY OF CLI 
    clis = {
    "discretesupervisedtrain": dstrain(args.prefix, args.rank_features),
    "discretesupervisedtune": dstune(args.prefix, args.max_tune, args.n_cv)
    }

    clis[args.data + args.method + args.mode]


    
