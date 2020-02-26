# Import the necessary packages
import subprocess
import pandas as pd

# Import the necessary internal GenoML packages
from genoml.preprocessing import utils

# Define the munging class

class munging:
    def __init__(self, pheno_df, addit_df, gwas_df, impute_type, pheno_path, addit_path, gwas_path, geno_path, p_gwas, run_prefix, args):
        self.pheno_path = pheno_path
        self.addit_path = addit_path
        self.gwas_path = gwas_path
        self.geno_path = geno_path
        self.pheno_df = pheno_df
        self.addit_df = addit_df
        self.gwas_df = gwas_df
        self.run_prefix = run_prefix
        self.impute_type = impute_type
        self.p_gwas = p_gwas
        self.args = args

    def plink_inputs(self):
        """
        Mode -- munging
        > Function -- plink_inputs

        This function takes in the arguments passed by the user. Arguments that are allowed are 
            - run_prefix : Prefix for your training data build             
            - pheno_path : Phenotype: (string file path). Path to CSV phenotype file
            - addit_path : 
            - gwas_path :
            - geno_path : 
            - pheno_df :
            - addit_df :
            - gwas_df :
            - impute_type :
            - p_gwas :

        Returns:
            [print (str)] -- Returns a print statement whether PLINK has been successfully loaded to path or not.
        """
        # Initializing some variables
        impute_type = self.impute_type
        addit_df = self.addit_df
        pheno_df = self.pheno_df

        outfile_h5 = self.run_prefix + ".dataForML.h5"
        pheno_df.to_hdf(outfile_h5, key='pheno', mode='w')

        # Set the bashes
        bash1a = "plink --bfile " + self.geno_path + " --indep-pairwise 1000 50 0.05"
        bash1b = "plink --bfile " + self.geno_path + " --extract " + self.run_prefix + \
            ".p_threshold_variants.tab" + " --indep-pairwise 1000 50 0.05"
        bash2 = "plink --bfile " + self.geno_path + \
            " --extract plink.prune.in --make-bed --out temp_genos"
        bash3 = "plink --bfile temp_genos --recode A --out " + self.run_prefix
        bash4 = "cut -f 2,5 temp_genos.bim > " + \
            self.run_prefix + ".variants_and_alleles.tab"
        bash5 = "rm temp_genos.*"
        bash6 = "rm " + self.run_prefix + ".raw"
        bash7 = "rm plink.log"
        bash8 = "rm plink.prune.*"
        bash9 = "rm " + self.run_prefix + ".log"

        # Set the bash command groups
        cmds_a = [bash1a, bash2, bash3, bash4, bash5, bash7, bash8, bash9]
        cmds_b = [bash1b, bash2, bash3, bash4, bash5, bash7, bash8, bash9]

        if (self.gwas_path != "nope") & (self.geno_path != "nope"):
            p_thresh = self.p_gwas
            gwas_df_reduced = self.gwas_df[['SNP', 'p']]
            snps_to_keep = gwas_df_reduced.loc[(
                gwas_df_reduced['p'] <= p_thresh)]
            outfile = self.run_prefix + ".p_threshold_variants.tab"
            snps_to_keep.to_csv(outfile, index=False, sep="\t")
            print(
                f"Your candidate variant list prior to pruning is right here: {outfile}.")

        if (self.gwas_path == "nope") & (self.geno_path != "nope"):
            print(
                f"A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here: {self.run_prefix}.variants_and_alleles.tab")
            for cmd in cmds_a:
                subprocess.run(cmd, shell=True)

        if (self.gwas_path != "nope") & (self.geno_path != "nope"):
            print(
                f"A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here: {self.run_prefix}.variants_and_alleles.tab")
            for cmd in cmds_b:
                subprocess.run(cmd, shell=True)

        if (self.geno_path != "nope"):
            raw_path = self.run_prefix + ".raw"
            raw_df = pd.read_csv(raw_path, engine='c', sep=" ")
            raw_df.drop(columns=['FID', 'MAT', 'PAT',
                                 'SEX', 'PHENOTYPE'], inplace=True)
            raw_df.rename(columns={'IID': 'ID'}, inplace=True)
            subprocess.run(bash6, shell=True)

    # Checking the impute flag and execute
        # Currently only supports mean and median

        impute_list = ["mean", "median"]

        if impute_type not in impute_list:
            return "The 2 types of imputation currently supported are 'mean' and 'median'"
        elif impute_type.lower() == "mean":
            raw_df = raw_df.fillna(raw_df.mean())
        elif impute_type.lower() == "median":
            raw_df = raw_df.fillna(raw_df.median())
        print("")
        print(
            f"You have just imputed your genotype features, covering up NAs with the column {impute_type} so that analyses don't crash due to missing data.")
        print("Now your genotype features might look a little better (showing the first few lines of the left-most and right-most columns)...")
        print("#"*70)
        print(raw_df.describe())
        print("#"*70)
        print("")

    # Checking the imputation of non-genotype features

        if (self.addit_path != "nope"):
            if impute_type not in impute_list:
                return "The 2 types of imputation currently supported are 'mean' and 'median'"
            elif impute_type.lower() == "mean":
                addit_df = addit_df.fillna(addit_df.mean())
            elif impute_type.lower() == "median":
                addit_df = addit_df.fillna(addit_df.median())
            print("")
            print(
                f"You have just imputed your non-genotype features, covering up NAs with the column {impute_type} so that analyses don't crash due to missing data.")
            print("Now your non-genotype features might look a little better (showing the first few lines of the left-most and right-most columns)...")
            print("#"*70)
            print(addit_df.describe())
            print("#"*70)
            print("")

            # Remove the ID column
            cols = list(addit_df.columns)
            cols.remove('ID')
            addit_df[cols]

            # Z-scale the features
            for col in cols:
                if (addit_df[col].min() != 0) & (addit_df[col].max() != 1):
                    addit_df[col] = (
                        addit_df[col] - addit_df[col].mean())/addit_df[col].std(ddof=0)

            print("")
            print("You have just Z-scaled your non-genotype features, putting everything on a numeric scale similar to genotypes.")
            print("Now your non-genotype features might look a little closer to zero (showing the first few lines of the left-most and right-most columns)...")
            print("#"*70)
            print(addit_df.describe())
            print("#"*70)

    # Saving out the proper HDF5 file
        if (self.geno_path != "nope"):
            merged = raw_df.to_hdf(outfile_h5, key='geno')

        if (self.addit_path != "nope"):
            merged = addit_df.to_hdf(outfile_h5, key='addit')

        if (self.geno_path != "nope") & (self.addit_path != "nope"):
            pheno = pd.read_hdf(outfile_h5, key="pheno")
            geno = pd.read_hdf(outfile_h5, key="geno")
            addit = pd.read_hdf(outfile_h5, key="addit")
            temp = pd.merge(pheno, addit, on='ID', how='inner')
            merged = pd.merge(temp, geno, on='ID', how='inner')

        if (self.geno_path != "nope") & (self.addit_path == "nope"):
            pheno = pd.read_hdf(outfile_h5, key="pheno")
            geno = pd.read_hdf(outfile_h5, key="geno")
            merged = pd.merge(pheno, geno, on='ID', how='inner')

        if (self.geno_path == "nope") & (self.addit_path != "nope"):
            pheno = pd.read_hdf(outfile_h5, key="pheno")
            addit = pd.read_hdf(outfile_h5, key="addit")
            merged = pd.merge(pheno, addit, on='ID', how='inner')

        self.merged = merged
        merged.to_hdf(outfile_h5, key='dataForML')

        return merged
