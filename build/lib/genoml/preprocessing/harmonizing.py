# Copyright 2020 The GenoML Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Import the necessary packages
import subprocess
import numpy as np
import sys
import joblib
import pandas as pd
from pandas_plink import read_plink1_bin

# Define the munging class
import genoml.dependencies

class harmonizing:
    def __init__(self, test_geno_prefix, test_out_prefix, ref_model_prefix, training_SNPs):
        
        # Initializing the variables we will use 
        self.test_geno_prefix = test_geno_prefix
        self.test_out_prefix = test_out_prefix
        self.ref_model_prefix = ref_model_prefix
        self.training_SNPs = training_SNPs

        infile_h5 = ref_model_prefix + ".dataForML.h5"
        self.df = pd.read_hdf(infile_h5, key = "dataForML")

    def generate_new_PLINK(self):        
        # Show first few lines of the dataframe
        print("")
        print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
        print("#"*70)
        print(self.df.describe())
        print("#"*70)
        print("")

        # Save out and drop the PHENO and sample ID columns 
        y_test = self.df.PHENO
        X_test = self.df.drop(columns=['PHENO'])
        IDs_test = X_test.ID
        X_test = X_test.drop(columns=['ID'])

        # Save variables to use globally within the class 
        self.y_test = y_test
        self.X_test = X_test
        self.IDs_test = IDs_test

        # Read in the column of SNPs from the SNP+Allele file read in 
        snps_alleles_df = pd.read_csv(self.training_SNPs, header=None)
        snps_only = snps_alleles_df.iloc[:, 0]
        snps_temp = self.test_out_prefix + '.SNPS_only_toKeep_temp.txt'
        snps_only.to_csv(snps_temp, header=None, index=False)

        print(f"A temporary file of SNPs from your reference dataset to keep in your testing dataset has been exported here: {snps_temp}")

        # Prepare the bashes to keep the SNPs of interest from the reference dataset
        plink_exec = genoml.dependencies.check_plink()

        # Creating outfile with SNPs
        # Force the allele designations based on the reference dataset
        plink_outfile = self.test_out_prefix + ".refSNPs_andAlleles"
        
        print("")
        print(f"Now we will create PLINK binaries where the reference SNPS and alleles will be based off of your file here: {self.training_SNPs}")
        print("")

        bash1 = f"{plink_exec} --bfile " + self.test_geno_prefix + " --extract " + snps_temp + " --reference-allele " + self.training_SNPs + " --make-bed --out " + plink_outfile
        # Remove the .log file 
        bash2 = "rm " + plink_outfile + ".log"
        # Remove the .SNPS_only_toKeep_temp.txt file 
        bash3 = "rm " + snps_temp

        cmds_a = [bash1, bash2, bash3]

        for cmd in cmds_a:
            subprocess.run(cmd, shell=True)

        self.plink_outfile = plink_outfile

        print("")
        print(f"A new set of PLINK binaries generated from your test dataset with the SNPs you decided to keep from the reference dataset have been made here: {plink_outfile}")
        print("")

    # def read_PLINK(self):
    # # Read in using pandas PLINK (similar to munging)

    #     bed_file = self.plink_outfile + ".bed"
    #     plink_files_py = read_plink1_bin(bed_file)
    #     plink_files = plink_files_py.drop(['fid','father','mother','gender', 'trait', 'chrom', 'cm', 'pos','a1'])

    #     plink_files = plink_files.set_index({'sample':'iid','variant':'snp'})
    #     plink_files.values = plink_files.values.astype('int')

    #     # swap pandas-plink genotype coding to match .raw format...more about that below:

    #     # for example, assuming C in minor allele, alleles are coded in plink .raw labels homozygous for minor allele as 2 and homozygous for major allele as 0:
    #     #A A   ->    0   
    #     #A C   ->    1   
    #     #C C   ->    2
    #     #0 0   ->   NA

    #     # where as, read_plink1_bin flips these, with homozygous minor allele = 0 and homozygous major allele = 2
    #     #A A   ->    2   
    #     #A C   ->    1   
    #     #C C   ->    0
    #     #0 0   ->   NA

    #     two_idx = (plink_files.values == 2)
    #     zero_idx = (plink_files.values == 0)

    #     plink_files.values[two_idx] = 0
    #     plink_files.values[zero_idx] = 2

    #     plink_pd = plink_files.to_pandas()
    #     plink_pd.reset_index(inplace=True)
    #     raw_df = plink_pd.rename(columns={'sample': 'ID'})

    #     print("")
    #     print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
    #     print("#"*70)
    #     print(raw_df.describe())
    #     print("#"*70)
    #     print("")

    #     self.raw_df = raw_df

    #     return raw_df

    def prep_refCols_file(self):
        # Make a list of the column names from the reference dataset
        ref_columns_list = self.df.columns.values.tolist()

        # Write out the columns to a text file we will use in munge later 
        ref_cols_outfile = self.test_out_prefix + ".refColsHarmonize_toKeep.txt"

        with open(ref_cols_outfile, 'w') as filehandle:
            for col in ref_columns_list:
                filehandle.write('%s\n' % col)

        print("")
        print(f"A file with the columns in the reference file, to later use in the munging step and keep these same columns for the test dataset, has been generated here: {ref_cols_outfile}")
        print("")

        return ref_columns_list 


