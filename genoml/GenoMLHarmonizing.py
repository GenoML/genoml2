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

import argparse
import sys

import genoml.dependencies
from genoml import preprocessing

def main():
    genoml.dependencies.check_dependencies()

    # Create the arguments 
    parser = argparse.ArgumentParser(description='Arguments for prepping a test dataset')    
    
    parser.add_argument('--testGenoPrefix', type=str, default='genotype_binaries', help='Prefix of the genotypes for the test dataset in PLINK binary format.', required=True)
    parser.add_argument('--testOutPrefix', type=str, default='GenoML_model', help='Prefix of the output that will be generated.', required=True)
    parser.add_argument('--refDatasetPrefix', type=str, default=None, help='Prefix for the training dataset we will use to compare, you can leave off the \'.joblib\' suffix.', required=True)
    parser.add_argument('--trainingSNPsAlleles', type=str, default=None, help='File to the SNPs and alleles file generated in the training phase that we will use to compare.', required=True)
    print("")

    # Process the arguments 
    args = parser.parse_args()
    
    test_geno_prefix = args.testGenoPrefix
    test_out_prefix = args.testOutPrefix

    ref_model_prefix = args.refDatasetPrefix
    training_SNPs = args.trainingSNPsAlleles

    # Print configurations
    print("Here is some basic info on the command you are about to run.")
    print("Python version info...")
    print(sys.version)
    print("CLI argument info...")
    print(f"You are importing test dataset {test_geno_prefix}.")
    print(f"Applying the model saved from your reference dataset in {ref_model_prefix}.")
    print(f"Reading in the SNP and allele information we will use to compare from {training_SNPs}.")
    print(f"The results of this test application of your model will be saved in files tagged {test_out_prefix}.")
    print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")

    # Run the harmonize script in genoml.preprocessing 
    harmonizer = preprocessing.harmonizing(test_geno_prefix=test_geno_prefix, test_out_prefix=test_out_prefix, ref_model_prefix=ref_model_prefix, training_SNPs=training_SNPs)

    # Generate new binaries from the test dataset using the reference dataset SNPs 
    harmonizer.generate_new_PLINK()

    # Read in PLINK binaries
    #harmonizer.read_PLINK()

    # Generate reference columns to keep for munging 
    harmonizer.prep_refCols_file() 

    # Thank the user
    print("Thank you for harmonizing with GenoML!")
