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


# This shows how to structure inputs for GenoML CLI 

## SETTING UP A VIRTUAL ENVIRONMENT 
# Making a virtual environment
conda create -n GenoML python=3.12

# Activating and changing directories to environment
# conda activate GenoML
    # Deactivating a conda environment 
        # conda deactivate 
    # Removing a conda environment 
        # conda env remove -n ENV_NAME

# Installing from a requirements file using pip
pip install -r requirements.txt
    # If issues installing xgboost from requirements - use Homebrew to 
        # xcode-select --install
        # brew install gcc@7
        # conda install -c conda-forge xgboost [or] pip install xgboost==2.0.3

# Install the package at this path
pip install .

# Saving out environment requirements to a .txt file
#pip freeze > requirements.txt

# Removing a conda virtualenv
#conda remove --name GenoML --all

## 1. MUNGE

# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv

# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file with a detailed log printed to the console
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--verbose

# Running GenoML munging on discrete data using PLINK genotype binary files and phenotype files for both the training and testing datasets.
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--geno_test examples/discrete/validation \
--pheno_test examples/discrete/validation_pheno.csv

# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--r2_cutoff 0.3 \
--pheno examples/discrete/training_pheno.csv

# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--skip_prune \
--pheno examples/discrete/training_pheno.csv

# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file and specifying impute
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--impute mean

# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file while using VIF to remove multicollinearity 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--vif 5 \
--vif_iter 1

# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and a GWAS summary statistics file 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv

# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and two GWAS summary statistics files
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--gwas examples/discrete/example_GWAS_2.csv

# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and a GWAS summary statistics file with a p-value cut-off 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--p 0.01

# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and an addit file
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv

# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and an addit file
genoml discrete supervised munge \
--prefix outputs \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv

# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and running feature selection 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv \
--feature_selection 50

# Running GenoML munging on discreate data using PLINK binary files, a phenotype file, using UMAP to reduce dimensions and account for features, and running feature selection
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv \
--umap_reduce \
--adjust_data \
--adjust_normalize \
--target_features examples/discrete/to_adjust.txt \
--confounders examples/discrete/training_addit_confounder_example.csv 

# Running GenoML munging on discreate data using PLINK binary files, a phenotype file, using UMAP to reduce dimensions and account for features, and running feature selection, for both the training and testing data together.
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv \
--geno_test examples/discrete/validation \
--pheno_test examples/discrete/validation_pheno.csv \
--addit_test examples/discrete/validation_addit.csv \
--umap_reduce \
--adjust_data \
--adjust_normalize \
--target_features examples/discrete/to_adjust.txt \
--confounders examples/discrete/training_addit_confounder_example.csv \
--confounders_test examples/discrete/validation_addit_confounder_example.csv 



## 1b. HARMONIZE

# Running GenoML harmonization on discrete data using PLINK genotype binary files and a phenotype file 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/validation \
--pheno examples/discrete/validation_pheno.csv

# Running GenoML harmonization on discrete data using PLINK genotype binary files and a phenotype file 
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/validation \
--pheno examples/discrete/validation_pheno.csv \
--confounders examples/discrete/validation_addit_confounder_example.csv

# Running GenoML harmonization on discrete data using PLINK genotype binary files and a phenotype file, while imputing any missing columns (ie, if an addit file was used during training and is not present for the harmonization participants).
genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/validation \
--pheno examples/discrete/validation_pheno.csv \
--force_impute



## 2. TRAIN

# Running GenoML supervised training after munging on discrete data
genoml discrete supervised train \
--prefix outputs

# Running GenoML supervised training after munging on discrete data and specifying Sensitivity as the metric to optimize
genoml discrete supervised train \
--prefix outputs \
--metric_max Sensitivity




## 3. TUNE

# Running GenoML supervised tuning after munging and training on discrete data
genoml discrete supervised tune \
--prefix outputs

# Running GenoML supervised tuning after munging and training on discrete data, modifying the number of iterations and cross-validations 
genoml discrete supervised tune \
--prefix outputs \
--max_tune 10 \
--n_cv 3

# Running GenoML supervised tuning after munging and training on discrete data, modifying the metric to tune by
genoml discrete supervised tune \
--prefix outputs \
--metric_tune Balanced_Accuracy




## 4. TEST

# Running GenoML test
genoml discrete supervised test \
--prefix outputs




## 5. FULL PIPELINE EXAMPLE

# MUNGE THE REFERENCE DATASET
genoml discrete supervised munge \
--prefix outputs \
--pheno examples/discrete/training_pheno.csv \
--geno examples/discrete/training \
--addit examples/discrete/training_addit.csv \
--pheno_test examples/discrete/validation_pheno.csv \
--geno_test examples/discrete/validation \
--addit_test examples/discrete/validation_addit.csv \
--r2_cutoff 0.3 \
--impute mean \
--vif 10 \
--vif_iter 1 \
--gwas examples/discrete/example_GWAS.csv \
--gwas examples/discrete/example_GWAS_2.csv \
--p 0.05 \
--feature_selection 50 \
--adjust_data \
--adjust_normalize \
--umap_reduce \
--confounders examples/discrete/training_addit_confounder_example.csv \
--confounders_test examples/discrete/validation_addit_confounder_example.csv \
--target_features examples/discrete/to_adjust.txt \
--verbose

# TRAIN THE REFERENCE MODEL
genoml discrete supervised train \
--prefix outputs \
--metric_max Balanced_Accuracy

# OPTIONAL: TUNING YOUR REFERENCE MODEL
genoml discrete supervised tune \
--prefix outputs \
--max_tune 10 \
--n_cv 3 \
--metric_tune Balanced_Accuracy

# TEST TUNED MODEL ON UNSEEN DATA
genoml discrete supervised test \
--prefix outputs
