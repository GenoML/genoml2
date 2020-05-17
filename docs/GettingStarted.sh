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
conda create -n GenoML python=3.7

# Activating and changing directories to environment
conda activate GenoML
    # Deactivating a conda environment 
        # conda deactivate 
    # Removing a conda environment 
        # conda env remove -n ENV_NAME

# Installing from a requirements file using pip
pip install -r requirements.txt
    # If issues installing xgboost from requirements - use Homebrew to 
        # xcode-select --install
        # brew install gcc@7
        # conda install -c conda-forge xgboost [or] pip install xgboost==0.90

# Install the package at this path
pip install .

# Saving out environment requirements to a .txt file
#pip freeze > requirements.txt

# Removing a conda virtualenv
#conda remove --name GenoML --all

# Run GenoML 
genoml
#usage: GenoML [-h] [--prefix PREFIX] [--geno GENO] [--addit ADDIT]
            #  [--pheno PHENO] [--gwas GWAS] [--p P] [--vif VIF] [--iter ITER]
            #  [--impute IMPUTE] [,run}]
            #  [--max_tune MAX_TUNE] [--n_cv N_CV]
            #  {discrete,continuous} {supervised,unsupervised} {train,tune}
#GenoML: error: the following arguments are required: data, method, mode

##### TESTED 

## MUNGING 
# Running the munging script [discrete]
genoml-munging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv 

# Running the munging script with VIF and iterations [discrete]
genoml-munging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--vif 5 \
--iter 1

# Running the munging script with GWAS [discrete]
genoml-munging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv 

# Running the munging script with VIF and GWAS [discrete]
genoml-munging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--vif 5 --iter 1

# Running the munging script with addit, VIF, and GWAS [discrete]
genoml-munging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--addit examples/discrete/training_addit.csv \
--vif 5 --iter 1

# Running the munging script with featureSelection [discrete]
genoml-munging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--featureSelection 50

# Running the munging script with everything [discrete]
genoml-munging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--addit examples/discrete/training_addit.csv \
--p 0.01 \
--vif 5 --iter 1

# Running the munging script [continuous]
genoml-munging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv 

# Running the munging script with VIF [continuous]
genoml-munging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--vif 5 \
--iter 2

# Running the munging script with GWAS [continuous]
genoml-munging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--gwas examples/continuous/example_GWAS.csv 

# Running the munging script with VIF and GWAS [continuous]
genoml-munging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--gwas examples/continuous/example_GWAS.csv \
--vif 5 --iter 1

# Running the munging script with addit, VIF, and GWAS [continuous]
genoml-munging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--gwas examples/continuous/example_GWAS.csv \
--addit examples/continuous/training_addit.csv \
--vif 5 --iter 1

# Running the munging script with everything [continuous]
genoml-munging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--gwas examples/continuous/example_GWAS.csv \
--addit examples/continuous/training_addit.csv \
--p 0.01 \
--vif 5 --iter 1

## TRAIN 
# Running the supervised training script [discrete]
genoml discrete supervised train \
--prefix outputs/test_discrete_geno \
--metric_max Sensitivity

# Running the supervised training script [continuous]
genoml continuous supervised train \
--prefix outputs/test_continuous_geno \

## TUNE 
# Running the supervised tuning script [discrete]
genoml discrete supervised tune \
--prefix outputs/test_discrete_geno \
--max_tune 10 --n_cv 3

# Running the supervised tuning script [continuous]
genoml continuous supervised tune \
--prefix outputs/test_continuous_geno \
--max_tune 10 --n_cv 3


## TEST DISCRETE
# MUNGE THE REFERENCE
genoml-munging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv

# TRAIN THE REFERENCE
genoml discrete supervised train \
--prefix outputs/test_discrete_geno
  
# HARMONIZE TEST DATASET
genoml-harmonizing --test_geno_prefix examples/discrete/validation \
--test_prefix outputs/validation_test_discrete_geno \
--refModel_prefix outputs/test_discrete_geno \
--training_SNPsAlleles outputs/test_discrete_geno.variants_and_alleles.tab
  
# MUNGE THE TEST DATASET ON INTERSECTING COLUMNS
genoml-munging --prefix outputs/validation_test_discrete_geno \
--datatype d \
--geno outputs/validation_test_discrete_geno_refSNPs_andAlleles \
--pheno examples/discrete/validation_pheno.csv \
--addit examples/discrete/validation_addit.csv \
--refColsHarmonize outputs/validation_test_discrete_geno_refColsHarmonize_toKeep.txt
  
# RETRAIN REFERENCE ON INTERSECTING COLUMNS
genoml discrete supervised train \
--prefix outputs/test_discrete_geno \
--matchingCols outputs/validation_test_discrete_geno_finalHarmonizedCols_toKeep.txt

# TEST RETRAINED REFERENCE MODEL ON UNSEEN DATA
genoml discrete supervised test \
--prefix outputs/validation_test_discrete_geno \
--test_prefix outputs/validation_test_discrete_geno \
--refModel_prefix outputs/test_discrete_geno.trainedModel


## TEST continuous
# MUNGE THE REFERENCE
genoml-munging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv

# TRAIN THE REFERENCE
genoml continuous supervised train \
--prefix outputs/test_continuous_geno
  
# HARMONIZE TEST DATASET
genoml-harmonizing --test_geno_prefix examples/continuous/validation \
--test_prefix outputs/validation_test_continuous_geno \
--refModel_prefix outputs/test_continuous_geno \
--training_SNPsAlleles outputs/test_continuous_geno.variants_and_alleles.tab
  
# MUNGE THE TEST DATASET ON INTERSECTING COLUMNS
genoml-munging --prefix outputs/validation_test_continuous_geno \
--datatype d \
--geno outputs/validation_test_continuous_geno_refSNPs_andAlleles \
--pheno examples/continuous/validation_pheno.csv \
--addit examples/continuous/validation_addit.csv \
--refColsHarmonize outputs/validation_test_continuous_geno_refColsHarmonize_toKeep.txt
  
# RETRAIN REFERENCE ON INTERSECTING COLUMNS
genoml continuous supervised train \
--prefix outputs/test_continuous_geno \
--matchingCols outputs/validation_test_continuous_geno_finalHarmonizedCols_toKeep.txt

# TEST RETRAINED REFERENCE MODEL ON UNSEEN DATA
genoml continuous supervised test \
--prefix outputs/validation_test_continuous_geno \
--test_prefix outputs/validation_test_continuous_geno \
--refModel_prefix outputs/test_continuous_geno.trainedModel
