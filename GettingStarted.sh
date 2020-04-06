# This shows how to structure inputs for GenoML CLI 

## SETTING UP A VIRTUAL ENVIRONMENT 
# Making a virtual environment
conda create -n GenoML python=3.7

# Activating and changing directories to environment
conda activate GenoML
    # Deactivating a conda environment 
        # conda deactivate ENV_NAME 
    # Removing a conda environment 
        # conda env remove -n ENV_NAME

# Installing from a requirements file using pip
pip install -r requirements.txt
    # If issues installing xgboost from requirements - use Homebrew to 
        # xcode-select --install
        # brew install gcc@7
        # conda install -c conda-forge xgboost

# Install the package at this path
pip install .

# Saving out environment requirements to a .txt file
#pip freeze > requirements.txt

# Removing a conda virtualenv
#conda remove --name GenoML --all

# Run GenoML 
GenoML
#usage: GenoML [-h] [--prefix PREFIX] [--geno GENO] [--addit ADDIT]
            #  [--pheno PHENO] [--gwas GWAS] [--p P] [--vif VIF] [--iter ITER]
            #  [--impute IMPUTE] [--rank_features {skip,run}]
            #  [--max_tune MAX_TUNE] [--n_cv N_CV]
            #  {discrete,continuous} {supervised,unsupervised} {train,tune}
#GenoML: error: the following arguments are required: data, method, mode

##### TESTED 

## MUNGING 
# Running the munging script [discrete]
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv 

# Running the munging script with VIF and iterations [discrete]
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--vif 5 \
--iter 1

# Running the munging script with GWAS [discrete]
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv 

# Running the munging script with VIF and GWAS [discrete]
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--vif 5 --iter 1

# Running the munging script with addit, VIF, and GWAS [discrete]
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--addit examples/discrete/training_addit.csv \
--vif 5 --iter 1

# Running the munging script with featureSelection [discrete]
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--featureSelection 50

# Running the munging script with everything [discrete]
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--addit examples/discrete/training_addit.csv \
--p 0.01 \
--vif 5 --iter 1

# Running the munging script [continuous]
GenoMLMunging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv 

# Running the munging script with VIF [continuous]
GenoMLMunging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--vif 5 \
--iter 2

# Running the munging script with GWAS [continuous]
GenoMLMunging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--gwas examples/continuous/example_GWAS.csv 

# Running the munging script with VIF and GWAS [continuous]
GenoMLMunging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--gwas examples/continuous/example_GWAS.csv \
--vif 5 --iter 1

# Running the munging script with addit, VIF, and GWAS [continuous]
GenoMLMunging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--gwas examples/continuous/example_GWAS.csv \
--addit examples/continuous/training_addit.csv \
--vif 5 --iter 1

# Running the munging script with everything [continuous]
GenoMLMunging --prefix outputs/test_continuous_geno \
--datatype c \
--geno examples/continuous/training \
--pheno examples/continuous/training_pheno.csv \
--gwas examples/continuous/example_GWAS.csv \
--addit examples/continuous/training_addit.csv \
--p 0.01 \
--vif 5 --iter 1

## TRAIN 
# Running the supervised training script [discrete]
GenoML discrete supervised train \
--prefix outputs/test_discrete_geno \
--rank_features run

# Running the supervised training script [continuous]
GenoML continuous supervised train \
--prefix outputs/test_continuous_geno \
--rank_features run

## TUNE 
# Running the supervised tuning script [discrete]
GenoML discrete supervised tune \
--prefix outputs/test_discrete_geno \
--max_tune 10 --n_cv 3

# Running the supervised tuning script [continuous]
GenoML continuous supervised tune \
--prefix outputs/test_continuous_geno \
--max_tune 10 --n_cv 3

## TEST