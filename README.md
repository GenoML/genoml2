<p align="center">
  <img width="300" height="300" src="logo.png">
</p>

# How to Get Started with GenoML

### Introduction
GenoML is an Automated Machine Learning (AutoML) for genomics data. In general, use a Linux or Mac with Python >3.5 for best results.

This README is a brief look into how to structure arguments and what arguments are available at each phase for the GenoML CLI. 

### Table of Contents 
#### [0. (OPTIONAL) How to Set Up a Virtual Environment via Conda](#0)
#### [1. Munging with GenoML](#1)
#### [2. Training with GenoML](#2)
#### [3. Tuning with GenoML](#3)
#### [4. Testing/Validating with GenoML](#4)
#### [5. Experimental Features](#5)

<a id="0"></a>
## 0. [OPTIONAL] How to Set Up a Virtual Environment via Conda

You can create a virtual environment to run GenoML, if you prefer.
If you already have the Anaconda Distribution, this is fairly simple.

To create and activate a virtual environment:

```bash
# To create a virtual environment
conda create -n GenoML python=3.7

# To activate a virtual environment
conda activate GenoML

## MISC
# To deactivate the virtual environment
# conda deactivate GenoML

# To delete your virtual environment
# conda env remove -n GenoML
```

To install the package at this path:
```bash
# Install the package at this path
pip install .

# MISC
	# To save out the environment requirements to a .txt file
# pip freeze > requirements.txt

	# Removing a conda virtualenv
# conda remove --name GenoML --all 
```
<a id="1"></a>
## 1. Munging with GenoML

Munging with GenoML was written as a separate portion in case you did not want to preprocess using GenoML, so it's structured a little differently from the rest of the package.

**Required** arguments for munging are `--prefix` , `--pheno` and `--datatype` 
- `--prefix` : Where and what would you like to name this file when it is munged?
- `--pheno` : Where is your phenotype file? This file only has 2 columns, ID in one, and PHENO in the other (0 for controls and 1 for cases)
- `--datatype`: What type of data are you putting into GenoML? The options for ``--datatype`` are `d` for discrete or `c` for continuous

*Note:* The following examples are for discrete data, but if you substitute following commands with `--datatype c`, you can preprocess your continuous data!

*Note:* Be sure to have your files formatted the same as the examples, key points being: 
- 0=controls and 1=case in your PLINK and phenotype files 
- Your phenotype file consisting only of the "ID" and "PHENO" columns
- Your sample IDs matching across all files
- Your sample IDs not consisting with only integers (add a prefix or suffix to all sample IDs if this is the case prior to running GenoML)  

If you would like to munge just with genotypes (in PLINK binary format), the simplest command is the following: 
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file 
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv 
```

You can choose to impute on `mean` or `median` by modifying the `--impute` flag, like so *(default is median)*:
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file and specifying impute
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--impute median \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv 
```

If you suspect collinear variables, and think this will be a problem for training the model moving forward, you can use VIF: 
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file while using VIF to remove multicollinearity 
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--vif 5 \
--iter 1
```
The `--vif` flag specifies the VIF threshold you would like to use (5 is recommended) and the number of iterations you'd like to run can be modified with the `--iter` flag (if you have or anticipate many collinear variables, it's a good idea to increase the iterations)


Well, what if you had GWAS summary statistics handy, and would like to incorporate that data in? You can do so by running the following:
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and a GWAS summary statistics file 
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv 
```
*Note:* When using the GWAS flag, the PLINK binaries will be pruned to include matching SNPs to the GWAS file. 

...and if you wanted to add a p-value cut-off...
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and a GWAS summary statistics file with a p-value cut-off 
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--p 0.01
```

Do you have additional data you would like to incorporate? Perhaps clinical, demographic, or transcriptomics data? If coded and all numerical, these can be added as an `--addit` file by doing the following: 
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and an addit file
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv 
```
You also have the option of not using PLINK binary files if you would like to just preprocess (and then, later train) on a phenotype and addit file by doing the following:
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and an addit file
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv 
```
Are you interested in selecting and ranking your features? If so...:
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and running feature selection 
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--featureSelection 50
```
This will output an `*_approx_feature_importance.txt` file with the features most contributing to your model at the top. 

<a id="2"></a>
## 2. Training with GenoML
Training, tuning, and testing with GenoML have a slightly different structure where there are 3 required arguments before moving forward

If you were to run just `GenoML` in the command line, you would see the following:
```bash
# Run GenoML
GenoML
#usage: GenoML [-h] [--prefix PREFIX] [--geno GENO] [--addit ADDIT]
            #  [--pheno PHENO] [--gwas GWAS] [--p P] [--vif VIF] [--iter ITER]
            #  [--impute IMPUTE] [,run}]
            #  [--max_tune MAX_TUNE] [--n_cv N_CV]
            #  {discrete,continuous} {supervised,unsupervised} {train,tune}
#GenoML: error: the following arguments are required: data, method, mode
```

**Required** arguments for GenoML are the following: 
- `data` : Is the data `continuous` or `discrete`?
- `method`: Do you want to use `supervised` or `unsupervised` machine learning? *(unsupervised currently under development)*
- `mode`:  would you like to `train` or `tune` your model?
- `--prefix` : Where would you like your outputs to be saved?

The most basic command to train your model looks like the following, it looks for the `*.dataForML` file that was generated in the munging step: 
```bash
# Running GenoML supervised training after munging on discrete data
GenoML discrete supervised train \
--prefix outputs/test_discrete_geno
```

If you would like to determine the best competing algorithm by something other than the AUC, you can do so by changing the `--metric_max` flag (Options include `AUC`, `Balanced_Accuracy`, `Sensitivity`, and `Specificity`) :

```bash
# Running GenoML supervised training after munging on discrete data and specifying the metric to maximize by 
GenoML discrete supervised train \
--prefix outputs/test_discrete_geno \
--metric_max Balanced_Accuracy
```
<a id="3"></a>
## 3. Tuning with GenoML
Training, tuning, and testing with GenoML have a slightly different structure where there are 3 required arguments before moving forward

**Required** arguments for GenoML are the following: 
- `data` : Is the data `continuous` or `discrete`?
- `method`: Do you want to use `supervised` or `unsupervised` machine learning? *(unsupervised currently under development)*
- `mode`:  would you like to `train` or `tune` your model?
- `--prefix` : Where would you like your outputs to be saved?

The most basic command to tune your model looks like the following, it looks for the file that was generated in the training step: 
```bash
# Running GenoML supervised tuning after munging and training on discrete data
GenoML discrete supervised tune \
--prefix outputs/test_discrete_geno 
```

If you are interested in changing the number of iterations the tuning process goes through by modifying `--max_tune` *(default is 50)*, or the number of cross-validations by modifying `--n_cv` *(default is 5)*, this is what the command would look like: 
```bash
# Running GenoML supervised tuning after munging and training on discrete data, modifying the number of iterations and cross-validations 
GenoML discrete supervised tune \
--prefix outputs/test_discrete_geno \
--max_tune 50 --n_cv 5
```

If you are interested in tuning on another metric other than AUC *(default is AUC)*, you can modify `--metric_tune` (options are `AUC` or `Balanced_Accuracy`) by doing the following: 
```bash
# Running GenoML supervised tuning after munging and training on discrete data, modifying the metric to tune by
GenoML discrete supervised tune \
--prefix outputs/test_discrete_geno \
--metric_tune Balanced_Accuracy
```
<a id="4"></a>
## 4. Testing/Validation with GenoML
**UNDER ACTIVE DEVELOPMENT!** 

In order to properly test how your model performs on a dataset it's never seen before (but you start with different PLINK binaries), we have created the harmonization step that will:
1. Keep only the same SNPs between your reference dataset and the dataset you are using for validation
2. Force the reference alleles in the validation dataset to match your reference dataset
3. Export a `.txt` file with the column names from your reference dataset to later use in the munging of your validation dataset 

If using GenoML for both your reference dataset and then your validation dataset, the process will look like the following: 

1. Munge and train your first dataset
    	- That will be your “reference” model
2. Use the outputs of step 1's munge for your reference model to harmonize your incoming validation dataset
3.  Run through harmonization step with your validation dataset
4.  Run through munging with your newly harmonized dataset
5.  Retrain your reference model with only the matching columns of your unseen data 
	- Given the nature of ML algorithms, you cannot test a model on a set of data that does not have identical features
6. Test your newly retrained reference model on the unseen data

### Harmonizing your Validation/Test Dataset 
**Required** arguments for GenoMLHarmonizing are the following: 
- `--test_geno_prefix` : What is the prefix of your validation dataset PLINK binaries?
- `--test_prefix`: What do you want the output to be named?
- `--refModel_prefix`:  What is the name of the previously GenoML-munged dataset you would like to use as your reference dataset? (Without the `.dataForML.h5` suffix)
- `--training_SNPsAlleles` : What are the SNPs and alleles you would like to use? (This is generated at the end of your previously-GenoML munged dataset with the suffix `variants_and_alleles.tab`)

To harmonize your incoming validation dataset to match the SNPs and alleles to your reference dataset, the command would look like the following:
```bash
# Running GenoMLHarmonizing 
GenoMLHarmonizing --test_geno_prefix examples/discrete/validation \
--test_prefix outputs/validation_test_discrete_geno \
--refModel_prefix outputs/test_discrete_geno \
--training_SNPsAlleles outputs/test_discrete_geno.variants_and_alleles.tab
```
This step will generate: 
- a `*_refColsHarmonize_toKeep.txt` file of columns to keep for the next step 
- `*_refSNPs_andAlleles.*` PLINK binary files (.bed, .bim, and .fam) that have the SNPs and alleles match your reference dataset

Now that you have harmonized your validation dataset to your reference dataset, you can now munge using a command similar to the following:
```bash
# Running GenoMLMunging after GenoMLHarmonizing
GenoMLMunging --prefix outputs/validation_test_discrete_geno \
--datatype d \
--geno outputs/validation_test_discrete_geno_refSNPs_andAlleles \
--pheno examples/discrete/validation_pheno.csv \
--addit examples/discrete/validation_addit.csv \
--refColsHarmonize outputs/validation_test_discrete_geno_refColsHarmonize_toKeep.txt
```
All munging options discussed above are available at this step, the only difference here is you will add the `--refColsHarmonize` flag to include the `*_refColsHarmonize_toKeep.txt` file generated at the end of harmonizing to only keep the same columns that the reference dataset had. 

After munging and training your reference model and harmonizing and munging your unseen test data, **you will retrain your reference model to include only matching features**. Given the nature of ML algorithms, you cannot test a model on a set of data that does not have identical features. 

To retrain your model appropriately, after munging your test data with the `--refColsHarmonize` flag, a final columns list will be generated at `*_finalHarmonizedCols_toKeep.txt`. This includes all the features that match between your unseen test data and your reference model. Use the `--matchingCols` flag when retraining your reference model to use the appropriate features.

When retraining of the reference model is complete, you are ready to test!

A step-by-step guide on how to achieve this is listed below:
```bash
# 0. MUNGE THE REFERENCE DATASET
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv

# 1. TRAIN THE REFERENCE DATASET
GenoML discrete supervised train \
--prefix outputs/test_discrete_geno

# 2. HARMONIZE TEST DATASET
GenoMLHarmonizing --test_geno_prefix examples/discrete/validation \
--test_prefix outputs/validation_test_discrete_geno \
--refModel_prefix outputs/test_discrete_geno \
--training_SNPsAlleles outputs/test_discrete_geno.variants_and_alleles.tab

# 3. MUNGE THE TEST DATASET ON REFERENCE MODEL COLUMNS
GenoMLMunging --prefix outputs/validation_test_discrete_geno \
--datatype d \
--geno outputs/validation_test_discrete_geno_refSNPs_andAlleles \
--pheno examples/discrete/validation_pheno.csv \
--addit examples/discrete/validation_addit.csv \
--refColsHarmonize outputs/validation_test_discrete_geno_refColsHarmonize_toKeep.txt

# 4. RETRAIN REFERENCE MODEL ON INTERSECTING COLUMNS BETWEEN REFERENCE AND TEST
GenoML discrete supervised train \
--prefix outputs/test_discrete_geno \
--matchingCols outputs/validation_test_discrete_geno_finalHarmonizedCols_toKeep.txt
  
# 5. TEST RETRAINED REFERENCE MODEL ON UNSEEN DATA
GenoML discrete supervised test \
--prefix outputs/validation_test_discrete_geno \
--test_prefix outputs/validation_test_discrete_geno \
--refModel_prefix outputs/test_discrete_geno.trainedModel
```

<a id="5"></a>
## 5. Experimental Features
**UNDER ACTIVE DEVELOPMENT** 

Planned experimental features include, but are not limited to:
- Unsupervised training, tuning, and testing
- Network analyses
- Meta-learning
- Federated learning
- Biobank-scale support
- ...?
