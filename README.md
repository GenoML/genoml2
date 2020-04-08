# How to Get Started with GenoML

### Introduction
This README is a brief look into how to structure arguments and what arguments are available at each phase for the GenoML CLI

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

  To create a virtual environment:

```bash
# To create a virtual environment 
conda activate GenoML

# MISC 
	# To deactivate the virutal environment
#conda deactivate GenoML	
	# To delete your virutal environment 
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

If you would like to munge just with genotypes (in PLINK binary format), the simplest command is the following: 
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file 
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
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

Well, what if you had GWAS summary statistics handy, and would like to incorporate that data in? You can do so by running the following:
```bash
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and a GWAS summary statistics file 
GenoMLMunging --prefix outputs/test_discrete_geno \
--datatype d \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv 
```

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

If you are interested in changing the number of iterations the tuning process goes through by modifying `--max-tune` *(default is 50)*, or the number of cross-validations by modifying `--n_cv` *(default is 5)*, this is what the command would look like: 
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
**UNDER ACTIVE DEVELOPMENT** 

<a id="5"></a>
## 5. Experimental Features
**UNDER ACTIVE DEVELOPMENT** 