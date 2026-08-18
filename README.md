# GenoML 

<p align="center">
  <img width="300" height="300" src="https://github.com/GenoML/genoml2/blob/master/logo.png">
</p>

[![Downloads](https://static.pepy.tech/personalized-badge/genoml2?period=total&units=international_system&left_color=black&right_color=grey&left_text=Downloads)](https://pepy.tech/project/genoml2)

> Updated 19 May 2026. Previous release (17 June 2025) on pip! v1.5.4

# How to Get Started with GenoML


### Introduction
[GenoML (**Geno**mics + **M**achine **L**earning)](https://genoml.com) is an automated Machine Learning (autoML) for genomics data. In general, use a Linux or Mac with Python 3.9-3.12 for best results. **This [repository](https://github.com/GenoML/genoml2) and [pip package](https://pypi.org/project/genoml2/) are under active development!** 

This README is a brief look into how to structure arguments and what arguments are available at each phase for the GenoML CLI. 

If you are using GenoML for your own work, please cite the following papers: 
- Makarious, M. B., Leonard, H. L., Vitale, D., Iwaki, H., Saffo, D., Sargent, L., ... & Faghri, F. (2021). GenoML: Automated Machine Learning for Genomics. arXiv preprint arXiv:2103.03221
- Makarious, M. B., Leonard, H. L., Vitale, D., Iwaki, H., Sargent, L., Dadu, A., ... & Nalls, M. A. (2022). Multi-Modality Machine Learning Predicting Parkinson’s Disease. NPJ Parkinson's Disease.

### Installing + Downloading Example Data 
- Within your conda environment, ensure necessary conda packages are installed first

`conda install -c conda-forge numba llvmlite -y`

- Install this repository directly from GitHub (from source; master branch)

`git clone https://github.com/GenoML/genoml2.git`

- Install using pip or upgrade using pip

`pip install genoml2`

OR

`pip install genoml2 --upgrade`

### CHANGELOG
- 19-MAY-2026: We have added outer cross-validation functionality across all modules allowing the user to create an n-fold split of their data upfront to be used turing the tuning and testing steps. We also refined the continuous prediction module, particularly with the output plots, and the multiclass prediction module, which now uses the same models that are used for binary prediction without a OvR wrapper needed. We additionally now include Pearson's Correlation and OLS as filtering steps during munging. As always, we squashed a few bugs along the way. `README` updated to reflect these changes.
- 16-JUN-2025: Addition of multiclass prediction functionality using the same base models that are used for the discrete module. We have additionally restructured the munging functionality to allow users to process training and testing data all at once to ensure they are munged under the same conditions, as well as including multiple GWAS summary stats files for SNP filtering. We also upgraded from plink1.9 to plink2 for genomic data processing. Finally, we have added a log file in the output directory to facilitate full reproducbility of results. `README` updated to reflect these changes.
- 8-OCT-2024: Big changes to output file structure, so now output files go in subdirectories named for each step, and prefixes are not required. `README` updated to reflect these changes.


### Table of Contents 
#### [0. (OPTIONAL) How to Set Up a Virtual Environment via Conda](#0)
#### [1. Munging with GenoML](#1)
#### [1b. Harmonizing with GenoML](#1b)
#### [2. Training with GenoML](#2)
#### [3. Tuning with GenoML](#3)
#### [4. Testing/Validating with GenoML](#4)
#### [5. Full pipeline example](#5)
#### [6. Experimental Features](#6)

<a id="0"></a>
## 0. [OPTIONAL] How to Set Up a Virtual Environment via Conda

You can create a virtual environment to run GenoML, if you prefer.
If you already have the [Anaconda Distribution](https://www.anaconda.com/products/individual#download), this is fairly simple.

To create and activate a virtual environment:

```shell
# To create a virtual environment
conda create -n GenoML python=3.12 # or whatever version of python you prefer

# To activate a virtual environment
conda activate GenoML

# To install requirements via pip 
pip install -r requirements.txt
    # If issues installing xgboost from requirements - (3 options)
        # Option 1: use Homebrew to 
            # xcode-select --install
            # brew install gcc@7
        # or Option 2: conda install -c conda-forge xgboost 
        # or Option 3: pip install xgboost==2.0.3
    # If issues installing umap 
        # pip install umap-learn
    # If issues installing pytables/dependency issue 
        # conda install -c conda-forge pytables
    # If issues with blosc 
        # conda install -c conda-forge tables blosc
```

To install the GenoML in the user's path in a virtual environment, you can do the following:
```shell
# Navigate to local directory with GenoML code
cd path_to_genoml2_directory

# Install the package at this path
pip install .
```

> *Note:* The following examples are for `discrete` data, but if you substitute following commands with `continuous` or `multiclass` instead of `discrete`, you can munge, harmonize, train, tune, and test your `continuous` or `multiclass` data! 

<a id="1"></a>
## 1. Munging with GenoML

Munging with GenoML will, at minimum, do the following: 
- Prune your genotypes using PLINK v2 (if `--geno` flag is used)
- Impute per column using median (default) or mean (if indicated with the `--impute` flag)
- Z-scaling of features and removing columns with a std dev = 0 

**Required** arguments for GenoML munging are `--prefix` and `--pheno` 
- `data` : Are the data `continuous`, `discrete`, or `multiclass`?
- `method`: Do you want to use `supervised` or `unsupervised` machine learning? *(`unsupervised` currently under development)*
- `mode`:  would you like to `munge`, `harmonize`, `train`, `tune`, or `test` your model? Here, you will use `munge`.
- `--prefix` : Where would you like your outputs to be saved?
- `--pheno` : Where is your phenotype file? This file only has 2 columns, ID in one, and PHENO in the other (0 for controls and 1 for cases when using the `discrete` module, 0, ..., *n-1* when using the `multiclass` module for *n* distinct phenotypes, or numeric values when using the `continuous` module).


Be sure to have your files formatted the same as the examples, key points being: 
- Your phenotype file consisting **only** of the "ID" and "PHENO" columns
- Your sample IDs matching across all files
- Please avoid the use of characters like commas, semi-colons, etc. in the column headers (it is Python after all!)  

If you would like to munge just with genotypes (in PLINK binary format), the simplest command is the following: 
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file 

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv
```

If you would like a more detailed log printed to your console, you may use the `--verbose` flag as follows: 
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file with a detailed log printed to the console

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--verbose
```
> *Note:* The `--verbose` flag may be used like this for any GenoML command, not just munging.

To properly evaluate your model, it must be applied to (a) dataset(s) it's never seen before. One way of doing this is using cross-validation, in which you train multiple models in parallel with different subsets of the full dataset being used to train the model vs. being withheld to test how well it generalizes. GenoML allows you to define the number of unique splits/models (folds) to use, which will result in the data being split during the munging stepto maintain consistency while cross-validation is used through the tuning and testing steps. This can be done using the `--n_outer_cv` flag:
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and phenotype files using 10-fold cross-validation.

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--n_outer_cv 10
```

You may also evaluate model accuracy by providing separate training and testing datasets. This will train and tune one model on the training datset, and apply it to the testing dataset. To do this with your training and testing phenotype/genotype data, the simplest command is the following: 
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and phenotype files for both the training and testing datasets.

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--geno_test examples/discrete/validation \
--pheno_test examples/discrete/validation_pheno.csv
```

> *Note:* It is strongly discouraged to generate a model that has not been evaluated on withheld data. However, you may also elect to fit the model on your full dataset without evaluating it, and later harmonizing a new dataset for evaluation. Thus, model fitting can be done without specifying any cross-validation folds or testing data. If this is done, the testing step will fail to execute until you harmonize a new dataset using the `harmonize` module, which is discussed in more detail below.

If you would like to control the pruning stringency in genotypes: 
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file 

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--r2_cutoff 0.3 \
--pheno examples/discrete/training_pheno.csv
```

You can choose to skip pruning your SNPs at this stage by including the `--skip_prune` flag
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file 

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--skip_prune \
--pheno examples/discrete/training_pheno.csv
```

You can choose to impute on `mean` or `median` by modifying the `--impute` flag, like so *(default is median)*:
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file and specifying impute

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--impute mean
```

Well, what if you had GWAS summary statistics handy, and would like to just use the same SNPs outlined in that file? You can do so by running the following:
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and a GWAS summary statistics file 

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv
```
> *Note:* When using the GWAS flag, the PLINK binaries will be pruned to include matching SNPs to the GWAS file. 

And if you have more than one GWAS summary statistics file, we support that too! Just use the `--gwas` flag multiple times, once for each of the files you would like to include, as follows:
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and two GWAS summary statistics files

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--gwas examples/discrete/example_GWAS_2.csv
```
> *Note:* This is particularly helpful when using the `multiclass` module when you have multiple phenotypes of interest and would like to include SNPs that are relevant for each phenotype. 

...and if you wanted to add a p-value cut-off...
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and a GWAS summary statistics file with a p-value cut-off 

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--gwas examples/discrete/example_GWAS.csv \
--p 0.01
```

Do you have additional data you would like to incorporate? Perhaps clinical, demographic, or other -omics data? If coded and all numerical, these can be added as an `--addit` file by doing the following: 
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and an addit file

genoml discrete supervised munge \
--prefix outputs \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv
```

Are you interested in pruning your data to keep only the most important features as inputs for your model? GenoML provides several methods for removing extraneous features, which can be used in isolation or in combination with each other.

If you would like to use a tree-based pruning method, you can use the `--n_trees` flag and specify the number of trees like so:
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and running ExtraTrees for feature selection

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv \
--n_trees 50
```
The `--n_trees` flag uses extraTrees ([classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) for discrete data; [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) for continuous data) to output an `*.approx_feature_importance.txt` file with the features most contributing to your model at the top. 

If you would like to use a linear regression-based method, you can use the `--ols_threshold` flag to specify the p-value for the relationship between each individual feature and the dependent variable, above which features will be dropped, as such:
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files, a phenotype file, and running OLS for feature selection 

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--addit examples/discrete/training_addit.csv \
--ols_threshold 0.05
```

If you suspect collinear variables, and think this will be a problem for training the model moving forward, you have two options. One option is to use [variance inflation factor](https://www.investopedia.com/terms/v/variance-inflation-factor.asp) (VIF) filtering, using the `--vif` flag to specify the VIF threshold you would like to use (5 is recommended) and the `--vif_iter` flag to specify the number of iterations you'd like to run (if you have or anticipate many collinear variables, it's a good idea to increase the iterations): 
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file while using VIF to remove multicollinearity 

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--vif 5 \
--vif_iter 3
```

Another option is to use Pearson's Correlation for filtering. To do this, use the `--pearson_threshold` flag to specify the correlation coefficient (Pearson's r) value (0-1) above which features will be dropped: 
```shell
# Running GenoML munging on discrete data using PLINK genotype binary files and a phenotype file while using Pearson's Correlation to remove multicollinearity 

genoml discrete supervised munge \
--prefix outputs \
--geno examples/discrete/training \
--pheno examples/discrete/training_pheno.csv \
--pearson_threshold 0.9
```

Do you have additional covariates and confounders you would like to adjust for in the munging step prior to training your model and/or would like to reduce your data? To adjust, use the `--adjust_data` flag with the following necessary flags: 
- `--target_features`: A .txt file, one column, with a list of features to adjust (no header). These should correspond to features in the munged dataset
- `--confounders`: A .csv of confounders to adjust for with ID column and header. Numeric, with no missing data and the ID column is mandatory (this can be PCs, for example)

You may also include the following optional flag:
- `--adjust_normalize`: Would you like to normalize your final adjusted data?

To reduce your data prior to adjusting, use the `--umap_reduce` flag. This flag will also prompt you for if you want to adjust your data, normalize, and what your target features and confounders might be. We use the [Uniform Manifold Approximation and Projection for Dimension Reduction](https://umap-learn.readthedocs.io/en/latest/) (UMAP) to reduce your data into 2D, adjust, and export a plot and an adjusted dataframe moving forward. This can be done by running the following: 

```shell
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
```

And if you're munging your training and testing data together, you must include a confounders file for your test dataset as well using the `--confounders_test` flag:

```shell
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
```

Here, the `--confounders` and `--confounders_test` flags take in datasets of features that should be accounted for. This is a .csv file with the ID column and header included and is numeric with no missing data. **The ID column is mandatory.** The `--target_features` flag takes in a .txt with a list of features (column names) you are adjusting for.

<a id="1b"></a>
## 1b. Harmonizing with GenoML
GenoML allows you to munge your testing data separately from your training data as well using the harmonization feature. This is particularly helpful if you would like to apply a model pre-trained elsewhere on your own datasets. This will apply the same preprocessing and normalization parameters that were used during the original munging step to ensure your datasets are consistent with the original model inputs.

**Required** arguments for GenoML harmonizing are the following: 
- `data`: Are the data `continuous`, `discrete`, or `multiclass`?
- `method`: Do you want to use `supervised` or `unsupervised` machine learning? *(`unsupervised` currently under development)*
- `mode`:  would you like to `munge`, `harmonize`, `train`, `tune`, or `test` your model? Here, you will use `harmonize`.
- `--prefix` : Where would you like your outputs to be saved?
- `--pheno` : Where is your phenotype file? This file only has 2 columns: ID in one, and PHENO in the other (0 for controls and 1 for cases when using the `discrete` module, 0, ..., *n-1* when using the `multiclass` module for *n* distinct phenotypes, or numeric values when using the `continuous` module).

Be sure to have your files formatted the same as the examples, key points being: 
- Your phenotype file consisting **only** of the "ID" and "PHENO" columns
- Your sample IDs matching across all files
- Please avoid the use of characters like commas, semi-colons, etc. in the column headers (it is Python after all!)  

> *Note:* The following examples are for discrete data, but if you substitute following commands with `continuous` or `multiclass` instead of discrete, you can preprocess your continuous/multiclass data!

If you would like to harmonize just with genotypes (in PLINK binary format), the simplest command is the following: 
```shell
# Running GenoML harmonization on discrete data using PLINK genotype binary files and a phenotype file 

genoml discrete supervised harmonize \
--prefix outputs \
--geno examples/discrete/validation \
--pheno examples/discrete/validation_pheno.csv
```
> *Note:* You must use the same `--prefix` that was used for training. This is how GenoML will know where to look for your munged data!

If the training data were adjusted by confounders, you must include a file with the same features to adjust your harmonized data accordingly. You can do this by providing a path to this file using the `--confounders` flag (see "**1. Munging with GenoML**" for further explanation) as follows:
```shell
# Running GenoML harmonization on discrete data using PLINK genotype binary files and a phenotype file 

genoml discrete supervised harmonize \
--prefix outputs \
--geno examples/discrete/validation \
--pheno examples/discrete/validation_pheno.csv \
--confounders examples/discrete/validation_addit_confounder_example.csv
```

Machine learning models require that your datasets include all of the features that were used to train the model. Because of this, we (and we cannot emphasize this enough) STRONGLY recommend that your harmonization dataset include every feature used in the model. However, if for some reason this is not possible and you would like to test a pre-trained model on your data anyways, we provide the option of adding the entire set of missing features to your harmonization dataset. This will take the average value from the training dataset for each feature (as determined from the `--impute` flag used during munging) and use that same value for each of your samples. You may do so using the `--force_impute` flag as follows:
```shell
# Running GenoML harmonization on discrete data using PLINK genotype binary files and a phenotype file, while imputing any missing columns (ie, if an addit file was used during training and is not present for the harmonization samples).

genoml discrete supervised harmonize \
--prefix outputs \
--geno examples/discrete/validation \
--pheno examples/discrete/validation_pheno.csv \
--force_impute
```

<a id="2"></a>
## 2. Training with GenoML
Training with GenoML competes a number of different algorithms and outputs the best algorithm based on a specific metric that can be tweaked using the `--metric_max` flag *(default is AUC for `discrete` and `multiclass` data, and Explained Variance for `continuous` data)*.

**Required** arguments for GenoML training are the following: 
- Are the data `continuous`, `discrete`, or `multiclass`?
- `method`: Do you want to use `supervised` or `unsupervised` machine learning? *(unsupervised currently under development)*
- `mode`:  would you like to `munge`, `harmonize`, `train`, `tune`, or `test` your model? Here, you will use `train`.
- `--prefix` : Where would you like your outputs to be saved?

The most basic command to train your model looks like the following: 
```shell
# Running GenoML supervised training after munging on discrete data

genoml discrete supervised train \
--prefix outputs
```
> *Note:* You must use the same `--prefix` that was used for training. This is how GenoML will know where to look for the `train_dataset.h5` file with your munged data!

If you would like to determine the best competing algorithm by something other than the AUC, you can do so by changing the `--metric_max` flag (Options include `AUC`, `Balanced_Accuracy`, `Sensitivity`, and `Specificity` for `discrete`, `AUC` for `multiclass`, or `Explained_Variance`, `Root_Mean_Squared_Error`, `Median_Absolute_Error`, and `Pearson_R` for `continuous` datasets):
```shell
# Running GenoML supervised training after munging on discrete data and specifying Sensitivity as the metric to optimize

genoml discrete supervised train \
--prefix outputs \
--metric_max Sensitivity
```

The training module automatically splits your training data, with 70% being used to directly fit the model and the other 30% being withheld to evaluate which model class generalizes best to unseen data. If you would like to use a different split, you may use `--train_split` to define the percentage of samples used for training, with the remainder being withheld for validation `(Note: Numbers less than 1 will be treated as fractions, with numbers greater than 1 treated as percentages)`:
```shell
# Running GenoML supervised training after munging on discrete data and electing to use an 80/20 train/validation split

genoml discrete supervised train \
--prefix outputs \
--train_split 80
```
or
```shell
# Running GenoML supervised training after munging on discrete data and electing to use an 80/20 train/validation split

genoml discrete supervised train \
--prefix outputs \
--train_split 0.8
```

<a id="3"></a>
## 3. Tuning with GenoML
Tuning with GenoML applies fine-tuning with cross-validation using the trained model as a starting point to find the best set of hyperparameters for your datasets.

**Required** arguments for GenoML training are the following: 
- `data`: Are the data `continuous`, `discrete`, or `multiclass`?
- `method`: Do you want to use `supervised` or `unsupervised` machine learning? *(`unsupervised` currently under development)*
- `mode`:  would you like to `munge`, `harmonize`, `train`, `tune`, or `test` your model? Here, you will use `tune`.
- `--prefix` : Where would you like your outputs to be saved?

The most basic command to tune your model looks like the following: 
```shell
# Running GenoML supervised tuning after munging and training on discrete data

genoml discrete supervised tune \
--prefix outputs
```
> *Note:* You must use the same `--prefix` that was used for training. This is how GenoML will know where to look for the `train_dataset.h5` file with your munged data!

If you are interested in changing the number of iterations the tuning process goes through by modifying `--max_tune` *(default is 50)*, or the number of cross-validations by modifying `--n_inner_cv` *(default is 5)*, this is what the command would look like: 
```shell
# Running GenoML supervised tuning after munging and training on discrete data, modifying the number of iterations and cross-validations 

genoml discrete supervised tune \
--prefix outputs \
--max_tune 10 \
--n_inner_cv 3
```

If you are interested in tuning on another metric other than AUC *(default is AUC)*, you can modify `--metric_tune` (Options include `AUC` and `Balanced_Accuracy` for `discrete` datasets, `AUC` for `multiclass` datasets, or `Explained_Variance`, `Rot_Mean_Squared_Error`, `Median_Absolute_Error`, and `Pearson_R` for `continuous` datasets) by doing the following: 
```shell
# Running GenoML supervised tuning after munging and training on discrete data, modifying the metric to tune by

genoml discrete supervised tune \
--prefix outputs \
--metric_tune Balanced_Accuracy
```

<a id="4"></a>
## 4. Testing/Validation with GenoML
Testing/validation with GenoML applies your fully-tuned model on a new dataset to evaluate how well its performance generalizes beyond data it was trained on.

**Required** arguments for GenoML training are the following: 
- `data`: Are the data `continuous`, `discrete`, or `multiclass`?
- `method`: Do you want to use `supervised` or `unsupervised` machine learning? *(`unsupervised` currently under development)*
- `mode`:  would you like to `munge`, `harmonize`, `train`, `tune`, or `test` your model? Here, you will use `test`.
- `--prefix` : Where would you like your outputs to be saved?

```shell
# Running GenoML test

genoml discrete supervised test \
--prefix outputs
```

<a id="5"></a>
## 5. Full pipeline example

A step-by-step guide on how to achieve this is listed below:
```shell
# 0. MUNGE THE REFERENCE DATASET
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
--n_trees 50 \
--adjust_data \
--adjust_normalize \
--umap_reduce \
--confounders examples/discrete/training_addit_confounder_example.csv \
--confounders_test examples/discrete/validation_addit_confounder_example.csv \
--target_features examples/discrete/to_adjust.txt \
--verbose
# Files made: 
    # outputs/log.txt
    # outputs/Munge/approx_feature_importance.txt
    # outputs/Munge/list_features.txt
    # outputs/Munge/params.pkl
    # outputs/Munge/p_threshold_variants.tab
    # outputs/Munge/test_dataset.h5
    # outputs/Munge/train_dataset.h5
    # outputs/Munge/umap_clustering.joblib
    # outputs/Munge/umap_data_reduction_test.txt
    # outputs/Munge/umap_data_reduction_train.txt
    # outputs/Munge/umap_plot_test.png
    # outputs/Munge/umap_plot_train.png
    # outputs/Munge/variants_and_alleles.tab
    # outputs/Munge/variants.txt

# 1. TRAIN THE REFERENCE MODEL
genoml discrete supervised train \
--prefix outputs \
--metric_max Balanced_Accuracy
# Files made: 
    # outputs/model.joblib
    # outputs/algorithm.txt
    # outputs/Train/precision_recall.png
    # outputs/Train/predictions.tsv
    # outputs/Train/probabilities.png
    # outputs/Train/roc.png
    # outputs/Train/train_predictions.tsv
    # outputs/Train/withheld_performance_metrics.tsv
# Files updated: 
    # outputs/log.txt

# 2. OPTIONAL: TUNING YOUR REFERENCE MODEL
genoml discrete supervised tune \
--prefix outputs \
--max_tune 10 \
--n_inner_cv 3 \
--metric_tune Balanced_Accuracy
# Files made: 
    # outputs/Tune/cv_summary.tsv
    # outputs/Tune/precision_recall.png
    # outputs/Tune/predictions.tsv
    # outputs/Tune/probabilities.png
    # outputs/Tune/roc.png
    # outputs/Tune/tuning_summary.tsv
# Files updated: 
    # outputs/model.joblib
    # outputs/log.txt

# 3. TEST TUNED MODEL ON UNSEEN DATA
genoml discrete supervised test \
--prefix outputs
# Files made: 
    # outputs/Test/performance_metrics.tsv
    # outputs/Test/precision_recall.png
    # outputs/Test/predictions.tsv
    # outputs/Test/probabilities.png
    # outputs/Test/roc.png
# Files updated: 
    # outputs/log.txt
```

<a id="6"></a>
## 6. Experimental Features
**UNDER ACTIVE DEVELOPMENT** 

Planned experimental features include, but are not limited to:
- Support for unsupervised learning models
- Multilabel prediction
- GWAS QC and Pipeline
- Network analyses
- Multi-omic munging
- Meta-learning
- Federated learning
- Cross-silo checks for genetic duplicates
- Outlier detection
- ...?
