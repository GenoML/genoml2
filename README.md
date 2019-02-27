# GenoML-core
GenoML is an Automated Machine Learning (AutoML) for Genomic. This is the core package of GenoML. 
Please note this repo is under development for "functional testing". This is not the end product, just for preliminary evaluation of software logic. 
Package website and on-going documentation: https://genoml.github.io

## Goals 

Please test the code in different environments and on different datasets. The goal is to resolve the following issues:

 - **Dependencies:** did you need to install any package not already [listed](https://github.com/GenoML/genoml-core/blob/master/otherPackages/readMe_otherPackages.txt)? do you get dependency errors? 
 - **Errors:** do you get any error? is the error not clear? 
 - **Wrong output:** any discrepancies in the expected output? 
 -  **Corner cases:** did the code break on a particular case you were testing? 
 - **Usability:** could we improve the way user interacts with the code? is there any particular feature or document you fit helpful? 

## Install
For now just download or clone the genoml-core repo and run the genoml with `python genoml.py`.

## Step-by-step examples 
Please refer to the following quick examples for running GonML (for full `usage`, please refer to [Usage](#usage)):

### Step 1 - genoml data-prune:
To perform `data-prune` only on `genotype` and `phenotype` data:

    python genoml.py data-prune --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno

To perform `data-prune`  on `genotype`, `phenotype` , `GWAS`, and `covariance` data:
 
    python genoml.py data-prune --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno  --gwas-file=./exampleData/example_GWAS.txt  

To perform `data-prune`  on `genotype`, `phenotype` , `GWAS`, `covariance`, and `additional` data:

    python genoml.py data-prune --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno --cov-file=./exampleData/training.cov --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit  

To perform `data-prune`  on `genotype`, `phenotype` , `GWAS`, and `additional` data, as well as `Heritability estimate`:

    python genoml.py data-prune --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno  --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit --herit=0.2  

To perform `data-prune`  on `genotype`, `phenotype` , `GWAS`, `covariance`, and `additional` data, as well as `Heritability estimate`:

    python genoml.py data-prune --geno-prefix=./exampleData/training --pheno-file=./exampleData/example.pheno --cov-file=./exampleData/training.cov --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit --herit=0.5 

### Step 2 - genoml model-train:
To perform `model-train`  on the output of `data-prune` with the prefix given to you from the prune step `prune-prefix=./tmp/20181225-230052`:
 
    python genoml.py model-train --prune-prefix=./tmp/20181225-230052 --pheno-file=./exampleData/training.pheno  

### Step 3 - genoml model-tune:
To perform `model-tune`  after `model-train`  on the output of `data-prune` with the prefix given to you from the prune step `prune-prefix=./tmp/20181225-230052`:

    python genoml.py model-tune --prune-prefix=./tmp/20181225-230052 --pheno-file=./exampleData/training.pheno

### Step 4 - genoml model-validation:
To perform external `model-validate` only when `genotype` and `phenotype` data present:

    python genoml.py model-validate --prune-prefix=./tmp/20181225-230052 --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno --valid-geno-prefix=./exampleData/validation --valid-pheno-file=./exampleData/validation.pheno

To perform external `model-validate`  when `genotype`, `phenotype`, and `GWAS` data present:
    
    python genoml.py model-validate --prune-prefix=./tmp/20181225-230052 --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno --valid-geno-prefix=./exampleData/validation --valid-pheno-file=./exampleData/validation.pheno --gwas-file=./exampleData/example_GWAS.txt

To perform external `model-validate`  when `genotype`, `phenotype`, `GWAS`, and `additional` data present:

    python genoml.py model-validate --prune-prefix=./tmp/20181225-230052 --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno --valid-geno-prefix=./exampleData/validation --valid-pheno-file=./exampleData/validation.pheno --gwas-file=./exampleData/example_GWAS.txt --valid-addit-file=./exampleData/validation.addit
    
To perform external `model-validate`  when `genotype`, `phenotype`, `GWAS`, `additional`, and `covariance` data present:

    python genoml.py model-validate --prune-prefix=./tmp/20181225-230052 --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno --valid-geno-prefix=./exampleData/validation --valid-pheno-file=./exampleData/validation.pheno --gwas-file=./exampleData/example_GWAS.txt --valid-addit-file=./exampleData/validation.addit --valid-cov-file=./exampleData/validation.cov


## Usage 
Full GenoML usage:

     Usage:
       genoml data-prune  (--geno-prefix=geno_prefix) (--pheno-file=<pheno_file>) [--gwas-file=<gwas_file>] [--cov-file=<cov_file>] [--herit=<herit>] [--addit-file=<addit_file>] [--temp-dir=<directory>]
       genoml model-train (--prune-prefix=prune_prefix) (--pheno-file=<pheno_file>) [--n-cores=<n_cores>] [--train-speed=<train_speed>] [--cv-reps=<cv_reps>] [--grid-search=<grid_search>] [--impute-data=<impute_data>]
       genoml model-tune (--prune-prefix=prune_prefix) (--pheno-file=<pheno_file>) [--cv-reps=<cv_reps>] [--grid-search=<grid_search>] [--impute-data=<impute_data>] [--best-model-name=<best_model_name>]
       genoml model-validate (--prune-prefix=prune_prefix) (--pheno-file=<pheno_file>) (--geno-prefix=geno_prefix) (--valid-geno-prefix=valid_geno_prefix) (--valid-pheno-file=<valid_pheno_file>) [--valid-cov-file=<valid_cov_file>] [--gwas-file=<gwas_file>] [--valid-addit-file=<valid_addit_file>] [--n-cores=<n_cores>] [--impute-data=<impute_data>]  [--best-model-name=<best_model_name>]
       genoml -h | --help
       genoml --version

     Options:
       --geno-prefix=geno_prefix               Prefix with path to genotype files in PLINK format, *.bed, *.bim and *.fam.
       --pheno-file=<pheno_file>               Path to the phenotype file in PLINK format, *.pheno.
       --gwas-file=<gwas_file>                 Path to the GWAS file, if available.
       --cov-file=<cov_file>                   Path to the covariance file, if available.
       --herit=<herit>                         Heritability estimate of phenotype between 0 and 1, if available.
       --addit-file=<addit_file>               Path to the additional file, if avialable.
       --temp-dir=<directory>                  Directory for temporary files [default: ./tmp/].
       --n-cores=<n_cores>                     Number of cores to be allocated for computation [default: 1].
       --prune-prefix=prune_prefix             Prefix given to you at the end of pruning stage.
       --train-speed=<train_speed>             Training speed: (ALL, FAST, FURIOUS, BOOSTED). Run all models, only  the fastest models, run slightly slower models, or just run boosted models which usually perform best when using genotype data [default: BOOSTED].
       --cv-reps=<cv_reps>                     Number of cross-validation. An integer greater than 5. Effects the speed [default: 5].
       --impute-data=<impute_data>             Imputation: (knn, median). Governs secondary imputation and data transformation [default: median].
       --grid-search=<grid_search>             Grid search length for parameters, integer greater than 10, 30 or greater recommended, effects speed of initial tune [default: 10].
       --best-model-name=<best_model_name>     Name for the best model [default: best_model].
       --valid-geno-prefix=valid_geno_prefix   Prefix with path to the validation genotype files in PLINK format, *.bed, *.bim and *.fam.
       --valid-pheno-file=<valid_pheno_file>   Path to the validation phenotype file in PLINK format, *.pheno.
       --valid-cov-file=<valid_cov_file>       Path to the validation covariance file, if available.
       --valid-addit-file=<valid_addit_file>   Path to the the validation additional file, if avialable.
       -h --help                               Show this screen.
       --version                               Show version.

     Examples:
       genoml data-prune --geno-prefix=./exampleData/example --pheno-file=./exampleData/training.pheno
       genoml data-prune --geno-prefix=./exampleData/example --pheno-file=./exampleData/training.pheno  --gwas-file=./exampleData/example_GWAS.txt
       genoml data-prune --geno-prefix=./exampleData/example --pheno-file=./exampleData/training.pheno --cov-file=./exampleData/training.cov --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit
       genoml data-prune --geno-prefix=./exampleData/example --pheno-file=./exampleData/training.pheno  --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit --herit=0.2
       genoml data-prune --geno-prefix=./exampleData/example --pheno-file=./exampleData/training.pheno --cov-file=./exampleData/training.cov --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit --herit=0.5
       genoml model-train --prune-prefix=./tmp/20181225-230052 --pheno-file=./exampleData/training.pheno
       genoml model-tune --prune-prefix=./tmp/20181225-230052 --pheno-file=./exampleData/training.pheno
       genoml model-validate --prune-prefix=./tmp/20181225-230052 --pheno-file=./exampleData/training.pheno --valid-geno-prefix=./exampleData/validation --valid-pheno-file=./exampleData/validation.pheno

     Help:
       For help using this tool, please open an issue on the Github repository:
       https://github.com/GenoML/genoml-core/issues
     
     


## Report issues 
Please report any issue or suggestions on the [GenoML-core issues page](https://github.com/GenoML/genoml-core/issues).
