# GenoML-core
GenoML is an Automated Machine Learning (AutoML) for Genomic. This is the core package of GenoML. 
this repo is under development, please report any issues (bug, performance, documentation) on the [GenoML issues page](https://github.com/GenoML/genoml/issues). 

## Install
Run `pip install genoml`.

## Run
Please refer to the following quick examples for running GonML (for full `usage`, please refer to [Usage](#usage)):

### Machine learning model train:
This step performs data pruning as well as model training and tunning.
Using `genotype` and `phenotype` data, run:

    genoml-train --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno --model-file=./model

Using `genotype`, `phenotype` , and `GWAS` data, run:
 
    genoml-train --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno  --gwas-file=./exampleData/example_GWAS.txt --model-file=./model

Using `genotype`, `phenotype` , `GWAS`, `covariance`, and `additional` data, run:

    genoml-train --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno --cov-file=./exampleData/training.cov --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit --model-file=./model 

Using `genotype`, `phenotype` , `GWAS`, and `additional` data, as well as `Heritability estimate`, run:

    genoml-train --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno  --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit --herit=0.2 --model-file=./model 

Using `genotype`, `phenotype` , `GWAS`, `covariance`, and `additional` data, as well as `Heritability estimate`:

    genoml-train --geno-prefix=./exampleData/training --pheno-file=./exampleData/example.pheno --cov-file=./exampleData/training.cov --gwas-file=./exampleData/example_GWAS.txt --addit-file=./exampleData/training.addit --herit=0.5 --model-file=./model

### Using the machine learning model for inference:
To perform external `model-validate` only when `genotype` and `phenotype` data present:

    genoml-inference --model-file=./model --valid-geno-prefix=./exampleData/validation --valid-pheno-file=./exampleData/validation.pheno

     
## Report issues 
Please report any issue or suggestions on the [GenoML issues page](https://github.com/GenoML/genoml/issues).
