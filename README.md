# GenoML-core
GenoML is an Automated Machine Learning (AutoML) for Genomic. This is the core package of GenoML. 
this repo is under development, please report any issues (bug, performance, documentation) on the [GenoML issues page](https://github.com/GenoML/genoml/issues). 

Here are some quick "get started" exmaples, please checkout the additional options and details in the [Usage][] and [CLI][].
In general, use linux or mac with python > 3.5 for best results.

## Install 
Run:
~~~~
pip install genoml
~~~~

## Train the ML model 
You can use the IPDGC (International Parkinson's Disease Genomics Consortium) test data. This data is a simulation of the genetic and clinical data used for Parkinson's diagnosis in previous publications. You can find it at [IPDGC example data][].

Download and unzip data:
~~~
wget https://github.com/ipdgc/GenoML-Brief-Intro/raw/master/exampleData.zip
unzip exampleData.zip 
~~~

To train, run:
~~~~
genoml-train --geno-prefix=./exampleData/training --pheno-file=./exampleData/training.pheno --model-dir=./exampleModel
~~~~

Final tuned model and performance metrics are stored in the ```--model-dir``` directory. 

## Using the trained ML model for inference
~~~~
genoml-inference --model-dir=./exampleModel --valid-dir=./exampleData --valid-geno-prefix=./exampleData/validation --valid-pheno-file=./exampleData/validation.pheno
~~~~

Valdiation results and model performance metrics are stored in the ```--valid-dir``` directory. 

> For debugging purposes, include the ```-v``` or ```-vvv``` flags at the end of a command.

[usage]: https://genoml.github.io/docs/usage
[CLI]: https://genoml.github.io/docs/cli
[IPDGC example data]: https://github.com/ipdgc/GenoML-Brief-Intro/raw/master/exampleData.zip

     
## Report issues 
Please report any issue or suggestions on the [GenoML issues page](https://github.com/GenoML/genoml/issues).
