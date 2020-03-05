## README

Look at `GettingStarted.sh` to see how a virtual environment is set up 

### DONE
- Munging
  - Added --gwas and --p flags
- Discrete
  - Supervised
    - Train
    - Tune
- Continuous 
  - Supervised 
    - Train
    - Tune
- Update file structure 

### CURRENTLY WORKING ON
- Adding appropriate function descriptions
- Removing preproccesing.utils to the main munging class to limit the number of input arguments

### NOT DONE
- Discrete Supervised Test
- Continuous Supervised Test
- Unsupervised?
- UKBB?
- Add experimental folder?
  
### NEED TO IMPLEMENT 
- Verbose option for Python package mimicking the CLI interface 
- Default error message for when users do not put proper inputs for GenoMLMunging
- Add error catching 
- Clean up munging 
- Add default parameters for the classes (to be run as a Python package and not CLI)


### USE CASE REQUIREMENTS THAT STILL NEED TO BE ADDRESSED
- [Train+Tune] Ability to set seeds for reproducibility in tune and train 
- [Pruning] (UKBiobank need) Add sex check to remove samples in wrong ICD-10 codes (and then remove sex as covariate)
- [Pruning] (UKBiobank need) Ability to specify phenotype name (list of phenotypes) as an argument to pull from a large file of genotypes and phenotypes
- [Pruning] (UKBiobank need) Optimizing merge – taking too much time/memory
- [Pruning] Output intermediate file – proper, clear documentation (?)
- [Training] K-fold Cross validation for training – setting K
- [Training] Supervised prediction multi-class
- [Docs] Updating the website, tutorials, and sample use-cases

### NOTES
- Munging double scalars issue? (Is this just a warning and it's actually dropping the SNPs or no?)
  - This happens when you re-munge on already munged data. 