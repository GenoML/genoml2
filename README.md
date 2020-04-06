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
- Removing preproccesing.utils to the main munging class to limit the number of input arguments
  - Overhaul of munging.py
- Adding extraTrees feature selection for discrete and continuous data
- Splitting up munging so it only takes in --pheno and --addit files (no --geno necessary; right now it's programmed in such a way that they can't be separated)
- Dynamically load in PLINK and check dependencies   

### CURRENTLY WORKING ON
- Discrete Supervised Test
- Continuous Supervised Test
- Adding argparse option and switch to training.py (d+c) to specify which metric to maximize by (`AUC_Percent`, `Balanced_Accuracy_Percent`, `Sensitivity`, or `Specificity`)
- Modifying tuning.py to specify which metric to maximize by `AUC_Percent` or `Balanced_Accuracy_Percent`
- Adding minor fixes 
- Adding appropriate function descriptions


### NOT DONE
- Unsupervised?
- UKBB?
- Add experimental folder?
  
### NEED TO IMPLEMENT 
- feature_rank only works... some of the time? why? 
- Verbose option for Python package mimicking the CLI interface 
- Default error message for when users do not put proper inputs for GenoMLMunging
- Add error catching 
- Clean up munging 
- Add default parameters for the classes (to be run as a Python package and not CLI)

### CURRENT PAIN POINTS AND LIMITATIONS 
- Right after the .raw file post-pruning is generated after `plink_inputs()`. To reconstruct those files to impute, z-score, etc to them is a big bottleneck
- Using RFE for feature ranking at the end of tune, but now when we use extraTrees an approx list of features get spit out a lot faster 
- Larger biobank scale, merging dataframes in general has been quite painful 

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