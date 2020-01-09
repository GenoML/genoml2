# -*- coding: utf-8 -*-
"""
# Intro
 - **Project:** GenoML
 - **Author(s):** Mary B. Makarious, Mike A. Nalls, Juan Botia, Hampton Leonard
 - **Date Notebook Started:** 19.09.2019
    - **Quick Description:** Notebook relating to experimental implementation of networkx for community generation.

---
### Quick Description: 
**Problem:** Many software have been used for building networks, within a biological context and in social network and sales analysis. Lets try to do something outside of simple networks.

**Solution:** Lets try to use networkx prioritizing adding Louvain community detection through its widespread application in business https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008. Eventualy we will add decomposition tree based analyses as suggested by https://www.nature.com/articles/s41467-018-03424-4. 

### Motivation/Background:
Data suggests we can do better than very simple coexpression networks also more flexible for SNPs and clinical in nodes?  

### Concerns/Impact on Related Code: 
- Depends on switch to detect discrete or continuous workflows.

### Thoughts for Future Development of Code in this Notebook: 
- Networks could be helpful in partitioning samples and unsupervised learning.

# Imports
"""

import argparse
import sys
import pandas as pd
import numpy as np

"""
# Command args
"""

parser = argparse.ArgumentParser(description='Arguments for training a discrete model')    
parser.add_argument('--infile', type=str, default='plink_glm_file', help='Your results file needing formatting. Default = plink_glm_file.')
parser.add_argument('--outfile', type=str, default='plink_glm_reformatted', help='Your output file. Default = plink_glm_reformatted.')
parser.add_argument('--B-or-C', type=str, default='B', help='B = binary outcome, C = contimuous outcome. Default = B.')
parser.add_argument('--minimal-output', type=str, default='NOPE', help='YEP or NOPE, output only minimal results set to save space. Default = False.')


args = parser.parse_args()

print("")
print("Here is some basic info on the command you are about to run.")
print("Python version info...")
print(sys.version)
print("CLI argument info...")
print("Working with tab-delimited input data", args.infile, "from previous GWAS analyses.")
print("Output file prefix is", args.outfile, "containing your reformatted data in tab delimited format.")
print("Is your GWAS outcome B (binary) or C (continuous)?", args.B_or_C)
print("Only output a minimal set of columns in the results?", args.minimal_output)


print("")

infile = args.infile
outfile = args.outfile
B_or_C = args.B_or_C
minimal_output = args.minimal_output

"""
# Read in your data

This the generic PLINK2 output generated using one of the following commands for binary then continuous outcomes.

module load plink/2.0-dev-20191128

for chnum in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22};
  do
plink2 --pfile /data/CARD/UKBIOBANK/FILTER_IMPUTED_DATA/chr$chnum.UKBB.EU.filtered \
--pheno-name PHENO_PLINK --pheno covariates/FINAL_cov_UKB_PD_cases_control_over60.txt \
--covar covariates/FINAL_cov_UKB_PD_cases_control_over60.txt --memory 235000 \
--glm hide-covar firth-fallback cols=+a1freq,+a1freqcc,+a1count,+totallele,+a1countcc,+totallelecc,+err \
--out RESULTS/UKB_case_control_chr$chnum --covar-name AGE,SEX,TOWNSEND,PC1,PC2,PC3,PC4,PC5 --covar-variance-standardize
done

or

plink2 --pfile /data/CARD/UKBIOBANK/FILTER_IMPUTED_DATA/chr22.UKBB.EU.filtered \
--pheno-name AGE --pheno covariates/FINAL_cov_UKB_Proxy_cases_control_over60.txt \
--covar covariates/FINAL_cov_UKB_Proxy_cases_control_over60.txt --memory 235000 \
--glm hide-covar firth-fallback cols=+a1freq,+a1freqcc,+a1count,+totallele,+a1countcc,+totallelecc,+err \
--out MIKE_continues --covar-name TOWNSEND,PC1,PC2,PC3,PC4,PC5 --covar-variance-standardize

Just as an additional note, this doesn't scale the contimuous outcome to Z, so be careful with unit standardization across studies with continuous traits.
"""

# For testing
# infile = 'binary.txt'
# outfile = 'test_B_false.tab'
# B_or_C = 'B'
# minimal_output = False

df = pd.read_csv(infile, engine = 'c', sep = '\t')

df.rename(columns = {'#CHROM':'CHROM'}, inplace = True)

print("")
print("Your data looks like this (showing the first few lines of the left-most and right-most columns) ...")
print(df.describe())
print("")
print("Now lets get to the data processing ...")

"""
# Define functions to reformat
"""
def make_log_OR(o):
  b_adjusted = np.log(o)
  return b_adjusted

def reformat_binary_GWAS(x):
    global df_edited
    df_editing = df
    df_editing['markerID'] = df_editing['CHROM'].astype(str) + ":" + df_editing['POS'].astype(str) + ":" + df_editing['REF'].astype(str) + ":" + df_editing['ALT'].astype(str)
    df_editing['effectAllele'] = df_editing['A1']
    df_editing['alternateAllele'] = df_editing['ALT'].where(df_editing['A1'] == df_editing['REF'], df_editing['REF'])
    df_editing['beta']= np.vectorize(make_log_OR)(df_editing['OR'])
    df_editing.rename(columns={'LOG(OR)_SE':'se', 'P':'P', 'A1_FREQ':'effectAlleleFreq', 'OBS_CT':'N', 'ID':'rsID', 'A1_CASE_FREQ':'effectAlleleFreq_cases', 'A1_CTRL_FREQ':'effectAlleleFreq_controls', 'FIRTH?':'firthUsed', 'ERRCODE':'error'}, inplace=True)
    df_edited = df_editing[['markerID','effectAllele','alternateAllele','beta','se','P','effectAlleleFreq','N','rsID','effectAlleleFreq_cases','effectAlleleFreq_controls', 'OR', 'firthUsed','error']]
    del df_editing
    return df_edited

def reformat_continuous_GWAS(x):
    global df_edited
    df_editing = df
    df_editing['markerID'] = df_editing['CHROM'].astype(str) + ":" + df_editing['POS'].astype(str) + ":" + df_editing['REF'].astype(str) + ":" + df_editing['ALT'].astype(str)
    df_editing['effectAllele'] = df_editing['A1']
    df_editing['alternateAllele'] = df_editing['ALT'].where(df_editing['A1'] == df_editing['REF'], df_editing['REF'])
    df_editing.rename(columns={'BETA':'beta', 'SE':'se', 'P':'P', 'A1_FREQ':'effectAlleleFreq', 'OBS_CT':'N', 'ID':'rsID', 'ERRCODE':'error'}, inplace=True)
    df_edited = df_editing[['markerID','effectAllele','alternateAllele','beta','se','P','effectAlleleFreq','N','rsID','error']]
    del df_editing
    return df_edited

if (B_or_C == "B"):
    reformat_binary_GWAS(df)

if (B_or_C == "C"):
    reformat_continuous_GWAS(df)

if (minimal_output == "YEP"):
    df_out = df_edited[['markerID','effectAllele','alternateAllele','beta','se','P','effectAlleleFreq','N']]
    del df_edited

if (minimal_output == "NOPE"):
    df_out = df_edited
    del df_edited

df_out.to_csv(outfile, index=False, sep = '\t')

print("")
print("Here is a little preview of your output (showing the first few lines of the left-most and right-most columns) ...")
print(df_out.describe())
print("")
print("Thanks for using GWAS utilities from GenoML.")

"""
# Quick test
"""
# python reformat_plink2_results.py --infile continuous.txt --outfile test_C_NOPE.tab --B-or-C C --minimal-output NOPE
# python reformat_plink2_results.py --infile continuous.txt --outfile test_C_YEP.tab --B-or-C C --minimal-output YEP
# python reformat_plink2_results.py --infile binary.txt --outfile test_B_NOPE.tab --B-or-C B --minimal-output NOPE
# python reformat_plink2_results.py --infile binary.txt --outfile test_B_YEP.tab --B-or-C B --minimal-output YEP