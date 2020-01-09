# -*- coding: utf-8 -*-
"""get_the_SEs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cn5yCIzU5Gtc7Cn4b3py7I6BaKtJj0m0
"""

# Imports

import argparse
import sys
import sklearn
import h5py
import pandas as pd
import numpy as np
import time
import math
import scipy
import scipy.stats

# Set options

run_prefix = "test_Asian_GWAS.gz"
beta_name = 'BETA'
p_name = 'P'
# freq_name = 'Freq' no allele freq either, WTF?
outfile = "test_Asian_GWAS_with_SE.csv"

# Read data

df = pd.read_csv(run_prefix, engine = "c", sep = "\t", compression='gzip')

print("Top few lines fo your file...")
print(df.head())

print("Quick summary of your input data...")
print(df.describe())

# Rename columns

df.rename(columns={beta_name:'b', p_name:'p'}, inplace=True)

# Derive SE

print("Now deriving the adjusted SE from the adjusted beta.")

def fx_beta_p_to_SE(b, p):
  if (p < 1E-15):
    p = 1E-15
    z_score = scipy.stats.norm.ppf(1 - (p/2))
  if (p >= 1E-15):
    z_score = scipy.stats.norm.ppf(1 - (p/2))
  se_adj = abs(b/z_score)
  return se_adj

df['se'] = np.vectorize(fx_beta_p_to_SE)(df['b'], df['p'])

print("Quick sanity check for P derived from adjusted stats. NOTE: \"keep it real\" regarding floats, so we are capping estimates at 15 decimal places, so there may be some conservative estiamtes but after p < 1E-15 does that really matter?")

# Sanity check

def fx_test_P(b, se):
  z_derived = b/se
  p_derived = scipy.special.ndtr(-1 * abs(z_derived)) * 2
  return p_derived

df['p_derived'] = np.vectorize(fx_test_P)(df['b'], df['se'])

print("After the run looks like this...")

print("Top few lines fo your file...")
print(df.head())

print("Quick summary of your input data...")
print(df.describe())

# Export

df.to_csv(outfile, index=False)