# Copyright 2020 The GenoML Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import sys

import genoml.dependencies
from genoml import preprocessing


def main(prefix, impute, geno, skip_prune, r2_cutoff, pheno, addit, feature_selection, gwas, p, vif, iter, ref_cols_harmonize, umap_reduce, adjust_data, adjust_normalize, target_features, confounders, data_type):
    genoml.dependencies.check_dependencies()

    run_prefix = prefix
    impute_type = impute
    geno_path = geno
    prune_choice = skip_prune
    pheno_path = pheno
    addit_path = addit
    n_est = feature_selection
    gwas_path = gwas
    p_gwas = p
    r2_cutoff = r2_cutoff
    vif_thresh = vif
    vif_iter = iter
    refColsHarmonize = ref_cols_harmonize
    umap_reduce = umap_reduce
    adjust_data = adjust_data
    adjust_normalize = adjust_normalize
    target_features = target_features
    confounders = confounders

    # Print configurations
    print("")
    print("Here is some basic info on the command you are about to run.")
    print("Python version info...")
    print(sys.version)
    print("CLI argument info...")
    print(
        f"The output prefix for this run is {run_prefix} and will be appended to later runs of GenoML.")
    print(f"Working with genotype data? {geno_path}")
    print(f"Do you want GenoML to prune your SNPs for you? {prune_choice}")
    print(f"The pruning threshold you've chosen is {r2_cutoff}")
    print(f"Working with additional predictors? {addit_path}")
    print(f"Where is your phenotype file? {pheno_path}")
    print(f"Any use for an external set of GWAS summary stats? {gwas_path}")
    print(
        f"If you plan on using external GWAS summary stats for SNP filtering, we'll only keep SNPs at what P value? {p_gwas}")
    print(f"How strong is your VIF filter? {vif_thresh}")
    print(f"How many iterations of VIF filtering are you doing? {vif_iter}")
    print(
        f"The imputation method you picked is using the column {impute_type} to fill in any remaining NAs.")
    print(f"Will you be adjusting additional features using UMAP dimensionality reduction? {umap_reduce}")
    print(
        "Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: os, sys, argparse, numpy, pandas, joblib, math and time. We also use PLINK v1.9 from https://www.cog-genomics.org/plink/1.9/.")
    print("")

    # Run the munging script in genoml.preprocessing
    munger = preprocessing.munging(pheno_path=pheno_path, run_prefix=run_prefix, impute_type=impute_type, skip_prune=prune_choice,
                     p_gwas=p_gwas, addit_path=addit_path, gwas_path=gwas_path, geno_path=geno_path, refColsHarmonize=refColsHarmonize, r2_cutoff=r2_cutoff)

    # Process the PLINK inputs (for pruning)
    df = munger.plink_inputs()

    # Run the UMAP dimension reduction/ adjuster 
    if (adjust_data == "yes" or umap_reduce == "yes"):
        adjuster = preprocessing.adjuster(run_prefix, df, target_features, confounders, adjust_data, adjust_normalize, umap_reduce)
        reduced_df = adjuster.umap_reducer()
        if (adjust_data == "yes"): 
            print(f"\n You have chosen to adjust your data! \n")
            if (adjust_normalize == "yes"):
                print(f"\n You have also chosen to normalize your adjusted data \n")
            else:
                print(f"\n You have also chosen NOT to normalize your adjusted data \n")
        df = adjuster.normalize(reduced_df)

    # Run the feature selection using extraTrees
    if n_est > 0:
        featureSelection_df = preprocessing.featureselection(run_prefix, df, data_type, n_est)
        df = featureSelection_df.rank()
        featureSelection_df.export_data()

    # Run the VIF calculation
    if vif_iter > 0:
        vif_calc = preprocessing.vif(vif_iter, vif_thresh, df, 100, run_prefix)
        vif_calc.vif_calculations()

    # Thank the user
    print("Thank you for munging with GenoML!")
