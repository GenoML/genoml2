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

# Import the necessary packages
from genoml.preprocessing.harmonizing import Harmonize
from genoml.preprocessing.munging import Munge


### TODO: Add notice that using b-file inputs might cause weird results for any variants where Alt allele == Major allele
### TODO: Standard format for input files, csv vs tsv?
### TODO: Keep as is and only handle specific column headers, or let user define which columns correspond to which features as with PRSice?
### TODO: Look into recoding genotype data differently -- like plink additive vs hethom?
def munge(
    prefix, impute_type, geno_path, pheno_path, addit_path, geno_test_path, pheno_test_path, 
    addit_test_path, skip_prune, r2, n_est, gwas_paths, p_gwas, vif_thresh, vif_iter, umap_reduce, 
    adjust_data, adjust_normalize, target_features, confounders, confounders_test, data_type,
):
    munger = Munge(
        prefix, impute_type, geno_path, pheno_path, addit_path, geno_test_path, pheno_test_path, 
        addit_test_path, skip_prune, r2, n_est, gwas_paths, p_gwas, vif_thresh, vif_iter, umap_reduce, 
        adjust_data, adjust_normalize, target_features, confounders, confounders_test, data_type,
    )
    munger.create_merged_datasets()
    munger.filter_shared_cols()
    munger.apply_adjustments()
    munger.feature_selection()
    munger.filter_shared_cols()
    munger.save_data()


### TODO: For --force-impute, show what % features are being imputed if they do it.
### TODO: If they are imputing something in, it can't be a feature they correct for with covariates.
def harmonize(
    prefix, geno_harmonize_path, pheno_harmonize_path, addit_harmonize_path, confounders, force_impute, data_type,
):
    harmonizer = Harmonize(
        prefix, geno_harmonize_path, pheno_harmonize_path, addit_harmonize_path, confounders, force_impute, data_type,
    )
    harmonizer.create_merged_datasets()
    harmonizer.filter_shared_cols()
    harmonizer.impute_missing_cols()
    harmonizer.apply_adjustments()
    harmonizer.save_data()
