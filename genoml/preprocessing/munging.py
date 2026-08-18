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
import genoml.preprocessing.utils as preprocessing_utils
import numpy as np
import pandas as pd
import pickle
import sys
from genoml import utils, dependencies
from genoml.preprocessing import adjuster, featureselection
from sklearn.model_selection import StratifiedKFold
from time import time


class Munge:
    @utils.DescriptionLoader.function_description("info", cmd="Munging")
    def __init__(
        self, prefix, impute_type, geno_path, pheno_path, addit_path, geno_test_path, pheno_test_path, 
        addit_test_path, skip_prune, r2, n_trees, gwas_paths, p_gwas, vif_thresh, vif_iter, pearson_threshold, 
        ols_threshold, umap_reduce, adjust_data, adjust_normalize, target_features, confounders, 
        confounders_test, n_outer_cv, random_state, data_type,
    ):
        self.start = time()
        utils.DescriptionLoader.print(
            "munging/info",
            python_version=sys.version,
            prefix = prefix, 
            geno_path = geno_path, 
            addit_path = addit_path, 
            pheno_path = pheno_path, 
            geno_test_path = geno_test_path, 
            addit_test_path = addit_test_path, 
            pheno_test_path = pheno_test_path, 
            skip_prune = skip_prune, 
            r2 = r2, 
            gwas_paths = ', '.join(gwas_paths) if len(gwas_paths) > 0 else '', 
            p_gwas = p_gwas, 
            vif_thresh = vif_thresh, 
            vif_iter = vif_iter, 
            pearson_threshold = pearson_threshold,
            ols_threshold = ols_threshold,
            impute_type = impute_type, 
            umap_reduce = umap_reduce,
            n_outer_cv = n_outer_cv,
            random_state = random_state,
        )

        dependencies.check_dependencies()

        # Initializing some variables
        self.plink_exec = dependencies.check_plink()
        self.prefix = utils.create_results_dir(prefix, "Munge")
        self.impute_type = impute_type
        self.geno_path = geno_path
        self.pheno_path = pheno_path
        self.addit_path = addit_path
        self.geno_test_path = geno_test_path
        self.pheno_test_path = pheno_test_path
        self.addit_test_path = addit_test_path
        self.skip_prune = skip_prune
        self.r2 = r2
        self.n_trees = n_trees
        self.gwas_paths = gwas_paths
        self.p_gwas = p_gwas
        self.vif_iter = vif_iter
        self.vif_thresh = vif_thresh
        self._pearson_threshold = pearson_threshold
        self._ols_threshold = ols_threshold
        self.umap_reduce = umap_reduce
        self.adjust_data = adjust_data
        self.adjust_normalize = adjust_normalize
        self.target_features = target_features 
        self.confounders = confounders 
        self.confounders_test = confounders_test
        self.data_type = data_type
        self.is_munging_test_data = self.pheno_test_path is not None
        self.n_outer_cv = n_outer_cv if not self.is_munging_test_data else 0
        self._random_state = random_state

        self.df_merged = None
        self.df_merged_test = None
        self.df_merged_cv_dict = None
        self._addit_col_names = None
        self._addit_means = None
        self._addit_stdevs = None
        self.features_list = None


    def create_merged_datasets(self, n_outer_cv=None):
        """ Merge phenotype, genotype, and additional data. """
        self.df_merged = preprocessing_utils.read_pheno_file(self.pheno_path, self.data_type)
        self.df_merged, self._addit_means, self._addit_stdevs = preprocessing_utils.merge_addit_data(self.df_merged, self.addit_path, self.impute_type)
        self._addit_col_names = self.df_merged.drop(columns=["ID", "PHENO"]).columns.values
        self.df_merged = preprocessing_utils.merge_geno_data(
            self.df_merged, self.geno_path, self.pheno_path, self.impute_type, self.prefix, self.gwas_paths, self.p_gwas, self.skip_prune, self.plink_exec, self.r2,
        )

        if self.is_munging_test_data:
            self.df_merged_test = preprocessing_utils.read_pheno_file(self.pheno_test_path, self.data_type)
            self.df_merged_test, _, _ = preprocessing_utils.merge_addit_data(self.df_merged_test, self.addit_test_path, self.impute_type)
            self.df_merged_test = preprocessing_utils.merge_geno_data(
                self.df_merged_test, self.geno_test_path, self.pheno_test_path, self.impute_type, self.prefix, self.gwas_paths, self.p_gwas, self.skip_prune, self.plink_exec, self.r2,
            )
        
        if n_outer_cv is not None:
            skf = StratifiedKFold(
                n_splits=self.n_outer_cv,
                shuffle=True,
                random_state=self._random_state,
            )
            self.df_merged_cv_dict = {}
            if self.data_type == "c":
                self.df_merged["PHENO_bin"] = pd.qcut(
                    self.df_merged["PHENO"],
                    q = max(2, min(10, len(self.df_merged) // self.n_outer_cv)),
                    labels=False,
                    duplicates="drop",
                )
                self.df_merged = self.df_merged.dropna(subset=["PHENO_bin"])
                for fold, (train_idx, test_idx) in enumerate(skf.split(self.df_merged, self.df_merged["PHENO_bin"])):
                    self.df_merged_cv_dict[fold+1] = {
                        "train": self.df_merged.iloc[train_idx].drop(columns="PHENO_bin"),
                        "test": self.df_merged.iloc[test_idx].drop(columns="PHENO_bin"),
                    }
            else:
                for fold, (train_idx, test_idx) in enumerate(skf.split(self.df_merged, self.df_merged["PHENO"])):
                    self.df_merged_cv_dict[fold+1] = {
                        "train": self.df_merged.iloc[train_idx],
                        "test": self.df_merged.iloc[test_idx],
                    }


    def filter_shared_cols(self):
        """ Initial filtering to keep only shared columns between train and test data. """
        if self.df_merged_test is not None:
            self.df_merged, self.df_merged_test = preprocessing_utils.filter_common_cols(self.df_merged, self.df_merged_test)


    def apply_adjustments(self, fold=None):
        """ Adjust datasets by covariates. """
        if self.adjust_data:
            # Adjust train data
            train_adjuster = adjuster.Adjuster(
                self.prefix,
                self.df_merged,
                self.target_features,
                self.confounders,
                self.adjust_normalize,
                self.umap_reduce,
                self._random_state,
            )
            self.features_list = train_adjuster.targets
            umap_reducer = train_adjuster.umap_reducer("train")
            self.df_merged, adjustment_models = train_adjuster.adjust_confounders(fold=fold)

            # Adjust test data if provided
            if self.df_merged_test is not None:
                test_adjuster = adjuster.Adjuster(
                    self.prefix,
                    self.df_merged_test,
                    self.target_features,
                    self.confounders_test,
                    self.adjust_normalize,
                    self.umap_reduce,
                    self._random_state,
                )
                _ = test_adjuster.umap_reducer("test", reducer=umap_reducer)
                self.df_merged_test, _ = test_adjuster.adjust_confounders(adjustment_models=adjustment_models, fold=fold)

    
    def feature_selection(self, fold=None):
        """ extraTrees and VIF for to prune unnecessary features. """
        # Run the feature selection using extraTrees
        feature_selector = featureselection.FeatureSelection(
            self.prefix, 
            self.df_merged, 
            self.data_type, 
            self.n_trees, 
            self.vif_iter, 
            self.vif_thresh, 
            100,
            self._pearson_threshold, 
            self._ols_threshold, 
            self._random_state,
            fold=fold,
        )
        self.df_merged = feature_selector.select_features()


    def save_data(self, fold=None):
        """ Save munged training and testing datasets """
        train_h5_path = f"train_dataset{f'_fold{fold}' if fold is not None else ''}.h5"
        self.df_merged.to_hdf(self.prefix.joinpath(train_h5_path), key='dataForML')
        if self.df_merged_test is not None:
            test_h5_path = f"test_dataset{f'_fold{fold}' if fold is not None else ''}.h5"
            self.df_merged_test.to_hdf(self.prefix.joinpath(test_h5_path), key='dataForML')

        # Also save parameters for harmonization
        cols = self.df_merged.columns
        numeric_cols = self.df_merged.select_dtypes(include=[np.number]).columns.values
        impute_vals = self.df_merged[numeric_cols].mean()
        if self.impute_type != "mean":
            impute_vals = self.df_merged[numeric_cols].median()
        params_for_harmonize = {
            "adjust_normalize" : self.adjust_normalize,
            "impute_type" : self.impute_type,
            "vif_thresh" : self.vif_thresh,
            "vif_iter" : self.vif_iter,
            "target_features" : self.features_list, 
            "cols" : cols,
            "impute_vals" : impute_vals,
            "mean_vals" : self._addit_means,
            "std_vals" : self._addit_stdevs,
        }
        params_path = f"params{f'_fold{fold}' if fold is not None else ''}.pkl"
        with open(self.prefix.joinpath(params_path), "wb") as file:
            pickle.dump(params_for_harmonize, file)

        # Thank the user
        if fold is None or fold == self.n_outer_cv:
            print(f"Your fully munged training data can be found here: {self.prefix.joinpath(train_h5_path)}")
            if self.pheno_test_path is not None:
                print(f"Your fully munged testing data can be found here: {self.prefix.joinpath(test_h5_path)}")
            print("Thank you for munging with GenoML!")
            print(f"Munging took {time() - self.start} seconds")
