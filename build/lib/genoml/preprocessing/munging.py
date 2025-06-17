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
import pickle
import sys
from genoml import utils, dependencies
from genoml.preprocessing import adjuster, vif, featureselection
from time import time


class Munge:
    @utils.DescriptionLoader.function_description("info", cmd="Munging")
    def __init__(
        self, prefix, impute_type, geno_path, pheno_path, addit_path, geno_test_path, pheno_test_path, 
        addit_test_path, skip_prune, r2, n_est, gwas_paths, p_gwas, vif_thresh, vif_iter, umap_reduce, 
        adjust_data, adjust_normalize, target_features, confounders, confounders_test, data_type,
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
            impute_type = impute_type, 
            umap_reduce = umap_reduce,
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
        self.n_est = n_est
        self.gwas_paths = gwas_paths
        self.p_gwas = p_gwas
        self.vif_thresh = vif_thresh
        self.vif_iter = vif_iter
        self.umap_reduce = umap_reduce
        self.adjust_data = adjust_data
        self.adjust_normalize = adjust_normalize
        self.target_features = target_features 
        self.confounders = confounders 
        self.confounders_test = confounders_test
        self.data_type = data_type
        self.is_munging_test_data = self.pheno_test_path is not None

        self.df_merged = None
        self.features_list = None


    def create_merged_datasets(self):
        """ Merge phenotype, genotype, and additional data. """
        self.df_merged = preprocessing_utils.read_pheno_file(self.pheno_path, self.data_type)
        self.df_merged = preprocessing_utils.merge_addit_data(self.df_merged, self.addit_path, self.impute_type)
        self.df_merged = preprocessing_utils.merge_geno_data(
            self.df_merged, self.geno_path, self.pheno_path, self.impute_type, self.prefix, self.gwas_paths, self.p_gwas, self.skip_prune, self.plink_exec, self.r2,
        )

        if self.is_munging_test_data:
            self.df_merged_test = preprocessing_utils.read_pheno_file(self.pheno_test_path, self.data_type)
            self.df_merged_test = preprocessing_utils.merge_addit_data(self.df_merged_test, self.addit_test_path, self.impute_type)
            self.df_merged_test = preprocessing_utils.merge_geno_data(
                self.df_merged_test, self.geno_test_path, self.pheno_test_path, self.impute_type, self.prefix, self.gwas_paths, self.p_gwas, self.skip_prune, self.plink_exec, self.r2,
            )


    def filter_shared_cols(self):
        """ Initial filtering to keep only shared columns between train and test data. """
        if self.is_munging_test_data:
            self.df_merged, self.df_merged_test = preprocessing_utils.filter_common_cols(self.df_merged, self.df_merged_test)


    def apply_adjustments(self):
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
            )
            self.features_list = train_adjuster.targets
            umap_reducer = train_adjuster.umap_reducer("train")
            self.df_merged, adjustment_models = train_adjuster.adjust_confounders()

            # Adjust test data if provided
            if self.is_munging_test_data:
                test_adjuster = adjuster.Adjuster(
                    self.prefix,
                    self.df_merged_test,
                    self.target_features,
                    self.confounders_test,
                    self.adjust_normalize,
                    self.umap_reduce,
                )
                _ = test_adjuster.umap_reducer("test", reducer=umap_reducer)
                self.df_merged_test, _ = test_adjuster.adjust_confounders(adjustment_models=adjustment_models)

    
    def feature_selection(self):
        """ extraTrees and VIF for to prune unnecessary features. """
        # Run the feature selection using extraTrees
        if self.n_est > 0:
            feature_selector = featureselection.FeatureSelection(self.prefix, self.df_merged, self.data_type, self.n_est)
            self.df_merged = feature_selector.rank()
            feature_selector.export_data()

        ### TODO: Check that VIF does what we want it to do.
        # Run the VIF calculation
        if self.vif_iter > 0:
            munge_vif = vif.VIF(self.vif_iter, self.vif_thresh, self.df_merged, 100, self.prefix)
            self.df_merged = munge_vif.vif_calculations()


    def save_data(self):
        """ Save munged training and testing datasets """
        outfile_h5 = self.prefix.joinpath("train_dataset.h5")
        self.df_merged.to_hdf(outfile_h5, key='dataForML')
        if self.is_munging_test_data:
            outfile_h5_test = self.prefix.joinpath("test_dataset.h5")
            self.df_merged_test.to_hdf(outfile_h5_test, key='dataForML')

        # Also save parameters for harmonization
        cols = self.df_merged.columns
        if self.impute_type == "mean":
            avg_vals = self.df_merged.select_dtypes(include=[np.number]).mean()
        else:
            avg_vals = self.df_merged.select_dtypes(include=[np.number]).median()
        params_for_harmonize = {
            "adjust_normalize" : self.adjust_normalize,
            "impute_type" : self.impute_type,
            "vif_thresh" : self.vif_thresh,
            "vif_iter" : self.vif_iter,
            "target_features" : self.features_list, 
            "cols" : cols,
            "avg_vals" : avg_vals,
        }
        with open(self.prefix.joinpath("params.pkl"), "wb") as file:
            pickle.dump(params_for_harmonize, file)

        # Thank the user
        print(f"Your fully munged training data can be found here: {outfile_h5}")
        if self.pheno_test_path is not None:
            print(f"Your fully munged testing data can be found here: {outfile_h5_test}")
        print("Thank you for munging with GenoML!")
        print(f"Munging took {time() - self.start} seconds")

