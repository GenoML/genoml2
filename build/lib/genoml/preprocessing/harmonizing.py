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
import joblib
import pickle
import sys
from genoml import utils, dependencies
from genoml.preprocessing import adjuster


class Harmonize:
    @utils.DescriptionLoader.function_description("info", cmd="Harmonizing")
    def __init__(self, prefix, geno_path, pheno_path, addit_path, confounders, force_impute, data_type):
        # Create subdirectory where all munging-related files will be saved
        self.prefix = utils.create_results_dir(prefix, "Munge")

        # Load params/models from munging
        self.confounders = confounders
        self.data_type = data_type
        self.force_impute = force_impute
        with open(self.prefix.joinpath("params.pkl"), "rb") as file:
            params = pickle.load(file)
        self.adjust_normalize = params["adjust_normalize"]
        self.impute_type = params["impute_type"]
        self.vif_thresh = params["vif_thresh"]
        self.vif_iter = params["vif_iter"]
        self.target_features = params["target_features"]
        self.cols = params["cols"]
        self.avg_vals = params["avg_vals"]
        self.umap_reducer = joblib.load(self.prefix.joinpath("umap_clustering.joblib")) if self.prefix.joinpath("umap_clustering.joblib").exists() else None

        utils.DescriptionLoader.print(
            "harmonizing/info",
            python_version=sys.version,
            prefix = self.prefix,
            geno_path = geno_path,
            addit_path = addit_path,
            pheno_path = pheno_path,
            vif_thresh = self.vif_thresh,
            vif_iter = self.vif_iter,
            impute_type = self.impute_type,
            force_impute = "" if force_impute else "NOT ",
            umap_reduce = self.umap_reducer is not None,
        )
    
        dependencies.check_dependencies()

        # Initializing some variables
        self.plink_exec = dependencies.check_plink()
        self.geno_path = geno_path
        self.pheno_path = pheno_path
        self.addit_path = addit_path
        if self.prefix.joinpath("adjustment_models.pkl").exists():
            self.adjust_data = True
            with open(self.prefix.joinpath("adjustment_models.pkl"), "rb") as file:
                self.adjustment_models = pickle.load(file)
        else:
            self.adjust_data = False
            self.adjustment_models = None 
        self.df_merged_harmonize = None


    def create_merged_datasets(self):
        """ Merge phenotype, genotype, and additional data. """
        self.df_merged_harmonize = preprocessing_utils.read_pheno_file(self.pheno_path, self.data_type)
        self.df_merged_harmonize = preprocessing_utils.merge_addit_data(self.df_merged_harmonize, self.addit_path, self.impute_type)
        self.df_merged_harmonize = preprocessing_utils.merge_geno_harmonize_data(
            self.df_merged_harmonize, self.geno_path, self.pheno_path, self.impute_type, self.prefix, self.plink_exec)


    def filter_shared_cols(self):
        """ Initial filtering to keep only shared columns between train and harmonizing data. """
        common_cols = self.cols.intersection(self.df_merged_harmonize.columns)
        self.df_merged_harmonize = self.df_merged_harmonize[common_cols]


    def impute_missing_cols(self):
        """ Add missing columns back in by imputing averages from training dataset. """
        missing_cols = self.cols.difference(self.df_merged_harmonize.columns)

        if len(missing_cols) > 0 and not self.force_impute:
            with open(self.prefix.joinpath("missing_cols.txt"), 'w') as file:
                for col in missing_cols:
                    file.write(f'{col}\n')

            val_err_str = f"{len(missing_cols)} of the {len(self.cols)} features that were used to fit the model which are not present in the dataset you're harmonizing! \n"
            val_err_str += "We STRONGLY recommend using the same features that were used to fit your model. \n"
            val_err_str += "However, if you want to impute these values using the average from the training data, please add the --force_impute flag. \n"
            val_err_str += f"A list of missing columns has been written to {self.prefix.joinpath('missing_cols.txt')} \n"
            if len(missing_cols) < 10:
                val_err_str += "These are:\n - "
                val_err_str += "\n - ".join(missing_cols)
            else:
                val_err_str += "These include:\n - "
                val_err_str += "\n - ".join(missing_cols[:5])
                val_err_str += "\n...\n"
                val_err_str += "\n - ".join(missing_cols[-5:])
            raise ValueError(val_err_str)

        elif len(missing_cols) > 0 and self.force_impute:
            with open(self.prefix.joinpath("missing_cols.txt"), 'w') as file:
                for col in missing_cols:
                    self.df_merged_harmonize[col] = self.avg_vals[col]
                    file.write(f'{col}\t{self.avg_vals[col]}\n')

            print_str = f"{len(missing_cols)} of the {len(self.cols)} features that were used to fit the model which are not present in the dataset you're harmonizing! \n"
            print_str = "You have chosen to impute these values using the average from the training data. \n"
            print_str = "However, in the future we STRONGLY recommend using the same features that were used to fit your model. \n"
            print_str = f"A list of missing columns and the values used for imputation has been written to {self.prefix.joinpath('missing_cols.txt')} \n"
            if len(missing_cols) < 10:
                print_str += "These are:\n - "
                print_str += "\n - ".join(missing_cols)
            else:
                print_str += "These include:\n - "
                print_str += "\n - ".join(missing_cols[:5])
                print_str += "\n...\n"
                print_str += "\n - ".join(missing_cols[-5:])
            print(print_str)


    def apply_adjustments(self):
        """ Adjust dataset by covariates using the same model fit on the training set. """
        if self.adjust_data:
            harmonize_adjuster = adjuster.Adjuster(
                self.prefix,
                self.df_merged_harmonize,
                self.target_features,
                self.confounders,
                self.adjust_normalize,
                self.umap_reducer is not None,
            )
            _ = harmonize_adjuster.umap_reducer("harmonize", reducer=self.umap_reducer)
            self.df_merged_harmonize, _ = harmonize_adjuster.adjust_confounders(adjustment_models=self.adjustment_models)


    def save_data(self):
        """ Save harmonized dataset. """
        outfile_h5 = self.prefix.joinpath("test_dataset.h5")
        self.df_merged_harmonize.to_hdf(outfile_h5, key='dataForML')

        # Thank the user
        print(f"Your fully munged harmonizing data can be found here: {outfile_h5}")
        print("Thank you for harmonizing with GenoML!")

