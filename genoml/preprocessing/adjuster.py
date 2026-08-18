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
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import statsmodels.formula.api as smf
import umap.umap_ as umap
from joblib import dump


class Adjuster:
    def __init__(self, prefix, df_merged, target_features, confounders, adjust_normalize, umap_reduce, random_state, force_impute=False):
        self.prefix = prefix
        self.umap_reduce = umap_reduce
        self.confounders = confounders
        self.normalize = adjust_normalize
        self.df_merged = df_merged
        self._random_state = random_state

        self.df_confounders = pd.read_csv(self.confounders, engine="python", sep=None, encoding="utf-8-sig")
        self.df_confounders["ID"] = self.df_confounders["ID"].astype(str)
        self.df_confounders = self.df_confounders.merge(self.df_merged[["ID"]], on="ID")
        self.df_confounders = self.df_confounders.fillna(
            self.df_confounders.mean(numeric_only=True)
        )

        if type(target_features) == list:
            self.targets = target_features
        else:
            with open(target_features, "r") as f:
                self.targets = list(set([str(line.strip()) for line in f if str(line.strip()) in self.df_merged.columns]))

        # Remove any targets that are being imputed
        if force_impute:
            with open(self.prefix.joinpath("missing_cols.txt"), "r") as file:
                imputed_features = [line.split("\t", 1)[0].strip() for line in file if line.strip()]
            self.targets = list(set(self.targets) - set(imputed_features))

        self.targets.sort()
        
        confounders_in_targets = [col for col in self.df_confounders.columns if col in self.targets]
        if len(confounders_in_targets) > 0:
            self.df_confounders.drop(columns=confounders_in_targets, errors="ignore", inplace=True)

        print(f"\nYou have chosen to adjust your data! \n")
        print(f"You have also chosen{' NOT' if not adjust_normalize else ''} to normalize your adjusted data \n")
        

    def umap_reducer(self, dataset_type, reducer=None):
        if self.umap_reduce:
            ids = self.df_confounders["ID"]
            to_umap = self.df_confounders.drop(columns=["ID"])

            if reducer is None:
                reducer = umap.UMAP(random_state=self._random_state)
                reducer = reducer.fit(to_umap)
            embedding = reducer.transform(to_umap)
            self.df_confounders = pd.DataFrame({
                "ID": ids.values,
                "UMAP_embedding1": embedding[:, 0],
                "UMAP_embedding2": embedding[:, 1]
            })

            fig, ax = plt.subplots(figsize=(12,10))
            plt.scatter(embedding[:,0], embedding[:,1])
            plt.title("Data Reduction to 2 Dimensions by UMAP", fontsize=18)

            plot_out = self.prefix.joinpath(f"umap_plot_{dataset_type}.png")
            plt.savefig(plot_out, dpi=600)
            print(f"The UMAP plot has been exported and can be found here: {plot_out}")
            
            embed_out = self.prefix.joinpath(f"umap_data_reduction_{dataset_type}.txt")
            self.df_confounders.to_csv(embed_out, index=False, sep="\t")
            print(f"The reduced UMAP 2 dimensions per sample file can be found here: {embed_out}")

            if dataset_type == "train":
                algo_out = self.prefix.joinpath("umap_clustering.joblib")
                dump(reducer, algo_out)
                print(f"The UMAP .joblib  file can be found here: {algo_out}")

        return reducer


    def adjust_confounders(self, adjustment_models=None, fold=None):
        confounder_list = [f"Q('{c}')" for c in self.df_confounders.columns[1:]]
        formula_for_confounders = " + ".join(confounder_list)
        
        df_adjustments = self.df_merged.merge(self.df_confounders, how="inner", on="ID", suffixes=["", "_y"])
        fitting_adjustment_models = adjustment_models is None

        if fitting_adjustment_models:
            adjustment_models = {}

        for target in self.targets:
            if fitting_adjustment_models:
                # Fit and save model
                current_formula = f"Q('{target}') ~ {formula_for_confounders}"
                model = smf.ols(formula=current_formula, data=df_adjustments).fit()
                residuals = pd.to_numeric(model.resid)

                adjustment_models[target] = {"model": model}
                if self.normalize:
                    mean = residuals.mean()
                    std = residuals.std()
                    residuals = (residuals - mean) / std
                    adjustment_models[target]["mean"] = mean
                    adjustment_models[target]["std"] = std
                
                adjustment_models_path = f"adjustment_models{f'_fold{fold}' if fold is not None else ''}.pkl"
                with open(self.prefix.joinpath(adjustment_models_path), "wb") as file:
                    pickle.dump(adjustment_models, file)

            else:
                # Apply stored model
                model = adjustment_models[target]["model"]
                predicted = model.predict(df_adjustments)
                residuals = pd.to_numeric(df_adjustments[target] - predicted)

                if self.normalize:
                    mean = adjustment_models[target]["mean"]
                    std = adjustment_models[target]["std"]
                    residuals = (residuals - mean) / std

            df_adjustments[target] = residuals

        df_adjusted = df_adjustments[list(self.df_merged.columns)]
        return df_adjusted, adjustment_models
