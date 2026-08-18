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

import joblib
import numpy as np
import pandas as pd
import re
import statsmodels.formula.api as smf
from sklearn import ensemble, feature_selection
from statsmodels.stats import outliers_influence
from tqdm import tqdm


class FeatureSelection:
    def __init__(self, prefix, df, data_type, n_est, vif_iter, vif_threshold, chunk_size, pearson_threshold, ols_threshold, random_state, fold=None):
        df.dropna(axis=1, inplace=True)
        self.prefix = prefix
        self.n_est = n_est
        self.data_type = data_type
        self.df = df
        self.y = self.df.PHENO
        self.ids = self.df.ID
        self.x = self.df.drop(columns=['PHENO','ID'])
        self.vif_iter = vif_iter
        self.vif_threshold = vif_threshold
        self.chunk_size = chunk_size
        self._pearson_threshold = pearson_threshold
        self._ols_threshold = ols_threshold
        self._random_state = random_state
        self.fold = fold

        self.df_list = None

    def select_features(self):
        if self.n_est > 0:
            self.extra_trees()
        if self.vif_iter > 0:
            self.vif()
        
        # OLS before Pearson Correlation for computational efficiency
        if self._ols_threshold > 0:
            self.ols_association()
        if self._pearson_threshold > 0:
            self.pearson_correlation()
        return self.df

    def pearson_correlation(self):
        df_features = self.df.copy().drop(columns=["ID", "PHENO"])
        numeric_features = df_features.select_dtypes(include=[np.number])

        arr = numeric_features.values.T.astype(np.float32)
        n = arr.shape[1]
        cols = numeric_features.columns.tolist()
        kept_idx = [0]
        kept_arr = arr[[0]]

        for i in range(1, len(cols)):
            if np.abs((kept_arr @ arr[i]) / n).max() < self._pearson_threshold:
                kept_idx.append(i)
                kept_arr = np.vstack([kept_arr, arr[i]])

        correlation_filtered_features = [cols[i] for i in kept_idx]
        self.df = self.df[["ID", "PHENO"] + correlation_filtered_features]
        print(f"{len(correlation_filtered_features)} features remaining after Pearson's correlation filtering with a threshold of {self._pearson_threshold}")

    def ols_association(self):
        regression_results = []
        feature_columns = self.df.drop(columns=["ID", "PHENO"]).columns.values
        for feature in tqdm(feature_columns, total=len(feature_columns), desc="Running independent regressions", miniters=len(feature_columns) // 100):
            safe_feature_name = re.sub(r'[^0-9a-zA-Z_]', '_', feature)
            
            df_regression = self.df[["PHENO", feature]].copy()
            df_regression = df_regression.rename(columns={feature: safe_feature_name})

            formula = f"PHENO ~ {safe_feature_name}"
            model = smf.ols(formula, data=df_regression).fit()

            conf_int = model.conf_int().loc[safe_feature_name]
            regression_results.append({
                "Feature": feature,
                "Coefficient": model.params[safe_feature_name],
                "StdErr": model.bse[safe_feature_name],
                "TValue": model.tvalues[safe_feature_name],
                "pval": model.pvalues[safe_feature_name],
                "CI_L95": conf_int[0],
                "CI_U95": conf_int[1],
                "R2": model.rsquared,
                "Adj_R2": model.rsquared_adj,
                "AIC": model.aic,
                "BIC": model.bic,
                "F_statistic": model.fvalue,
                "F_pvalue": model.f_pvalue,
                "n_obs": int(model.nobs),
            })
        
        # Sort features by p-value
        df_regression_results = pd.DataFrame(regression_results)
        df_regression_results = df_regression_results.sort_values(by="pval")

        # Find features that meet the user-defined threshold
        features_to_include = df_regression_results.loc[df_regression_results["pval"] < self._ols_threshold, "Feature"].tolist()
        print(f"{len(features_to_include)} out of {len(df_regression_results)} features achieved an OLS p-value below {self._ols_threshold} and will be included")

        # Filter data to keep significant features in order of importance
        self.df = self.df[["ID", "PHENO"] + features_to_include]

        # Save regression summary
        ols_regression_summary_path = f"ols_regression{f'_fold{self.fold}' if self.fold is not None else ''}.tsv"
        df_regression_results.to_csv(
            self.prefix.joinpath(ols_regression_summary_path), 
            sep="\t",
            index=False,
        )
        
    def extra_trees(self):
        # Fit extraTrees model.
        print(f"Using {self.n_est} extraTrees estimators...")
        if self.data_type == "d":
            clf = ensemble.ExtraTreesClassifier(n_estimators=self.n_est)
        if self.data_type == "c":
            clf = ensemble.ExtraTreesRegressor(n_estimators=self.n_est)
        clf.fit(self.x, self.y)
        
        # Drop the features below threshold.
        model = feature_selection.SelectFromModel(clf, prefit=True)
        df_feature_scores = pd.DataFrame(
            zip(self.x.columns, clf.feature_importances_),
            columns=["Feature_Name", "Score"]
        ).sort_values(by=['Score'], ascending=False)
        feature_importance_path = f"approx_feature_importance{f'_fold{self.fold}' if self.fold is not None else ''}.txt"
        df_feature_scores.to_csv(
            self.prefix.joinpath(feature_importance_path), 
            index=False, 
            sep="\t",
        )

        # Filter to only include important features.
        x_reduced = self.x.iloc[:, model.get_support()]
        self.df = pd.concat([
            self.ids.reset_index(drop=True), 
            self.y.reset_index(drop=True), 
            x_reduced.reset_index(drop=True),
        ], axis = 1, ignore_index=False)

        # Save features
        list_features_path = f"list_features{f'_fold{self.fold}' if self.fold is not None else ''}.txt"
        with open(self.prefix.joinpath(list_features_path), 'w') as f:
            for feature in self.df.columns.values.tolist():
                f.write(feature + "\n")

        print(f"An updated list of the {self.df.shape[1] - 2} features, plus ID and PHENO, in your munged dataForML.h5 file can be found at {self.prefix.joinpath("list_features.txt")}")

    def vif(self):
        self.df.columns = self.df.columns.str.strip()

        print("Sampling 100 rows at random to reduce memory overhead...")
        df_cleaned = self.df.sample(n=100, random_state=self._random_state).copy().reset_index()

        print("Dropping columns that are not features...")
        df_cleaned.drop(columns=["index", "PHENO", "ID"], inplace=True)
        df_cleaned = df_cleaned.astype(float)

        for iteration in range(self.vif_iter):
            print(f"\n\nIteration {iteration + 1}\n\n")
            print("Shuffling columns...")
            df_cleaned = df_cleaned.sample(frac=1, axis=1, random_state=self._random_state)

            print("Generating chunked, randomized dataframes...")
            self.df_list = []
            for i in range((df_cleaned.shape[1] + self.chunk_size - 1) // self.chunk_size):
                self.df_list.append(df_cleaned.iloc[:, i*self.chunk_size : (i+1)*self.chunk_size].copy())
            
            print(f"Dropping columns with a VIF threshold greater than {self.vif_threshold}")
            for df in self.df_list:
                while True:
                    if df.shape[1] == 1:
                        break

                    # Calculate VIF for all columns
                    vif_vals = joblib.Parallel(n_jobs=5)(
                        joblib.delayed(outliers_influence.variance_inflation_factor)(df.values, i)
                        for i in range(df.shape[1])
                    )

                    max_vif = max(vif_vals)
                    if max_vif <= self.vif_threshold:
                        break

                    # Drop the column with the highest VIF
                    max_idx = vif_vals.index(max_vif)
                    col_to_drop = df.columns[max_idx]
                    print(f'Dropping "{col_to_drop}" with VIF = {max_vif:.2f}')
                    df.drop(columns=[col_to_drop], inplace=True)

            print("\nVIF calculation on all chunks complete! \n")
            df_cleaned = pd.concat(self.df_list, axis=1)
            df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.duplicated()]

        # Use the original dataframe with the features to keep 
        self.df = self.df[["PHENO", "ID"] + df_cleaned.columns.values.tolist()]
               