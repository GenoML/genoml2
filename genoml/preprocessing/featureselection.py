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

import pandas as pd
from sklearn import ensemble
from sklearn import feature_selection

class FeatureSelection:
    def __init__(self, run_prefix, df, data_type, n_est):
        # Double check there are no NAs in the dataset before proceeding
        remove_cols = df.columns[df.isna().any()].tolist()
        df.drop(remove_cols, axis=1, inplace=True)

        self.run_prefix = run_prefix
        self.n_est = n_est
        self.data_type = data_type
        self.y = df.PHENO
        self.ids = df.ID
        self.x = df.drop(columns=['PHENO','ID'])


    def rank(self):
        print(f"Beginning featureSelection using {self.n_est} extraTrees estimators...")
        if self.data_type == "d":
            clf = ensemble.ExtraTreesClassifier(n_estimators=self.n_est)
        if self.data_type == "c":
            clf = ensemble.ExtraTreesRegressor(n_estimators=self.n_est)
        clf.fit(self.x, self.y)
        
        # Code to drop the features below threshold and return the data set like it was (aka add PHENO and IDs back)
        ### TODO: Look into warning message from this
        model = feature_selection.SelectFromModel(clf, prefit=True)
        df_feature_scores = pd.DataFrame(
            zip(self.x.columns, clf.feature_importances_),
            columns=["Feature_Name", "Score"]
        )
        df_feature_scores = df_feature_scores.sort_values(by=['Score'], ascending=False)
        feature_scores_outfile = self.run_prefix.joinpath("approx_feature_importance.txt")
        df_feature_scores.to_csv(feature_scores_outfile, index=False, sep="\t")

        x_reduced = self.x.iloc[:, model.get_support()]
        self.df_selecta = pd.concat([
            self.ids.reset_index(drop=True), 
            self.y.reset_index(drop=True), 
            x_reduced.reset_index(drop=True)
        ], axis = 1, ignore_index=False
        )
        print(f"You have reduced your dataset to {x_reduced.shape[0]} samples at {x_reduced.shape[1]} features, not including ID and PHENO.")
        return self.df_selecta


    def export_data(self):
        features_list = self.df_selecta.columns.values.tolist()
        features_listpath = self.run_prefix.joinpath("list_features.txt")
        with open(features_listpath, 'w') as f:
            for feature in features_list:
                f.write(feature + "\n")

        print(f"""
        An updated list of {len(features_list)} features, including ID and PHENO, that is in your munged dataForML.h5 file can be found here {features_listpath}
        """)
        