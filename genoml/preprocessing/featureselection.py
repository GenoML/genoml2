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

class featureselection:
    def __init__(self, run_prefix, df, data_type, n_est):
        self.run_prefix = run_prefix
        self.featureRanks = None
        self.n_est = n_est
        self.data_type = data_type

        self.y = df['PHENO']
        self.X = df.drop(columns=['PHENO'])
        X = self.X
        self.IDs = X.ID
        self.X = X.drop(columns=['ID'])

    def rank(self):
        print(f"""
            Beginning featureSelection using {self.n_est} estimators...""")

        if (self.data_type == "d"):
            print(f"""
            using extraTrees Classifier for your discrete dataset 
            """)
            clf = ensemble.ExtraTreesClassifier(n_estimators=self.n_est)
        
        if (self.data_type == "c"):
            print(f"""
            using extraTrees Regressor for your continuous dataset
            """)
            clf = ensemble.ExtraTreesRegressor(n_estimators=self.n_est)
        
        clf.fit(self.X, self.y)
        self.featureRanks = clf.feature_importances_
        
        # Code to drop the features below threshold and return the data set like it was (aka add PHENO and IDs back)
        model = feature_selection.SelectFromModel(clf, prefit=True) # find this import at top
        df_editing = model.transform(self.X)
        print("""
        Printing feature name that corresponds to the dataframe column name, then printing the relative importance as we go...
        """)
        
        list_featureScores = []

        for col,score in zip(self.X.columns,clf.feature_importances_):
            print(col,score)
            list_featureScores.append([col, score])

        df_featureScores = pd.DataFrame(list_featureScores, columns=["Feature_Name", "Score"])
        #df_featureScores = df_featureScores[df_featureScores['Score'] !=0]
        df_featureScores = df_featureScores.sort_values(by=['Score'], ascending=False)
        featureScores_outfile = self.run_prefix + ".approx_feature_importance.txt"
        df_featureScores.to_csv(featureScores_outfile, index=False, sep="\t")

        print(f"""
        You have reduced your dataset to {df_editing.shape[0]} samples at {df_editing.shape[1]} features, not including ID and PHENO.
        """)

        y_df = self.y
        ID_df = pd.DataFrame(self.IDs)
        features_selected = model.get_support()
        X_reduced = self.X.iloc[:,features_selected]
        df_selecta = pd.concat([ID_df.reset_index(drop=True), y_df.reset_index(drop=True), X_reduced.reset_index(drop=True)], axis = 1, ignore_index=False)
        
        self.df_selecta = df_selecta
        self.featureScores_outfile = featureScores_outfile

        return df_selecta
    
    def export_data(self):
        ## Export reduced data
        outfile_h5 = self.run_prefix + ".dataForML.h5"
        self.df_selecta.to_hdf(outfile_h5, key='dataForML')

        features_list = self.df_selecta.columns.values.tolist()
    
        features_listpath = self.run_prefix + ".list_features.txt"
        with open(features_listpath, 'w') as f:
            for feature in features_list:
                f.write("%s\n" % feature)

        print(f"""Exporting a new {outfile_h5} file that has a reduced feature set based on your importance approximations. 
        This is a good dataset for general ML applications for the chosen PHENO as it includes only features that are likely to impact the model.

        An updated list of {len(features_list)} features, including ID and PHENO, that is in your munged dataForML.h5 file can be found here {features_listpath}
        
        A file with all your features, ranked from largest contributors at the top to smallest contributors at the bottom, can be found at {self.featureScores_outfile}.
        """)
        