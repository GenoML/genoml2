import os
import sys
import argparse 
import math
import time
import h5py
import joblib
import subprocess
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, roc_curve, auc

# Select features and approximate ranks 

class featureselection:
    def __init__(self, run_prefix, df, dataType, n_est):
        self.run_prefix = run_prefix
        self.featureRanks = None
        self.n_est = n_est
        self.dataType = dataType

        self.y = df['PHENO']
        self.X = df.drop(columns=['PHENO'])
        X = self.X
        self.IDs = X.ID
        self.X = X.drop(columns=['ID'])

    def rank(self):

        if (self.dataType == "d"):
            print(f"""
            Beginning featureSelection using extraTrees Classifier for your discrete dataset using {self.n_est} estimators...
            """)
            clf = ExtraTreesClassifier(n_estimators=self.n_est)
        
        if (self.dataType == "c"):
            print(f"""
            Beginning featureSelection using extraTrees Regressor for your continuous dataset using {self.n_est} estimators...
            """)
            clf = ExtraTreesRegressor(n_estimators=self.n_est)
        
        clf.fit(self.X, self.y)
        self.featureRanks = clf.feature_importances_
        
        # Code to drop the features below threshold and return the data set like it was (aka add pheno and ids back)
        model = SelectFromModel(clf, prefit=True) # find this import at top
        df_editing = model.transform(self.X)
        print("""
        Printing feature name that corresponds to the dataframe column name, then printing the relative importance as we go...
        """)
        
        for col,score in zip(self.X.columns,clf.feature_importances_):
            print(col,score)
            print(col,score, file=open(self.run_prefix + "_approx_feature_importance.txt","a"))
        
        print(f"""
        You have reduced your dataset to {df_editing.shape[0]} samples at {df_editing.shape[1]} features.
        """)
        
        y_df = self.y
        ID_df = pd.DataFrame(self.IDs)
        features_selected = model.get_support()
        X_reduced = self.X.iloc[:,features_selected]
        df_selecta = pd.concat([ID_df.reset_index(drop=True), y_df.reset_index(drop=True), X_reduced.reset_index(drop=True)], axis = 1, ignore_index=False)
        
        self.df_selecta = df_selecta

        return df_selecta
    
    def export_data(self):
        ## Export reduced data
        outfile_h5 = self.run_prefix + ".dataForML.h5"
        self.df_selecta.to_hdf(outfile_h5, key='dataForML')
        print(f"""
        Exporting a new {outfile_h5} file that has a reduced feature set based on your importance approximations. 
        This is a good dataset for general ML applications for the chosen PHENO as it includes only features that are likely to impact the model.
        """)
