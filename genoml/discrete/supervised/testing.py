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
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import sys
import xgboost
import numpy as np
from time import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, roc_curve, auc

class test:
    def __init__(self, df, loaded_model, run_prefix):
        self.df = df 
        self.run_prefix = run_prefix
        self.loaded_model = loaded_model
    
    def prep_df(self):
        print("")
        print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
        print("#"*70)
        print(self.df.describe())
        print("#"*70)
        print("")

        # Save out and drop the PHENO and sample ID columns 
        df = self.df
        y_test = df.PHENO
        IDs_test= df.ID
        X_test = df.drop(columns=['PHENO', 'ID'])


        # Save variables to use globally within the class 
        self.y_test = y_test
        self.X_test = X_test
        self.IDs_test = IDs_test
 
        return X_test
    
    def export_ROC(self):
        
        # Define the output prefix  
        plot_out = self.run_prefix + '.testedModel_allSample_ROC.png'

        test_predictions = self.loaded_model.predict_proba(self.X_test)
        test_predictions = test_predictions[:, 1]

        fpr, tpr, thresholds = roc_curve(self.y_test, test_predictions)
        # Resolving issue #13 - ROC curve reporting to be consistent with performance metrics. 
        #roc_auc = auc(fpr, tpr)
        roc_auc = roc_auc_score(self.y_test, test_predictions)

        plt.figure()
        plt.plot(fpr, tpr, color='purple', label='All sample ROC curve (area = %0.2f)' % roc_auc + '.')
        plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Test Dataset' )
        plt.legend(loc="lower right")
        plt.savefig(plot_out, dpi = 600)

        print("")
        print(f"We are also exporting a ROC curve for you here {plot_out} this is a graphical representation of AUC in all samples for the best performing algorithm.")

        self.roc_auc = roc_auc

        return roc_auc
            

    def export_tested_data(self):
        test_predicteds_probs = self.loaded_model.predict_proba(self.X_test)
        test_case_probs = test_predicteds_probs[:, 1]
        test_predicted_cases = self.loaded_model.predict(self.X_test)

        test_case_probs_df = pd.DataFrame(test_case_probs)
        test_predicted_cases_df = pd.DataFrame(test_predicted_cases)
        y_test_df = pd.DataFrame(self.y_test)
        IDs_test_df = pd.DataFrame(self.IDs_test)

        test_out = pd.concat([IDs_test_df.reset_index(), y_test_df.reset_index(drop=True), test_case_probs_df.reset_index(drop=True), test_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        test_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
        test_out = test_out.drop(columns=['INDEX'])

        test_outfile = self.run_prefix + '.testedModel_allSample_predictions.csv'
        test_out.to_csv(test_outfile, index=False)

        print("")
        print(f"Preview of the exported predictions for the testing samples exported as {test_outfile} in the similar format as in the initial training phase of GenoML.")
        print("#"*70)
        print(test_out.head())
        print("#"*70)

        self.test_out = test_out 

        return test_out
    
    def export_histograms(self):
        genoML_colors = ["cyan","purple"]

        g = sns.FacetGrid(self.test_out, hue="CASE_REPORTED", palette=genoML_colors, legend_out=True,)
        g = (g.map(sns.distplot, "CASE_PROBABILITY", hist=True, rug=False))
        g.add_legend()

        plot_out = self.run_prefix + '.testedModel_allSample_probabilities.png'
        g.savefig(plot_out, dpi=600)

        print("")
        print(f"We are also exporting probability density plots to the file {plot_out} this is a plot of the probability distributions of being a case, stratified by case and control status for all samples.")
        print("")

    def additional_sumstats(self):
        self.loaded_model.fit(self.X_test, self.y_test)

        print("")
        print("#"*70)
        print("Some additional summary stats logging from your application of your model to the test dataset.")
        print("")

        test_predictions = self.loaded_model.predict_proba(self.X_test)
        test_predictions = test_predictions[:, 1]
        
        # Resolving issue #13 - ROC curve reporting to be consistent with performance metrics. 
        # rocauc = roc_auc_score(self.y_test, test_predictions)
        rocauc = self.roc_auc
        print("AUC: {:.4%}".format(rocauc))

        test_predictions = self.loaded_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, test_predictions)
        print("Accuracy: {:.4%}".format(acc))

        test_predictions = self.loaded_model.predict(self.X_test)
        balacc = balanced_accuracy_score(self.y_test, test_predictions)
        print("Balanced Accuracy: {:.4%}".format(balacc))
            
        CM = confusion_matrix(self.y_test, test_predictions)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)
        
        test_predictions = self.loaded_model.predict_proba(self.X_test)
        ll = log_loss(self.y_test, test_predictions)
        print("Log Loss: {:.4}".format(ll))

        log_cols=["AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV"]
        log_table = pd.DataFrame(columns=log_cols)
        log_entry = pd.DataFrame([[rocauc*100, acc*100, balacc*100, ll, sensitivity, specificity, PPV, NPV]], columns=log_cols)
        log_table = log_table.append(log_entry)
        
        print("")
        print("#"*70)
        print("")

        log_outfile = self.run_prefix + '.testedModel_allSamples_performanceMetrics.csv'

        print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
        print("#"*70)
        print(log_table)
        print("#"*70)

        print("")

        log_table.to_csv(log_outfile, index=False)

        self.log_table = log_table

        return log_table