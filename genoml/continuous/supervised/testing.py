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
import statsmodels.formula.api as sm
from sklearn.metrics import explained_variance_score, mean_squared_error, median_absolute_error, r2_score

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
        y_test = self.df.PHENO
        X_test = self.df.drop(columns=['PHENO'])
        IDs_test = X_test.ID
        X_test = X_test.drop(columns=['ID'])

        # Save variables to use globally within the class 
        self.y_test = y_test
        self.X_test = X_test
        self.IDs_test = IDs_test

        return X_test
    
    def performance_metrics(self):
        
        log_cols=["Explained_variance_score", "Mean_squared_error", "Median_absolute_error", "R2_score"]
        log_table = pd.DataFrame(columns=log_cols)
                
        self.loaded_model.fit(self.X_test, self.y_test)

        print("")
        print("#"*70)

        test_predictions = self.loaded_model.predict(self.X_test)
        test_predictions = test_predictions
        evs = explained_variance_score(self.y_test, test_predictions)
        print("Explained variance score: {:.4}".format(evs))
            
        test_predictions = self.loaded_model.predict(self.X_test)
        test_predictions = test_predictions
        mse = mean_squared_error(self.y_test, test_predictions)
        print("Mean squared error: {:.4}".format(mse))
            
        test_predictions = self.loaded_model.predict(self.X_test)
        test_predictions = test_predictions
        mae = median_absolute_error(self.y_test, test_predictions)
        print("Median absolute error: {:.4}".format(mae))
            
        test_predictions = self.loaded_model.predict(self.X_test)
        test_predictions = test_predictions
        r2s = r2_score(self.y_test, test_predictions)
        print("R^2 score: {:.4}".format(r2s))
            
        log_entry = pd.DataFrame([[evs, mse, mae, r2s]], columns=log_cols)
        log_table = log_table.append(log_entry)

        print("#"*70)

        print("")

        log_outfile = self.run_prefix + '.testedModel_allSamples_performanceMetrics.csv'

        print("")
        print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
        print("#"*70)
        print(log_table)
        print("#"*70)
        print("")

        log_table.to_csv(log_outfile, index=False)

        self.log_table = log_table 
        return log_table
    
    def export_pheno_predictions(self):
        
        test_predicted_values = self.loaded_model.predict(self.X_test)
        test_predicted_values_df = pd.DataFrame(test_predicted_values)
        y_test_df = pd.DataFrame(self.y_test)
        IDs_test_df = pd.DataFrame(self.IDs_test)

        test_out = pd.concat([IDs_test_df.reset_index(), y_test_df.reset_index(drop=True), test_predicted_values_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        test_out.columns=["INDEX","ID","PHENO_REPORTED","PHENO_PREDICTED"]
        test_out = test_out.drop(columns=["INDEX"])

        test_outfile = self.run_prefix + '.testedModel_allSample_predictions.csv'
        test_out.to_csv(test_outfile, index=False)

        print("")
        print(f"Preview of the exported predictions exported as {test_outfile}, these are pretty straight forward.")
        print("They generally include the sample ID, the previously reported phenotype, and the predicted phenotype from that algorithm.")
        print("#"*70)
        print(test_out.head())
        print("#"*70)

        self.test_out = test_out 
        return test_out
    
    def regression_summary(self):

        genoML_colors = ["cyan","purple"]

        sns_plot = sns.regplot(data=self.test_out, y="PHENO_REPORTED", x="PHENO_PREDICTED", scatter_kws={"color": "cyan"}, line_kws={"color": "purple"})

        plot_out = self.run_prefix + '.testedModel_allSamples_regressionPlot.png'
        sns_plot.figure.savefig(plot_out, dpi=600)

        print("")
        print(f"We are also exporting a regression plot for you here {plot_out}, this is a graphical representation of the difference between the reported and predicted phenotypes in the withheld test data for the best performing algorithm.")

        print("")
        print("Here is a quick summary of the regression comparing PHENO_REPORTED ~ PHENO_PREDICTED in the withheld test data...")
        print("")

        reg_model = sm.ols(formula='PHENO_REPORTED ~ PHENO_PREDICTED', data=self.test_out)
        fitted = reg_model.fit()
        print(fitted.summary())

        fitted_out = self.run_prefix + 'testedModel_allSamples_regressionSummary.csv'
        
        with open(fitted_out, 'w') as fh:
            fh.write(fitted.summary().as_csv())

        print(f"We are exporting this summary here: {fitted_out}")

        print("")
        print("...always good to see the P value for the predictor.")
