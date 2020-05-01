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

from joblib import dump
from time import time
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy import stats
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
import statsmodels.formula.api as sm
import xgboost


# Import the necessary internal GenoML packages 

# Define the tune class 
class tune():
    def __init__(self, df, run_prefix, max_iter, cv_count):
        self.run_prefix = run_prefix
        self.max_iter = max_iter
        self.cv_count = cv_count
       
        self.y_tune = df.PHENO
        self.X_tune = df.drop(columns=['PHENO'])
        self.IDs_tune = self.X_tune.ID
        self.X_tune = self.X_tune.drop(columns=['ID'])

        best_algo_name_in = run_prefix + '.best_algorithm.txt'
        best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
        self.best_algo = str(best_algo_df.iloc[0,0])

        self.algorithms = [
            linear_model.LinearRegression(),
            ensemble.RandomForestRegressor(),
            ensemble.AdaBoostRegressor(),
            ensemble.GradientBoostingRegressor(),
            linear_model.SGDRegressor(),
            svm.SVR(),
            neural_network.MLPRegressor(),
            neighbors.KNeighborsRegressor(),
            ensemble.BaggingRegressor(),
            xgboost.XGBRegressor()
        ]

        # Initialize a few variables we will be using later 
        self.log_table = None
        self.best_algo_name_in = None
        self.best_algo_df = None
        self.hyperparameters = None
        self.scoring_metric = None
        self.cv_tuned = None 
        self.cv_baseline = None 
        self.algo = None
        self.searchCVResults = None
        self.rand_search = None
        self.algo_tuned = None
        self.tune_out = None

    def select_tuning_parameters(self):
        best_algo = self.best_algo

        if best_algo == 'LinearRegression':
            algo = getattr(sklearn.linear_model, best_algo)()

        elif  best_algo == 'SGDRegressor':
            algo = getattr(sklearn.linear_model, best_algo)()

        elif (best_algo == 'RandomForestRegressor') or (best_algo == 'AdaBoostRegressor') or (best_algo == 'GradientBoostingRegressor') or  (best_algo == 'BaggingRegressor'):
            algo = getattr(sklearn.ensemble, best_algo)()

        elif best_algo == 'SVR':
            algo = getattr(sklearn.svm, best_algo)(gamma='auto')

        elif best_algo == 'MLPRegressor':
            algo = getattr(sklearn.neural_network, best_algo)()

        elif best_algo == 'XGBRegressor':
            algo = getattr(xgboost, best_algo)()

        elif best_algo == 'KNeighborsRegressor':
            algo = getattr(sklearn.neighbors, best_algo)()
        
        self.algo = algo 

        if best_algo == 'LinearRegression':
            hyperparameters = {"penalty": ["l1", "l2"], "C": stats.randint(1, 10)}
            scoring_metric = metrics.make_scorer(metrics.explained_variance_score)

        elif  best_algo == 'SGDRegressor':
            hyperparameters = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "learning_rate": ["constant", "optimal", "invscaling", "adaptive"]}
            scoring_metric = metrics.make_scorer(metrics.explained_variance_score)

        elif (best_algo == 'RandomForestRegressor') or (best_algo == 'AdaBoostRegressor') or (best_algo == 'GradientBoostingRegressor') or  (best_algo == 'BaggingRegressor'):
            hyperparameters = {"n_estimators": stats.randint(1, 1000)}
            scoring_metric = metrics.make_scorer(metrics.explained_variance_score)

        elif best_algo == 'SVR':
            hyperparameters = {"kernel": ["linear", "poly", "rbf", "sigmoid"], "C": stats.randint(1, 10)}
            scoring_metric = metrics.make_scorer(metrics.explained_variance_score)
            
        elif best_algo == 'MLPRegressor':
            hyperparameters = {"alpha": stats.uniform(0, 1), "learning_rate": ['constant', 'invscaling', 'adaptive']}
            scoring_metric = metrics.make_scorer(metrics.explained_variance_score)

        elif best_algo == 'XGBRegressor':
            hyperparameters = {"max_depth": stats.randint(1, 100), "learning_rate": stats.uniform(0, 1), "n_estimators": stats.randint(1, 100), "gamma": stats.uniform(0, 1)}
            scoring_metric = metrics.make_scorer(metrics.explained_variance_score)

        elif best_algo == 'KNeighborsRegressor':
            hyperparameters = {"leaf_size": stats.randint(1, 100), "n_neighbors": stats.randint(1, 10)}
            scoring_metric = metrics.make_scorer(metrics.explained_variance_score)

        self.hyperparameters = hyperparameters
        self.scoring_metric = scoring_metric

        return algo, hyperparameters, scoring_metric

    def apply_tuning_parameters(self):
        # Randomized search with CV to tune 
        print("Here is a summary of the top 10 iterations of the hyperparameter tuning...")
        rand_search = model_selection.RandomizedSearchCV(estimator=self.algo, param_distributions=self.hyperparameters, scoring=self.scoring_metric, n_iter=self.max_iter, cv=self.cv_count, n_jobs=-1, random_state=153, verbose=0)

        start = time()
        rand_search.fit(self.X_tune, self.y_tune)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
            " parameter iterations." % ((time() - start), self.max_iter))

        self.searchCVResults = rand_search.cv_results_
        self.rand_search = rand_search
        self.algo_tuned = rand_search.best_estimator_
        
        return rand_search.cv_results_

    def report_tune(self):
        # Summary of the top 10 iterations of the hyperparameter tune 
        n_top = 10
        results = self.searchCVResults

        top10_log_cols = ["Model_Rank", "Mean_Validation_Score", "Mean_Standard_Deviation", "Parameters"]
        top10_log_table = pd.DataFrame(columns=top10_log_cols)

        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with Rank: {0}".format(i))
                print("Mean Validation Score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
                top10_log_entry = pd.DataFrame([[i, results['mean_test_score'][candidate], results['std_test_score'][candidate], results['params'][candidate]]], columns=top10_log_cols)
                top10_log_table = top10_log_table.append(top10_log_entry)
        
        log_outfile = self.run_prefix + '.tunedModel_top10Iterations_Summary.csv'
        top10_log_table.to_csv(log_outfile, index=False)

        print(f"We are exporting a summary table of the top 10 iterations of the hyperparameter tuning step and its parameters here {log_outfile}.")

        return top10_log_table

    def summarize_tune(self):
        print("Here is the cross-validation summary of your best tuned model hyperparameters...")
        cv_tuned = model_selection.cross_val_score(estimator=self.rand_search.best_estimator_, X=self.X_tune, y=self.y_tune, scoring=self.scoring_metric, cv=self.cv_count, n_jobs=-1, verbose=0)
        print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC or Balanced_Accuracy for discrete phenotypes and explained variance for continuous phenotypes:")
        print(cv_tuned)
        print("Mean cross-validation score:")
        print(cv_tuned.mean())
        print("Standard deviation of the cross-validation score:")
        print(cv_tuned.std())

        print("")

        print("Here is the cross-validation summary of your baseline/default hyperparamters for the same algorithm on the same data...")
        cv_baseline = model_selection.cross_val_score(estimator=self.algo, X=self.X_tune, y=self.y_tune, scoring=self.scoring_metric, cv=self.cv_count, n_jobs=-1, verbose=0)
        print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC or Balanced_Accuracy for discrete phenotypes and explained variance for continuous phenotypes:")
        print(cv_baseline)
        print("")
        print("Mean Cross-Validation Score:")
        print(cv_baseline.mean())
        print("")
        print("Standard Deviation of the Cross-Validation Score:")
        print(cv_baseline.std())

        print("")
        print("Just a note, if you have a relatively small variance among the cross-validation iterations, there is a higher chance of your model being more generalizable to similar datasets.")

        self.cv_baseline = cv_baseline
        self.cv_tuned = cv_tuned
        
        # Output a log table summarizing CV mean scores and standard deviations 
        summary_CV_log_cols = ["Mean_CV_Score_Baseline", "Standard_Dev_CV_Score_Baseline", "Min_CV_Score_Baseline", "Max_CV_Score_Baseline", "Mean_CV_Score_BestTuned", "Standard_Dev_CV_Score_BestTuned", "Min_CV_Score_BestTuned", "Max_CV_Score_BestTuned"]
        summary_CV_log_table = pd.DataFrame(columns=summary_CV_log_cols)
        summary_CV_log_entry = pd.DataFrame([[cv_baseline.mean(), cv_baseline.std(), cv_baseline.min(), cv_baseline.max(), cv_tuned.mean(), cv_tuned.std(), cv_tuned.min(), cv_tuned.max()]], columns=summary_CV_log_cols)
        summary_CV_log_table = summary_CV_log_table.append(summary_CV_log_entry)
        log_outfile = self.run_prefix + '.tunedModel_CV_Summary.csv'
        summary_CV_log_table.to_csv(log_outfile, index=False)
        
        print(f"We are exporting a summary table of the cross-validation mean score and standard deviation of the baseline vs. best tuned model here {log_outfile}.")

        return summary_CV_log_table
    
    def compare_performance(self):
        cv_tuned = self.cv_tuned
        cv_baseline = self.cv_baseline

        if cv_baseline.mean() > cv_tuned.mean():
            print("Based on comparisons of the default parameters to your hyperparameter tuned model, the baseline model actually performed better.")
            print("Looks like the tune wasn't worth it, we suggest either extending the tune time or just using the baseline model for maximum performance.")
            print("")
            print("Let's shut everything down, thanks for trying to tune your model with GenoML.")
            return self.algo

        if cv_baseline.mean() < cv_tuned.mean():
            print("Based on comparisons of the default parameters to your hyperparameter tuned model, the tuned model actually performed better.")
            print("Looks like the tune was worth it, we suggest using this model for maximum performance, lets summarize and export this now.")
            print("In most cases, if opting to use the tuned model, a separate test dataset is a good idea. GenoML has a module to fit models to external data.")

            algo_tuned = self.rand_search.best_estimator_
                
            ### Save it using joblib
            algo_tuned_out = self.run_prefix + '.tunedModel.joblib'
            dump(algo_tuned, algo_tuned_out)

            return algo_tuned

    def export_tuned_data(self):
        tune_predicted_values = self.algo_tuned.predict(self.X_tune)
        tune_predicted_values_df = pd.DataFrame(tune_predicted_values)
        y_tune_df = pd.DataFrame(self.y_tune)
        IDs_tune_df = pd.DataFrame(self.IDs_tune)

        tune_out = pd.concat([IDs_tune_df.reset_index(), y_tune_df.reset_index(drop=True), tune_predicted_values_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        tune_out.columns=["INDEX","ID","PHENO_REPORTED","PHENO_PREDICTED"]
        tune_out = tune_out.drop(columns=["INDEX"])

        self.tune_out = tune_out

        tune_outfile = self.run_prefix + '.tunedModel_allSample_Predictions.csv'
        tune_out.to_csv(tune_outfile, index=False)

        print("")
        print(f"Preview of the exported predictions for the tuning samples which is naturally overfit and exported as {tune_outfile} in the similar format as in the initial training phase of GenoML.")
        print("#"*70)
        print(tune_out.head())
        print("#"*70)
           
    def export_tune_regression(self):
        genoML_colors = ["cyan","purple"]
        
        sns_plot = sns.regplot(data=self.tune_out, y="PHENO_REPORTED", x="PHENO_PREDICTED", scatter_kws={"color": "cyan"}, line_kws={"color": "purple"})
        
        plot_out = self.run_prefix + '.tunedModel_allSample_regression.png'
        sns_plot.figure.savefig(plot_out, dpi=600)

        print("")
        print(f"We are also exporting a regression plot for you here {plot_out} this is a graphical representation of the difference between the reported and predicted phenotypes in all data for the tuned algorithm.")	

        print("")
        print("Here is a quick summary of the regression comparing PHENO_REPORTED ~ PHENO_PREDICTED in the tuned data for all samples...")
        print("")

        reg_model = sm.ols(formula='PHENO_REPORTED ~ PHENO_PREDICTED', data=self.tune_out)
        fitted = reg_model.fit()
        print(fitted.summary())

        self.fitted = fitted 
        print("")
        print("...always good to see the P for the predictor.")

        return fitted

        