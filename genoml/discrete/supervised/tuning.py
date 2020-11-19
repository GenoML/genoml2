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
import seaborn as sns
from scipy import stats
import sklearn
from sklearn import discriminant_analysis, ensemble, linear_model, metrics, model_selection, \
    neighbors, neural_network, svm
from time import time
import xgboost
from genoml.discrete import supervised


class Tune:

    def __init__(self, df, run_prefix, max_iter, cv_count):
        self.run_prefix = run_prefix
        self.max_iter = max_iter
        self.cv_count = cv_count

        self.y_tune = df.PHENO
        self.X_tune = df.drop(columns=['PHENO', 'ID'])
        self.IDs_tune = df.ID

        self.algorithms = [
            linear_model.LogisticRegression(),
            ensemble.RandomForestClassifier(),
            ensemble.AdaBoostClassifier(),
            ensemble.GradientBoostingClassifier(),
            linear_model.SGDClassifier(loss='modified_huber'),
            svm.SVC(probability=True),
            neural_network.MLPClassifier(),
            neighbors.KNeighborsClassifier(),
            discriminant_analysis.LinearDiscriminantAnalysis(),
            discriminant_analysis.QuadraticDiscriminantAnalysis(),
            ensemble.BaggingClassifier(),
            xgboost.XGBClassifier()
        ]
        self.log_table = None
        self.hyperparameters = None
        self.hyperparameters_map = dict({'LogisticRegression': {"penalty": ["l1", "l2"], "C": stats.randint(1, 10)},
                                         'SGDClassifier': {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                                                           'loss': ['log'], 'penalty': ['l2'], 'n_jobs': [-1]},
                                         'RandomForestClassifier': {"n_estimators": stats.randint(1, 1000)},
                                         'AdaBoostClassifier': {"n_estimators": stats.randint(1, 1000)},
                                         'GradientBoostingClassifier': {"n_estimators": stats.randint(1, 1000)},
                                         'BaggingClassifier': {"n_estimators": stats.randint(1, 1000)},
                                         'SVC': {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                                                 "C": stats.randint(1, 10)},
                                         'ComplementNB': {"alpha": stats.uniform(0, 1)},
                                         'MLPClassifier': {"alpha": stats.uniform(0, 1),
                                                           "learning_rate": ['constant', 'invscaling', 'adaptive']},
                                         'XGBClassifier': {"max_depth": stats.randint(1, 100),
                                                           "learning_rate": stats.uniform(0, 1),
                                                           "n_estimators": stats.randint(1, 100),
                                                           "gamma": stats.uniform(0, 1)},
                                         'KNeighborsClassifier': {"leaf_size": stats.randint(1, 100),
                                                                  "n_neighbors": stats.randint(1, 10)},
                                         'LinearDiscriminantAnalysis': {"tol": stats.uniform(0, 1)},
                                         'QuadraticDiscriminantAnalysis': {"tol": stats.uniform(0, 1)}
                                         })
        self.cv_tuned = None
        self.cv_baseline = None
        self.rand_search = None
        self.algo_tuned = None
        self.algo = None
        self.best_algo =  None
        self.scoring_metric = metrics.make_scorer(metrics.roc_auc_score, needs_proba=True)

    def get_algorithm(self):
        """
        Return model implementation in sklearn library of the best algorithm found in the training step.
        """
        if self.best_algo == 'LogisticRegression':
            return getattr(sklearn.linear_model, self.best_algo)()

        elif self.best_algo == 'SGDClassifier':
            return getattr(sklearn.linear_model, self.best_algo)(loss='modified_huber')

        elif (self.best_algo == 'RandomForestClassifier') or (self.best_algo == 'AdaBoostClassifier') or (
                self.best_algo == 'GradientBoostingClassifier') or (self.best_algo == 'BaggingClassifier'):
            return getattr(sklearn.ensemble, self.best_algo)()

        elif self.best_algo == 'SVC':
            return getattr(sklearn.svm, self.best_algo)(probability=True, gamma='auto')

        elif self.best_algo == 'ComplementNB':
            return getattr(sklearn.naive_bayes, self.best_algo)()

        elif self.best_algo == 'MLPClassifier':
            return getattr(sklearn.neural_network, self.best_algo)()

        elif self.best_algo == 'XGBClassifier':
            return getattr(xgboost, self.best_algo)()

        elif self.best_algo == 'KNeighborsClassifier':
            return getattr(sklearn.neighbors, self.best_algo)()

        elif (self.best_algo == 'LinearDiscriminantAnalysis') or (self.best_algo == 'QuadraticDiscriminantAnalysis'):
            return getattr(sklearn.discriminant_analysis, self.best_algo)()

    def select_tuning_parameters(self, metric_tune="AUC"):
        """
        Finds and stores hyperparameters and implementation of the best algorithm.
        """
        best_algo_name_in = self.run_prefix + '.best_algorithm.txt'
        best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
        self.best_algo = str(best_algo_df.iloc[0, 0])

        self.hyperparameters = self.hyperparameters_map[self.best_algo]
        self.algo = self.get_algorithm()

        if metric_tune == "Balanced_Accuracy":
            self.scoring_metric = metrics.make_scorer(metrics.balanced_accuracy_score, needs_proba=False)

    def apply_tuning_parameters(self):
        """
        Perform randomized search with Cross Validation.
        Stores tuned model.
        """
        print("Here is a summary of the top 10 iterations of the hyperparameter tuning...")
        self.rand_search = model_selection.RandomizedSearchCV(estimator=self.algo,
                                                              param_distributions=self.hyperparameters,
                                                              scoring=self.scoring_metric, n_iter=self.max_iter,
                                                              cv=self.cv_count, n_jobs=-1, random_state=153, verbose=0)
        start = time()

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter iterations." % ((time() - start), self.max_iter))

        self.rand_search.fit(self.X_tune, self.y_tune)
        self.algo_tuned = self.rand_search.best_estimator_

    def report_tune(self):
        """
        Construct and return table containing summary of the top 10 iterations of the hyperparameter tune.
        Save table to csv file.
        """

        n_top = 10
        results = self.rand_search.cv_results_

        top10_log_cols = ["Model_Rank", "Mean_Validation_Score", "Mean_Standard_Deviation", "Parameters"]
        top10_log_table = pd.DataFrame(columns=top10_log_cols)

        # Rank top 10 iteration
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with Rank: {0}".format(i))
                print("Mean Validation Score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
                top10_log_entry = pd.DataFrame([[i, results['mean_test_score'][candidate],
                                                 results['std_test_score'][candidate], results['params'][candidate]]],
                                               columns=top10_log_cols)
                top10_log_table = top10_log_table.append(top10_log_entry)

        log_outfile = self.run_prefix + '.tunedModel_top10Iterations_Summary.csv'
        top10_log_table.to_csv(log_outfile, index=False)

        print(
            f"We are exporting a summary table of the top 10 iterations of the hyperparameter tuning step and its "
            f"parameters here {log_outfile}.")

        return top10_log_table

    def summarize_tune(self):
        """
        Print cross-validation summary of the best tuned model hyperparameters
        """
        print("Here is the cross-validation summary of your best tuned model hyperparameters...")
        self.cv_tuned = model_selection.cross_val_score(estimator=self.rand_search.best_estimator_, X=self.X_tune,
                                                        y=self.y_tune, scoring=self.scoring_metric, cv=self.cv_count,
                                                        n_jobs=-1, verbose=0)
        print(
            "Scores per cross-validation of the metric to be maximized, this scoring metric is AUC or Balanced_Accuracy "
            "for discrete phenotypes and explained variance for continuous phenotypes:")
        print(self.cv_tuned)
        print("Mean cross-validation score:")
        print(self.cv_tuned.mean())
        print("Standard deviation of the cross-validation score:")
        print(self.cv_tuned.std())

        print("")

        print(
            "Here is the cross-validation summary of your baseline/default hyperparamters for the same algorithm on "
            "the same data...")
        self.cv_baseline = model_selection.cross_val_score(estimator=self.algo, X=self.X_tune, y=self.y_tune,
                                                           scoring=self.scoring_metric, cv=self.cv_count, n_jobs=-1,
                                                           verbose=0)
        print(
            "Scores per cross-validation of the metric to be maximized, this scoring metric is AUC or Balanced_Accuracy "
            "for discrete phenotypes and explained variance for continuous phenotypes:")
        print(self.cv_baseline)
        print("Mean cross-validation score:")
        print(self.cv_baseline.mean())
        print("Standard deviation of the cross-validation score:")
        print(self.cv_baseline.std())

        print("")
        print(
            "Just a note, if you have a relatively small variance among the cross-validation iterations, there is a "
            "higher chance of your model being more generalizable to similar datasets.")

        # Output a log table summarizing CV mean scores and standard deviations 
        summary_CV_log_cols = ["Mean_CV_Score_Baseline", "Standard_Dev_CV_Score_Baseline", "Min_CV_Score_Baseline",
                               "Max_CV_Score_Baseline", "Mean_CV_Score_BestTuned", "Standard_Dev_CV_Score_BestTuned",
                               "Min_CV_Score_BestTuned", "Max_CV_Score_BestTuned"]
        summary_CV_log_table = pd.DataFrame(columns=summary_CV_log_cols)
        summary_CV_log_entry = pd.DataFrame([[self.cv_baseline.mean(), self.cv_baseline.std(), self.cv_baseline.min(),
                                              self.cv_baseline.max(), self.cv_tuned.mean(), self.cv_tuned.std(),
                                              self.cv_tuned.min(), self.cv_tuned.max()]], columns=summary_CV_log_cols)
        summary_CV_log_table = summary_CV_log_table.append(summary_CV_log_entry)
        log_outfile = self.run_prefix + '.tunedModel_CV_Summary.csv'
        summary_CV_log_table.to_csv(log_outfile, index=False)

        print(
            f"We are exporting a summary table of the cross-validation mean score and standard deviation of the baseline vs. best tuned model here {log_outfile}.")

        return summary_CV_log_table

    def compare_performance(self):
        """
        Compare performance of the best algorithm against baseline.
        """

        print("")
        if self.cv_baseline.mean() > self.cv_tuned.mean():
            print(
                "Based on comparisons of the default parameters to your hyperparameter tuned model, the baseline model "
                "actually performed better.")
            print(
                "Looks like the tune wasn't worth it, we suggest either extending the tune time or just using the "
                "baseline model for maximum performance.")
            print("")
            print("Let's shut everything down, thanks for trying to tune your model with GenoML.")

        if self.cv_baseline.mean() < self.cv_tuned.mean():
            print(
                "Based on comparisons of the default parameters to your hyperparameter tuned model, the tuned model "
                "actually performed better.")
            print(
                "Looks like the tune was worth it, we suggest using this model for maximum performance, lets summarize "
                "and export this now.")
            print(
                "In most cases, if opting to use the tuned model, a separate test dataset is a good idea. GenoML has a "
                "module to fit models to external data.")

            joblib.dump(self.rand_search.best_estimator_, self.run_prefix + '.tunedModel.joblib')
    
    def export_tuned_data(self):
        """
        Construct, save, and print predictions for the tuning samples.
        """
        
        tune_case_probs = self.algo_tuned.predict_proba(self.X_tune)[:, 1]
        tune_case_probs_df = pd.DataFrame(tune_case_probs)

        tune_predicted_cases = self.algo_tuned.predict(self.X_tune)
        tune_predicted_cases_df = pd.DataFrame(tune_predicted_cases)

        y_tune_df = pd.DataFrame(self.y_tune)
        IDs_tune_df = pd.DataFrame(self.IDs_tune)

        # Construct table
        tune_out = pd.concat(
            [IDs_tune_df.reset_index(), y_tune_df.reset_index(drop=True), tune_case_probs_df.reset_index(drop=True),
             tune_predicted_cases_df.reset_index(drop=True)], axis=1, ignore_index=True)
        tune_out.columns = ['INDEX', 'ID', "CASE_REPORTED", "CASE_PROBABILITY", "CASE_PREDICTED"]
        tune_out = tune_out.drop(columns=['INDEX'])

        # Save table
        tune_outfile = self.run_prefix + '.tunedModel_allSample_Predictions.csv'
        tune_out.to_csv(tune_outfile, index=False)

        print("")
        print(
            f"Preview of the exported predictions for the tuning samples which is naturally over-fit and exported as "
            f"{tune_outfile} in the similar format as in the initial training phase of GenoML.")
        print("#" * 70)
        print(tune_out.head())
        print("#" * 70)
        return tune_out

    def export_tune_hist_prob(self):
        """
        Saves histogram of the estimated distribution in .png file
        """
        sns_plot = sns.FacetGrid(self.export_tuned_data(), hue="CASE_REPORTED", palette=["cyan", "purple"], legend_out=True)
        sns_plot = (sns_plot.map(sns.distplot, "CASE_PROBABILITY", hist=True, rug=False))
        sns_plot.add_legend()
        plot_out = self.run_prefix + '.tunedModel_allSample_probabilities.png'
        sns_plot.savefig(plot_out, dpi=600)

        print("")
        print(
            f"We are also exporting probability density plots to the file {plot_out} this is a plot of the probability "
            f"distributions of being a case, stratified by case and control status for all samples.")
