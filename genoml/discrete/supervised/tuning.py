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
from pathlib import Path
import seaborn as sns
from scipy import stats
import sklearn
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from time import time
import xgboost

'''
class tune():

    # def ROC(self):
    #     ### Export the ROC curve

    #     plot_out = self.run_prefix.joinpath('tunedModel_allSample_ROC.png')

    #     test_predictions = self.algo_tuned.predict_proba(self.X_tune)
    #     test_predictions = test_predictions[:, 1]

    #     fpr, tpr, thresholds = metrics.roc_curve(self.y_tune, test_predictions)
    #     roc_auc = metrics.auc(fpr, tpr)

    #     plt.figure()
    #     plt.plot(fpr, tpr, color='purple', label='All sample ROC curve (area = %0.2f)' % roc_auc + '\nMean cross-validation ROC curve (area = %0.2f)' % self.cv_tuned.mean())
    #     plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False positive rate')
    #     plt.ylabel('True positive rate')
    #     plt.title('Receiver operating characteristic (ROC) - ' + self.best_algo + '- tuned' )
    #     plt.legend(loc="lower right")
    #     plt.savefig(plot_out, dpi = 600)

    #     print()
    #     print(f"We are also exporting a ROC curve for you here {plot_out} this is a graphical representation of AUC in all samples for the best performing algorithm.")
    
    def export_tuned_data(self):
        tune_predicteds_probs = self.algo_tuned.predict_proba(self.X_tune)
        tune_case_probs = tune_predicteds_probs[:, 1]
        tune_predicted_cases = self.algo_tuned.predict(self.X_tune)

        tune_case_probs_df = pd.DataFrame(tune_case_probs)
        tune_predicted_cases_df = pd.DataFrame(tune_predicted_cases)
        y_tune_df = pd.DataFrame(self.y_tune)
        IDs_tune_df = pd.DataFrame(self.IDs_tune)

        tune_out = pd.concat([IDs_tune_df.reset_index(), y_tune_df.reset_index(drop=True), tune_case_probs_df.reset_index(drop=True), tune_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        tune_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
        tune_out = tune_out.drop(columns=['INDEX'])

        self.tune_out = tune_out

        tune_outfile = self.run_prefix.joinpath('tunedModel_allSample_Predictions.csv')
        tune_out.to_csv(tune_outfile, index=False)
        self.tune_out = tune_out

        print("")
        print(f"Preview of the exported predictions for the tuning samples which is naturally overfit and exported as {tune_outfile} in the similar format as in the initial training phase of GenoML.")
        print("#"*70)
        print(tune_out.head())
        print("#"*70)
    
    def export_tune_hist_prob(self):
        # Export histograms of probabilities
        genoML_colors = ["cyan","purple"]

        # Using the withheld sample data 
        to_plot_df = self.tune_out
        to_plot_df['percent_probability'] = to_plot_df['CASE_PROBABILITY']*100
        to_plot_df['Probability (%)'] = to_plot_df['percent_probability'].round(decimals=0)
        to_plot_df['Reported Status'] = to_plot_df['CASE_REPORTED']
        to_plot_df['Predicted Status'] = to_plot_df['CASE_PREDICTED']

        to_plot_df.describe()

        # Start plotting
        sns.displot(data=to_plot_df, x="Probability (%)", hue="Predicted Status", col="Reported Status", kde=True, palette=genoML_colors, alpha=0.2)

        plot_out = self.run_prefix.joinpath('tunedModel_allSample_probabilities.png')
        plt.savefig(plot_out, dpi=300)

        print(f"We are also exporting probability density plots to the file {plot_out} this is a plot of the probability distributions of being a case, stratified by case and control status for all samples.")
'''

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics, model_selection
from time import time
import genoml.discrete.utils as discrete_utils


class tune:
    def __init__(self, df, run_prefix, max_iter, cv_count):
        path = Path(run_prefix).joinpath("Tune")
        if not path.is_dir():
            path.mkdir()

        self.run_prefix = path
        self.max_iter = max_iter
        self.cv_count = cv_count

        self.y_tune = df.PHENO
        self.IDs_tune = df.ID
        self.X_tune = df.drop(columns=['PHENO', 'ID'])

        best_algo_name_in = Path(run_prefix).joinpath("Train").joinpath('best_algorithm.txt')
        best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
        self.best_algo = str(best_algo_df.iloc[0, 0])

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

    def select_tuning_parameters(self, metric_tune):
        best_algo = self.best_algo
        self.metric_tune = metric_tune

        algo = discrete_utils.get_best_algo(best_algo)
        hyperparameters = discrete_utils.get_hyperparameters(best_algo)

        if metric_tune == "AUC":
            scoring_metric = metrics.make_scorer(metrics.roc_auc_score, needs_proba=True, multi_class="ovr")
        elif metric_tune == "Balanced_Accuracy":
            scoring_metric = metrics.make_scorer(metrics.balanced_accuracy_score, needs_proba=False)

        self.algo = algo
        self.hyperparameters = hyperparameters
        self.scoring_metric = scoring_metric

        return algo, hyperparameters, scoring_metric

    def apply_tuning_parameters(self):
        # Randomized search with CV to tune
        print("Here is a summary of the top 10 iterations of the hyperparameter tuning...")

        rand_search = model_selection.RandomizedSearchCV(
            estimator=self.algo,
            param_distributions=self.hyperparameters,
            scoring=self.scoring_metric,
            n_iter=self.max_iter,
            cv=self.cv_count,
            n_jobs=-1,
            random_state=153,
            verbose=0,
        )
        start = time()
        rand_search.fit(self.X_tune, self.y_tune)
        print("RandomizedSearchCV took %.2f seconds for %d candidates parameter iterations." % ((time() - start), self.max_iter))
        self.rand_search = rand_search
        self.searchCVResults = rand_search.cv_results_
        self.algo_tuned = rand_search.best_estimator_

        return rand_search.cv_results_

    def report_tune(self):
        # Summary of the top 10 iterations of the hyperparameter tune
        n_top = 10
        results = self.searchCVResults

        top10_log_cols = ["Model_Rank", "Mean_Validation_Score", "Mean_Standard_Deviation", "Parameters"]
        log_entries = []

        for rank in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == rank)
            for candidate in candidates:
                print("Model with Rank: {0}".format(rank))
                print("Mean Validation Score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate])
                )
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
                top10_log_entry = pd.DataFrame(
                    [[
                        rank,
                        results['mean_test_score'][candidate],
                        results['std_test_score'][candidate],
                        results['params'][candidate]
                    ]],
                    columns=top10_log_cols,
                )
                log_entries.append(top10_log_entry)

        top10_log_table = pd.concat(log_entries)
        log_outfile = self.run_prefix.joinpath('tunedModel_top10Iterations_Summary.csv')
        top10_log_table.to_csv(log_outfile, index=False)

        print(
            f"We are exporting a summary table of the top 10 iterations of the hyperparameter tuning step and its parameters here {log_outfile}.")

        return top10_log_table

    def summarize_tune(self):
        print("Here is the cross-validation summary of your best tuned model hyperparameters...")
        cv_tuned = model_selection.cross_val_score(
            estimator=self.rand_search.best_estimator_,
            X=self.X_tune,
            y=self.y_tune,
            scoring=self.scoring_metric,
            cv=self.cv_count,
            n_jobs=-1,
            verbose=0,
        )
        print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC or Balanced_Accuracy for discrete phenotypes and explained variance for continuous phenotypes:")
        print(cv_tuned)
        print("Mean cross-validation score:")
        print(cv_tuned.mean())
        print("Standard deviation of the cross-validation score:")
        print(cv_tuned.std())

        print("")

        print("Here is the cross-validation summary of your baseline/default hyperparameters for the same algorithm on the same data...")
        cv_baseline = model_selection.cross_val_score(
            estimator=self.algo,
            X=self.X_tune,
            y=self.y_tune,
            scoring=self.scoring_metric,
            cv=self.cv_count,
            n_jobs=-1,
            verbose=0,
        )
        print("Scores per cross-validation of the metric to be maximized (AUC or Balanced_Accuracy)")
        print(cv_baseline)
        print("Mean cross-validation score:")
        print(cv_baseline.mean())
        print("Standard deviation of the cross-validation score:")
        print(cv_baseline.std())

        print("")
        print("Just a note, if you have a relatively small variance among the cross-validation iterations, there is a higher chance of your model being more generalizable to similar datasets.")

        self.cv_baseline = cv_baseline
        self.cv_tuned = cv_tuned

        # Output a log table summarizing CV mean scores and standard deviations
        summary_CV_log_cols = [
            "Mean_CV_Score_Baseline",
            "Standard_Dev_CV_Score_Baseline",
            "Min_CV_Score_Baseline",
            "Max_CV_Score_Baseline",
            "Mean_CV_Score_BestTuned",
            "Standard_Dev_CV_Score_BestTuned",
            "Min_CV_Score_BestTuned",
            "Max_CV_Score_BestTuned",
        ]
        summary_CV_log_table = pd.DataFrame(columns=summary_CV_log_cols)
        summary_CV_log_entry = pd.DataFrame(
            [[
                cv_baseline.mean(),
                cv_baseline.std(),
                cv_baseline.min(),
                cv_baseline.max(),
                cv_tuned.mean(),
                cv_tuned.std(),
                cv_tuned.min(),
                cv_tuned.max(),
            ]],
            columns=summary_CV_log_cols,
        )
        summary_CV_log_table = pd.concat([summary_CV_log_table, summary_CV_log_entry])
        log_outfile = self.run_prefix.joinpath('tunedModel_CV_Summary.csv')
        summary_CV_log_table.to_csv(log_outfile, index=False)

        print(f"We are exporting a summary table of the cross-validation mean score and standard deviation of the baseline vs. best tuned model here {log_outfile}.")

    def compare_performance(self):
        cv_tuned = self.cv_tuned
        cv_baseline = self.cv_baseline
        algo_tuned_out = self.run_prefix.joinpath('tunedModel.joblib')

        print("")
        if cv_baseline.mean() > cv_tuned.mean():
            print("Based on comparisons of the default parameters to your hyperparameter tuned model, the baseline model actually performed better.")
            print("Looks like the tune wasn't worth it, we suggest either extending the tune time or just using the baseline model for maximum performance.")
            print("")
            print("Let's shut everything down, thanks for trying to tune your model with GenoML.")

        if cv_baseline.mean() < cv_tuned.mean():
            print("Based on comparisons of the default parameters to your hyperparameter tuned model, the tuned model actually performed better.")
            print("Looks like the tune was worth it, we suggest using this model for maximum performance, lets summarize and export this now.")
            print("In most cases, if opting to use the tuned model, a separate test dataset is a good idea. GenoML has a module to fit models to external data.")
            algo_tuned = self.rand_search.best_estimator_
            self.algo = algo_tuned

        joblib.dump(self.algo, algo_tuned_out)

    def plot_results(self, save=False):
        # Issue #24: RandomForestClassifier is finicky - can't recalculate moving forward like the other
        plot_path = self.run_prefix.joinpath('tunedModel_withheldSample_ROC.png')
        self.algo.fit(self.X_tune, self.y_tune)
        ground_truth = self.y_tune
        predictions = self.algo.predict(self.X_tune)
        discrete_utils.ROC(save, plot_path, ground_truth, predictions, plot_label=self.best_algo)
        discrete_utils.precision_recall_plot(save, plot_path, ground_truth, predictions, plot_label=self.best_algo)

    def export_prediction_data(self):
        tune_out = discrete_utils.export_prediction_tables(
            self.algo,
            self.y_tune,
            self.X_tune,
            self.IDs_tune,
            self.run_prefix.joinpath('tunedModel_withheldSample_testPredictions.csv'),
        )

        discrete_utils.export_prob_hist(
            tune_out,
            self.run_prefix.joinpath('tunedModel_withheldSample_probabilities'),
        )