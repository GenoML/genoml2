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
import pandas as pd
from pathlib import Path
from sklearn import discriminant_analysis, ensemble, linear_model, model_selection, neighbors, neural_network, svm
import xgboost
import genoml.discrete.utils as discrete_utils


# Define the train class
class train:

    def __init__(self, df, run_prefix):
        # code that will prepare the data
        y = df.PHENO
        X = df.drop(columns=['PHENO'])

        # Split the data
        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
        IDs_train = X_train.ID
        IDs_valid = X_valid.ID
        X_train = X_train.drop(columns=['ID'])
        X_valid = X_valid.drop(columns=['ID'])

        path = Path(run_prefix).joinpath("Train")
        if not path.is_dir():
            path.mkdir()

        # Saving the prepped data the other classes will need
        self.df = df
        self.run_prefix = path
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.IDs_train = IDs_train
        self.IDs_valid = IDs_valid

        # Where the results will be stored
        self.log_table = None
        self.best_algo = None
        self.algo = None
        self.rfe_df = None

        ### TODO: Weird results for: SGDClassifier, QuadraticDiscriminantAnalysis
        ### TODO: MLPClassifier and QuadraticDiscriminantAnalysis do not always converge on training data
        # The methods we will use
        self.algorithms = [
            linear_model.LogisticRegression(solver='lbfgs'),
            ensemble.RandomForestClassifier(n_estimators=100),
            ensemble.AdaBoostClassifier(),
            ensemble.GradientBoostingClassifier(),
            linear_model.SGDClassifier(loss='modified_huber'),
            svm.SVC(probability=True, gamma='scale'),
            neural_network.MLPClassifier(),
            neighbors.KNeighborsClassifier(),
            discriminant_analysis.LinearDiscriminantAnalysis(),
            discriminant_analysis.QuadraticDiscriminantAnalysis(),
            ensemble.BaggingClassifier(),
            xgboost.XGBClassifier(),
        ]

    # Report and data summary you want
    def summary(self):
        print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
        print("#" * 70)
        print(self.df.describe())
        print("#" * 70)

    def compete(self):
        self.log_table = []
        for algo in self.algorithms:
            log_entry = discrete_utils.summary_stats(algo, self.y_valid, self.X_valid)
            self.log_table.append(log_entry)
        self.log_table = pd.concat(self.log_table)
        print("#" * 70)
        print("")

    def results(self, metric_max):
        self.metric_max = metric_max
        metric_keys = {
            'AUC': 'AUC_Percent',
            'Balanced_Accuracy': 'Balanced_Accuracy_Percent',
            'Sensitivity': 'Sensitivity',
            'Specificity': 'Specificity'
        }

        # Drop those that have an accuracy less than 50%, balanced accuracy less than 50%, delta between sensitivity
        # and specificity greater than 0.85, sensitivity equal to 0 or 1, or specificity equal to 0 or 1.
        sorted_table = self.log_table[
            (self.log_table['AUC_Percent'] > 50)
            & (self.log_table['Balanced_Accuracy_Percent'] > 50)
            & (self.log_table['Sensitivity'].sub(self.log_table['Specificity'], axis=0).abs() < 0.85)
            & (self.log_table['Sensitivity'] != 0.0)
            & (self.log_table['Sensitivity'] != 1.0)
            & (self.log_table['Specificity'] != 0.0)
            & (self.log_table['Specificity'] != 1.0)
            ]

        # If for some reason ALL the algorithms are overfit...
        if sorted_table.empty:
            print(
                'It seems as though all the algorithms are over-fit in some way or another... We will report the best algorithm based on your chosen metric instead and use that moving forward.')
            sorted_table = self.log_table

        # Sort the table and reset the index so that we can access the best algorithm at index 0
        sorted_table = sorted_table.sort_values(metric_keys[self.metric_max], ascending=False)
        sorted_table = sorted_table.reset_index(drop=True)

        # Get the best algorithm's name and AUC.
        self.best_algo = sorted_table.at[0, 'Algorithm']
        self.roc_auc = float(sorted_table.at[0, 'AUC_Percent']) * 0.01

        # If for some reason algorithms report the exact same score, only choose the first one so things don't crash
        if isinstance(self.best_algo, list):
            self.best_algo = self.best_algo[0]

    def export_model(self):
        algo = discrete_utils.get_best_algo(self.best_algo)
        algo.fit(self.X_train, self.y_train)

        print("...remember, there are occasionally slight fluctuations in model performance on the same withheld samples...")
        print("#" * 70)
        print(self.best_algo)

        discrete_utils.calculate_accuracy_scores(
            algo,
            self.y_valid,
            self.X_valid,
        )

        ### Save it using joblib
        algo_out = self.run_prefix.joinpath('trainedModel.joblib')
        joblib.dump(algo, algo_out)

        print("#" * 70)
        print(f"... this model has been saved as {algo_out} for later use and can be found in your working directory.")

        self.algo = algo

    def plot_results(self, save=False):
        # Issue #24: RandomForestClassifier is finicky - can't recalculate moving forward like the other
        plot_path = self.run_prefix.joinpath('trainedModel_withheldSample_ROC.png')
        ground_truth = self.y_valid.values
        predictions = self.algo.predict(self.X_valid)
        discrete_utils.ROC(save, plot_path, ground_truth, predictions, plot_label=self.best_algo)
        discrete_utils.precision_recall_plot(save, plot_path, ground_truth, predictions, plot_label=self.best_algo)

    def export_prediction_data(self):
        discrete_utils.export_prediction_tables(
            self.algo,
            self.y_train,
            self.X_train,
            self.IDs_train,
            self.run_prefix.joinpath('trainedModel_withheldSample_trainPredictions.csv'),
        )

        valid_out = discrete_utils.export_prediction_tables(
            self.algo,
            self.y_valid,
            self.X_valid,
            self.IDs_valid,
            self.run_prefix.joinpath('trainedModel_withheldSample_validPredictions.csv'),
        )

        discrete_utils.export_prob_hist(
            valid_out,
            self.run_prefix.joinpath('trainedModel_withheldSample_validProbabilities'),
        )

    def save_results(self, algorithm_results=False, best_algorithm=False):
        if algorithm_results:
            log_table = self.log_table
            log_outfile = self.run_prefix.joinpath('training_withheldSamples_performanceMetrics.csv')
            print(f"""A complete table of the performance metrics can be found at {log_outfile}
            Note that any models that were overfit (if AUC or Balanced Accuracy was 50% or less, or sensitivity/specificity were 1 or 0) were not considered when nominating the best algorithm.""")

            print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
            print("#" * 70)
            print(log_table)
            print("#" * 70)

            log_table.to_csv(log_outfile, index=False)

        if best_algorithm:
            best_algo = self.best_algo
            print(
                f"Based on your withheld samples, the algorithm with the best {self.metric_max} is the {best_algo}... let's save that model for you.")
            best_algo_name_out = self.run_prefix.joinpath("best_algorithm.txt")
            file = open(best_algo_name_out, 'w')
            file.write(self.best_algo)
            file.close()

