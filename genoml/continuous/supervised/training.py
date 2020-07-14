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
import seaborn as sns
import statsmodels.formula.api as sm
import xgboost
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm

from genoml import utils


class train:
    def __init__(self, df, run_prefix):
        y = df.PHENO
        x = df.drop(columns=['PHENO'])

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3,
                                                                            random_state=42)  # 70:30
        ids_train = x_train.ID
        ids_test = x_test.ID
        x_train = x_train.drop(columns=['ID'])
        x_test = x_test.drop(columns=['ID'])

        self._df = df
        self._run_prefix = run_prefix
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._ids_train = ids_train
        self._ids_test = ids_test

        self.log_table = None
        self.best_algorithm = None
        self.algorithm = None
        self.rfe_df = None

        candidate_algorithms = [
            ensemble.AdaBoostRegressor(),
            ensemble.BaggingRegressor(),
            ensemble.GradientBoostingRegressor(),
            ensemble.RandomForestRegressor(n_estimators=10),
            linear_model.LinearRegression(),
            linear_model.SGDRegressor(),
            neighbors.KNeighborsRegressor(),
            neural_network.MLPRegressor(),
            svm.SVR(gamma='auto'),
            xgboost.XGBRegressor()
        ]

        self._algorithms = {algorithm.__class__.__name__: algorithm for algorithm in candidate_algorithms}
        self._best_algorithm_name = None
        self._best_algorithm = None
        self._best_algorithm_metrics = None

    def summary(self):
        """Report and data summary you want"""
        utils.DescriptionLoader.print("continuous/supervised/training/Train/summary", data=self._df.describe())

    @utils.DescriptionLoader.function_description("continuous/supervised/training/Train/compete")
    def compete(self):
        """Compete the algorithms"""

        competing_metrics = [metrics.explained_variance_score, metrics.mean_squared_error,
                             metrics.median_absolute_error, metrics.r2_score]
        column_names = ["algorithm", "runtime_s"] + [metric.__name__ for metric in competing_metrics]

        results = []
        for algorithm_name, algorithm in self._algorithms.items():
            with utils.DescriptionLoader.context("continuous/supervised/training/Train/compete/algorithm",
                                                 name=algorithm_name):
                algorithm.fit(self._x_train, self._y_train)
                with utils.Timer() as timer:
                    test_predictions = algorithm.predict(self._x_test)
                    metric_results = [metric_func(self._y_test, test_predictions) for metric_func in competing_metrics]
                row = [algorithm_name, timer.elapsed()] + metric_results

                results_str = self.metrics_to_str(dict(zip(column_names, row)))
                with utils.DescriptionLoader.context("continuous/supervised/training/Train/compete/algorithm/results",
                                                     name=algorithm_name, results=results_str):
                    results.append(row)

        self.log_table = pd.DataFrame(data=results, columns=column_names)

        best_id = self.log_table.explained_variance_score.idxmax()
        self._best_algorithm_name = self.log_table.iloc[best_id].algorithm
        self._best_algorithm = self._algorithms[self._best_algorithm_name]
        self._best_algorithm_metrics = self.log_table.iloc[best_id].to_dict()

        utils.DescriptionLoader.print("continuous/supervised/training/Train/compete/algorithm/best",
                                      algorithm=self._best_algorithm_name,
                                      metrics=self.metrics_to_str(self._best_algorithm_metrics))

    @staticmethod
    def metrics_to_str(metrics_dict):
        rows = []
        for key, value in metrics_dict.items():
            if key == "algorithm":
                rows.append("{}: {}".format(key, value))
            elif key == "runtime_s":
                rows.append("{}: {:0.3f} seconds\n".format(key, value))
            else:
                rows.append("{}: {:0.4f}".format(key, value))
        return str.join("\n", rows)

    def export_model(self):
        output_path = self._run_prefix + '.trainedModel.joblib'
        with utils.DescriptionLoader.context("continuous/supervised/training/Train/export_model",
                                             output_path=output_path):
            joblib.dump(self._best_algorithm, output_path)

    def export_predictions(self):
        output_columns = ["ID", "PHENO_REPORTED", "PHENO_PREDICTED"]

        train_predicted_values = self._best_algorithm.predict(self._x_train)
        results = pd.DataFrame(zip(self._ids_train, self._y_train, train_predicted_values), columns=output_columns)
        output_path = self._run_prefix + '.trainedModel_trainingSample_Predictions.csv'

        with utils.DescriptionLoader.context("continuous/supervised/training/Train/export_predictions/train_data",
                                             output_path=output_path, data=results.head()):
            results.to_csv(output_path, index=False)

        test_predicted_values = self._best_algorithm.predict(self._x_test)
        results = pd.DataFrame(zip(self._ids_test, self._y_test, test_predicted_values), columns=output_columns)
        output_path = self._run_prefix + '.trainedModel_withheldSample_Predictions.csv'

        with utils.DescriptionLoader.context("continuous/supervised/training/Train/export_predictions/test_data",
                                             output_path=output_path, data=results.head()):
            results.to_csv(output_path, index=False)

        output_path = self._run_prefix + '.trainedModel_withheldSample_regression.png'
        reg_model = sm.ols(formula='PHENO_REPORTED ~ PHENO_PREDICTED', data=results)
        fitted = reg_model.fit()
        with utils.DescriptionLoader.context("continuous/supervised/training/Train/export_predictions/plot",
                                             output_path=output_path, data=fitted.summary()):
            sns_plot = sns.regplot(data=results, y="PHENO_REPORTED", x="PHENO_PREDICTED", scatter_kws={"color": "cyan"},
                                   line_kws={"color": "purple"})

            sns_plot.figure.savefig(output_path, dpi=600)

    def save_algorithm_results(self, output_prefix):
        output_path = output_prefix + '.training_withheldSamples_performanceMetrics.csv'
        with utils.DescriptionLoader.context("continuous/supervised/training/Train/save_algorithm_results",
                                             output_path=output_path,
                                             data=self.log_table.describe()):
            self.log_table.to_csv(output_path, index=False)

    def save_best_algorithm(self, output_prefix):
        output_path = output_prefix + '.best_algorithm.txt'
        with utils.DescriptionLoader.context("continuous/supervised/training/Train/save_best_algorithm",
                                             output_path=output_path, best_algorithm=self._best_algorithm_name):
            with open(output_path, 'w') as fp:
                fp.write(self._best_algorithm_name)
