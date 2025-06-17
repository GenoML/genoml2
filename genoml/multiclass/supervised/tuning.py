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
import genoml.multiclass.utils as multiclass_utils
import joblib
import sys
from pathlib import Path
from genoml import utils
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier


class Tune:
    @utils.DescriptionLoader.function_description("info", cmd="Multiclass Supervised Tuning")
    def __init__(self, run_prefix, metric_tune, max_iter, cv_count):
        utils.DescriptionLoader.print(
            "tuning/info",
            python_version=sys.version,
            run_prefix=run_prefix,
            max_iter=max_iter,
            cv_count=cv_count,
        )

        df = utils.read_munged_data(run_prefix, "train")
        model_path = Path(run_prefix).joinpath('model.joblib')

        dict_hyperparams = utils.get_tuning_hyperparams("multiclass")

        ### TODO: Can "metric_tune" be anything other than the two listed options?
        if metric_tune == "AUC":
            self._scoring_metric = metrics.make_scorer(metrics.roc_auc_score, needs_proba=True, multi_class="ovr")
        elif metric_tune == "Balanced_Accuracy":
            self._scoring_metric = metrics.make_scorer(metrics.balanced_accuracy_score, needs_proba=False)

        self._run_prefix = Path(run_prefix).joinpath("Tune")
        if not self._run_prefix.is_dir():
            self._run_prefix.mkdir()
        self._max_iter = max_iter
        self._cv_count = cv_count
        self._y_tune = pd.get_dummies(df.PHENO)
        self._ids_tune = df.ID
        self._x_tune = df.drop(columns=['PHENO', 'ID'])
        self._algorithm = joblib.load(model_path)
        self._hyperparameters = dict_hyperparams[self._algorithm.estimator.__class__.__name__]
        self._cv_tuned = None
        self._cv_baseline = None
        self._cv_results = None
        self._algorithm_tuned = None
        self._num_classes = None
        self._y_pred = None
        self._algorithm_name = None

        # Communicate to the user the best identified algorithm 
        print(f"From previous analyses in the training phase, we've determined that the best "
              f"algorithm for this application is {self._algorithm.estimator.__class__.__name__}... "
              f"so let's tune it up and see what gains we can make!")


    def tune_model(self):
        """ Determine best-performing hyperparameters. """
        self._cv_results, self._algorithm_tuned = utils.tune_model(
            self._algorithm,
            self._x_tune,
            self._y_tune,
            self._hyperparameters,
            self._scoring_metric,
            self._max_iter,
            self._cv_count,
        )


    def report_tune(self):
        """ Save best-performing fine-tuning iterations. """
        utils.report_best_tuning(
            self._run_prefix, 
            self._cv_results, 
            10,
        )


    def summarize_tune(self):
        """ Report results for baseline and tuned models. """
        self._cv_baseline, self._cv_tuned = utils.sumarize_tune(
            self._run_prefix,
            self._algorithm, 
            self._algorithm_tuned, 
            self._x_tune, 
            self._y_tune, 
            self._scoring_metric, 
            self._cv_count, 
        )


    def compare_performance(self):
        """ Compare fine-tuned model with baseline model. """
        self._algorithm = next(utils.compare_tuning_performance(
            self._run_prefix, 
            self._cv_tuned, 
            self._cv_baseline, 
            self._algorithm_tuned, 
            self._algorithm, 
        ))
        self._y_pred = self._algorithm.predict_proba(self._x_tune)
        self._algorithm_name = self._algorithm.estimator.__class__.__name__


    def plot_results(self):
        """ Plot results from best-performing algorithm. """
        self._num_classes = multiclass_utils.plot_results(
            self._run_prefix,
            self._y_tune,
            self._y_pred,
            self._algorithm_name,
        )


    def export_prediction_data(self):
        """ Save results from best-performing algorithm. """
        multiclass_utils.export_prediction_data(
            self._run_prefix,
            self._y_tune,
            self._y_pred,
            self._ids_tune,
            self._num_classes,
        )
