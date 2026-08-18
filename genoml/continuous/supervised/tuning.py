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

import genoml.continuous.utils as continuous_utils
import joblib
import re
import sys
from genoml import utils
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import KFold


class Tune():
    @utils.DescriptionLoader.function_description("info", cmd="Continuous Supervised Tuning")
    def __init__(self, prefix, metric_tune, max_iter, cv_count, random_state):
        utils.DescriptionLoader.print(
            "tuning/info",
            python_version=sys.version,
            prefix=prefix,
            max_iter=max_iter,
            cv_count=cv_count,
        )

        if Path(prefix).joinpath("Munge").joinpath(f"train_dataset.h5").exists():
            self._is_using_outer_cv = False
            df_tune = utils.read_munged_data(Path(prefix).joinpath("Munge").joinpath(f"train_dataset.h5"))
            model_path = Path(prefix).joinpath("model.joblib")
            self._y_tune = df_tune.PHENO
            self._ids_tune = df_tune.ID.values
            self._x_tune = df_tune.drop(columns=["PHENO", "ID"])
            self._algorithm = joblib.load(model_path)
            algorithm_name = self._algorithm.__class__.__name__
        else:
            raise FileNotFoundError(
                f"No munged data found at {prefix}/Munge. Please run the munge step first."
            )

        if Path(prefix).joinpath("Munge").joinpath(f"train_dataset_fold1.h5").exists():
            self._is_using_outer_cv = True
            self._y_tune = [self._y_tune]
            self._ids_tune = [self._ids_tune]
            self._x_tune = [self._x_tune]
            self._algorithm = [self._algorithm]
            train_datasets = sorted(
                [f for f in Path(prefix).joinpath("Munge").iterdir() 
                if f.is_file() and f.name.startswith("train_dataset_fold")],
                key=lambda f: int(re.search(r'fold(\d+)', f.name).group(1))
            )
            for fold, train_dataset in enumerate(train_datasets):
                df_tune = utils.read_munged_data(train_dataset)
                self._y_tune.append(df_tune.PHENO)
                self._ids_tune.append(df_tune.ID.values)
                self._x_tune.append(df_tune.drop(columns=["PHENO", "ID"]))
                self._algorithm.append(joblib.load(model_path))
                algorithm_name = self._algorithm[0].__class__.__name__

        dict_hyperparams = utils.get_tuning_hyperparams("continuous", random_state)

        if metric_tune == "Explained_Variance":
            self._scoring_metric = metrics.make_scorer(metrics.explained_variance_score)
        elif metric_tune == "Root_Mean_Squared_Error":
            self._scoring_metric = metrics.make_scorer(metrics.mean_squared_error)
        elif metric_tune == "Median_Absolute_Error":
            self._scoring_metric = metrics.make_scorer(metrics.median_absolute_error)
        elif metric_tune == "R-Squared_Error":
            self._scoring_metric = metrics.make_scorer(metrics.r2_score)
        
        self._prefix = Path(prefix).joinpath("Tune")
        if not self._prefix.is_dir():
            self._prefix.mkdir()
        self._max_iter = max_iter
        self._random_state = random_state
        self._cv = KFold(n_splits=cv_count, shuffle=True, random_state=self._random_state)
            
        self._hyperparameters = dict_hyperparams[algorithm_name]
        self._cv_tuned = None
        self._cv_baseline = None
        self._cv_results = None
        self._algorithm_tuned = None
        self._tune_results = None
        self._y_predicted = None

        # Communicate to the user the best identified algorithm 
        print("From previous analyses in the training phase, we've determined that "
              f"the best algorithm for this application is {algorithm_name}... so "
              "let's tune it up and see what gains we can make!")


    def tune_model(self):
        """ Determine best-performing hyperparameters. """
        self._cv_results, self._algorithm_tuned = utils.tune_model(
            self._algorithm,
            self._x_tune,
            self._y_tune,
            self._hyperparameters,
            self._scoring_metric,
            self._max_iter,
            self._cv,
            self._random_state,
            self._is_using_outer_cv,
        )


    def report_tune(self):
        """ Save best-performing fine-tuning iterations. """
        utils.report_best_tuning(
            self._prefix, 
            self._cv_results, 
            10,
            self._is_using_outer_cv,
        )


    def summarize_tune(self):
        """ Report results for baseline and tuned models. """
        self._cv_baseline, self._cv_tuned = utils.summarize_tune(
            self._prefix,
            self._algorithm, 
            self._algorithm_tuned, 
            self._x_tune, 
            self._y_tune, 
            self._scoring_metric, 
            self._cv, 
            self._is_using_outer_cv,
        )
    

    def compare_performance(self):
        """ Compare tuned model with baseline model. """
        self._algorithm, _ = utils.compare_tuning_performance(
            self._prefix, 
            self._cv_tuned, 
            self._cv_baseline, 
            self._algorithm_tuned, 
            self._algorithm, 
            self._is_using_outer_cv,
            x = self._x_tune,
        )


    def export_prediction_data(self):
        """ Save results from best-performing algorithm. """
        continuous_utils.export_prediction_data(
            self._prefix,
            self._ids_tune,
            "tuning",
            self._algorithm,
            [y_tune.values for y_tune in self._y_tune] if self._is_using_outer_cv else self._y_tune.values,
            [x_tune.values for x_tune in self._x_tune] if self._is_using_outer_cv else self._x_tune.values,
            self._is_using_outer_cv,
        )
        