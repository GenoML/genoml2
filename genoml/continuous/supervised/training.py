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
import seaborn as sns
import statsmodels.formula.api as sm
import sys
from sklearn import model_selection
from genoml.models import get_candidate_algorithms
from genoml import utils
from genoml.continuous import utils as continuous_utils


class Train:
    @utils.DescriptionLoader.function_description("info", cmd="Continuous Supervised Training")
    def __init__(self, prefix, metric_max):
        utils.DescriptionLoader.print(
            "training/info",
            python_version=sys.version,
            prefix=prefix,
            metric_max=metric_max,
        )

        df = utils.read_munged_data(prefix, "train")

        y = df.PHENO
        x = df.drop(columns=['PHENO'])
        x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
            x, 
            y, 
            test_size=0.3, 
            random_state=42,
        )

        candidate_algorithms = get_candidate_algorithms("continuous_supervised")
        
        self._column_names = [
            "Algorithm", 
            "Runtime_Seconds",
            "Explained_Variance",
            "Mean_Squared_Error",
            "Median_Absolute_Error",
            "R-Squared_Error",
        ]
        self._run_prefix = Path(prefix).joinpath("Train")
        if not self._run_prefix.is_dir():
            self._run_prefix.mkdir()
        self._ids_train = x_train.ID
        self._ids_valid = x_valid.ID
        self._x_train = x_train.drop(columns=['ID'])
        self._x_valid = x_valid.drop(columns=['ID'])
        self._y_train = y_train
        self._y_valid = y_valid
        self._algorithms = {algorithm.__class__.__name__: algorithm for algorithm in candidate_algorithms}
        self._metric_max = metric_max
        self._best_algorithm = None
        self._log_table = []


    def compete(self):
        """ Compete the algorithms. """
        self._log_table = utils.fit_algorithms(
            self._run_prefix, 
            self._algorithms, 
            self._x_train, 
            self._y_train, 
            self._x_valid, 
            self._y_valid, 
            self._column_names, 
            continuous_utils.calculate_accuracy_scores,
        )


    def select_best_algorithm(self):
        """ Determine the best-performing algorithm. """
        self._best_algorithm = utils.select_best_algorithm(
            self._log_table, 
            self._metric_max, 
            self._algorithms,
        )
        with open(self._run_prefix.parent.joinpath("algorithm.txt"), "w") as file:
            file.write(self._best_algorithm.__class__.__name__)


    def export_model(self):
        """ Save best-performing algorithm. """
        utils.export_model(
            self._run_prefix.parent, 
            self._best_algorithm,
        )


    def export_prediction_data(self):
        """ Save results from best-performing algorithm. """
        continuous_utils.export_prediction_data(
            self._run_prefix, 
            self._ids_train, 
            "training",
            self._y_train, 
            self._best_algorithm.predict(self._x_train), 
            y_withheld = self._y_valid, 
            y_withheld_predicted = self._best_algorithm.predict(self._x_valid),
        )
