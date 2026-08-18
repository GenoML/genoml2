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

import genoml.discrete.utils as discrete_utils
import sys
from genoml import utils
from genoml.models import get_candidate_algorithms
from pathlib import Path
from sklearn import model_selection


class Train:
    @utils.DescriptionLoader.function_description("info", cmd="Discrete Supervised Training")
    def __init__(self, prefix, metric_max, train_split, random_state):
        utils.DescriptionLoader.print(
            "training/info",
            python_version=sys.version,
            prefix=prefix,
            metric_max=metric_max,
        )

        if Path(prefix).joinpath("Munge").joinpath(f"train_dataset.h5").exists():
            df_train = utils.read_munged_data(Path(prefix).joinpath("Munge").joinpath(f"train_dataset.h5"))
            x_train, x_valid, y_train, y_valid = utils.train_valid_split(df_train, train_split, random_state)
            self._x_train = x_train.drop(columns=['ID'])
            self._x_valid = x_valid.drop(columns=['ID'])
            self._y_train = y_train
            self._y_valid = y_valid
            self._ids_train = x_train.ID
            self._ids_valid = x_valid.ID
        else:
            raise FileNotFoundError(
                f"No munged data found at {prefix}/Munge. Please run the munge step first."
            )

        candidate_algorithms = get_candidate_algorithms("discrete_supervised", random_state)

        self._column_names = [
            "Algorithm",
            "Runtime_Seconds",
            "AUC",
            "Accuracy",
            "Balanced_Accuracy",
            "Log_Loss",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
        ]
        self._prefix = Path(prefix).joinpath("Train")
        if not self._prefix.is_dir():
            self._prefix.mkdir()
        self._algorithms = {algorithm.__class__.__name__: algorithm for algorithm in candidate_algorithms}
        self._metric_max = metric_max
        self._best_algorithm = None
        self._best_algorithm_name = None
        self._log_table = []


    def compete(self):
        """ Compete the algorithms. """
        self._log_table, self._algorithms = utils.fit_algorithms(
            self._prefix,
            self._algorithms,
            self._x_train,
            self._y_train,
            self._x_valid,
            self._y_valid,
            self._column_names,
            discrete_utils.calculate_accuracy_scores,
        )


    def select_best_algorithm(self):
        """ Determine the best-performing algorithm. """
        # Drop those that have an accuracy less than 50%, balanced accuracy less than 50%, delta between sensitivity
        # and specificity greater than 0.85, sensitivity equal to 0 or 1, or specificity equal to 0 or 1.
        filtered_table = self._log_table[
            (self._log_table["AUC"] > 0.5)
            & (self._log_table["Balanced_Accuracy"] > 50)
            & (self._log_table["Sensitivity"].sub(self._log_table["Specificity"], axis=0).abs() < 0.85)
            & (self._log_table["Sensitivity"] != 0.0)
            & (self._log_table["Sensitivity"] != 1.0)
            & (self._log_table["Specificity"] != 0.0)
            & (self._log_table["Specificity"] != 1.0)
        ]

        # If for some reason ALL the algorithms are overfit...
        if filtered_table.empty:
            print('It seems as though all the algorithms are over-fit in some way or another... We will report the best algorithm based on your chosen metric instead and use that moving forward.')
            filtered_table = self._log_table

        self._best_algorithm, self._best_algorithm_name = utils.select_best_algorithm(
            filtered_table, 
            self._metric_max, 
            self._algorithms,
        )
        with open(self._prefix.parent.joinpath("algorithm.txt"), "w") as file:
            file.write(self._best_algorithm_name)


    def export_model(self):
        """ Save best-performing algorithm """
        utils.export_model(
            self._prefix.parent, 
            self._best_algorithm,
        )


    def plot_results(self):
        """ Plot results from best-performing algorithm. """
        discrete_utils.plot_results(
            self._prefix,
            self._y_valid.values,
            self._x_valid.values,
            self._best_algorithm,
            False,
        )


    def export_prediction_data(self):
        """ Save results from best-performing algorithm. """
        discrete_utils.export_prediction_data(
            self._prefix,
            self._best_algorithm,
            self._y_valid,
            self._x_valid,
            self._ids_valid,
            False,
            y_train = self._y_train,
            x_train = self._x_train,
            ids_train = self._ids_train,
        )
