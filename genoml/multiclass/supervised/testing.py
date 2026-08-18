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

import genoml.multiclass.utils as multiclass_utils
import joblib
import pandas as pd
import re
import sys
from genoml import utils
from pathlib import Path


class Test:
    @utils.DescriptionLoader.function_description("info", cmd="Multiclass Supervised Testing")
    def __init__(self, prefix):
        utils.DescriptionLoader.print(
            "testing/info",
            python_version=sys.version,
            prefix=prefix,
        )

        if Path(prefix).joinpath("Munge").joinpath(f"test_dataset_fold1.h5").exists():
            self._is_using_outer_cv = True
            self._y_test = []
            self._ids_test = []
            self._x_test = []
            self._algorithm = []
            test_datasets = sorted(
                [f for f in Path(prefix).joinpath("Munge").iterdir() 
                if f.is_file() and f.name.startswith("test_dataset_fold")],
                key=lambda f: int(re.search(r'fold(\d+)', f.name).group(1))
            )
            for fold, test_dataset in enumerate(test_datasets):
                df_test = utils.read_munged_data(test_dataset)
                self._y_test.append(df_test.PHENO)
                self._ids_test.append(df_test.ID)
                self._x_test.append(df_test.drop(columns=["PHENO", "ID"]))
                self._algorithm.append(joblib.load(Path(prefix).joinpath(f"model_fold{fold+1}.joblib")))
            self._algorithm_name = utils.get_algorithm_name(self._algorithm[0])
        elif Path(prefix).joinpath("Munge").joinpath(f"test_dataset.h5").exists():
            self._is_using_outer_cv = False
            df_test = utils.read_munged_data(Path(prefix).joinpath("Munge").joinpath(f"test_dataset.h5"))
            self._y_test = df_test.PHENO
            self._ids_test = df_test.ID
            self._x_test = df_test.drop(columns=["PHENO", "ID"])
            self._algorithm = joblib.load(Path(prefix).joinpath("model.joblib"))
            self._algorithm_name = utils.get_algorithm_name(self._algorithm)
        else:
            raise FileNotFoundError(
                f"No munged data found at {prefix}/Munge. Please run the munge step first."
            )

        self._prefix = Path(prefix).joinpath("Test")
        if not self._prefix.is_dir():
            self._prefix.mkdir()

        self.num_classes = None
        

    def plot_results(self):
        """ Plot results from best-performing algorithm. """
        self._num_classes = multiclass_utils.plot_results(
            self._prefix,
            [pd.get_dummies(y_test).values for y_test in self._y_test] if self._is_using_outer_cv else pd.get_dummies(self._y_test).values, 
            [x_test.values for x_test in self._x_test] if self._is_using_outer_cv else self._x_test.values, 
            self._algorithm,
            self._is_using_outer_cv,
            is_testing=True,
        )


    def export_prediction_data(self):
        """ Save results from best-performing algorithm. """
        multiclass_utils.export_prediction_data(
            self._prefix,
            self._algorithm,
            [pd.get_dummies(y_test) for y_test in self._y_test] if self._is_using_outer_cv else pd.get_dummies(self._y_test), 
            self._x_test, 
            self._ids_test,
            self._num_classes,
            self._is_using_outer_cv,
            is_testing=True,
        )


    def additional_sumstats(self):
        """ Save performance metrics for testing data """
        multiclass_utils.additional_sumstats(
            self._algorithm_name, 
            self._y_test,
            self._x_test,
            self._algorithm,
            self._prefix,
            self._is_using_outer_cv,
            is_testing=True,
        )
