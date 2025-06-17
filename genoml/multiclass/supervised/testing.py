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
import genoml.multiclass.utils as multiclass_utils
import sys
from pathlib import Path
from genoml import utils


### TODO: Add functionality to apply models without having ground truth data
class Test:
    @utils.DescriptionLoader.function_description("info", cmd="Multiclass Supervised Testing")
    def __init__(self, prefix):
        utils.DescriptionLoader.print(
            "testing/info",
            python_version=sys.version,
            prefix=prefix,
        )

        df = utils.read_munged_data(prefix, "test")
        model_path = Path(prefix).joinpath("model.joblib")
        algorithm = joblib.load(model_path)

        self._run_prefix = Path(prefix).joinpath("Test")
        if not self._run_prefix.is_dir():
            self._run_prefix.mkdir()
        self._y_test = pd.get_dummies(df.PHENO)
        self._ids_test = df.ID
        x_test = df.drop(columns=['ID', 'PHENO'])
        self._y_pred = algorithm.predict_proba(x_test)
        self._algorithm_name = algorithm.estimator.__class__.__name__
        self.num_classes = None
        

    def plot_results(self):
        """ Plot results from best-performing algorithm. """
        self.num_classes = multiclass_utils.plot_results(
            self._run_prefix,
            self._y_test,
            self._y_pred,
            self._algorithm_name,
        )


    def export_prediction_data(self):
        """ Save results from best-performing algorithm. """
        multiclass_utils.export_prediction_data(
            self._run_prefix,
            self._y_test.values.argmax(axis=1),
            self._y_pred,
            self._ids_test,
            self.num_classes,
        )


    def additional_sumstats(self):
        """ Save performance metrics for testing data """
        log_table = pd.DataFrame(
            data=[[self._algorithm_name] + multiclass_utils._calculate_accuracy_scores(self._y_test, self._y_pred)], 
            columns=["Algorithm", "AUC", "Accuracy", "Balanced_Accuracy", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV"],
        )
        log_outfile = self._run_prefix.joinpath('performance_metrics.txt')
        log_table.to_csv(log_outfile, index=False, sep="\t")
