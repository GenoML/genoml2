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

# Import the necessary packages
from pathlib import Path
import genoml.discrete.utils as discrete_utils

class test:
    def __init__(self, df, loaded_model, run_prefix):
        self.df = df
        path = Path(run_prefix).joinpath("Test")
        if not path.is_dir():
            path.mkdir()
        self.run_prefix = path
        self.algo = loaded_model
    
    def prep_df(self):
        print("")
        print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
        print("#"*70)
        print(self.df.describe())
        print("#"*70)
        print("")

        # Save out and drop the PHENO and sample ID columns 
        y_test = self.df.PHENO
        X_test = self.df.drop(columns=['PHENO'])
        IDs_test = X_test.ID
        X_test = X_test.drop(columns=['ID'])

        # Save variables to use globally within the class 
        self.y_test = y_test
        self.X_test = X_test
        self.IDs_test = IDs_test

    def plot_results(self, save=False):
        # Issue #24: RandomForestClassifier is finicky - can't recalculate moving forward like the other
        self.algo.fit(self.X_test, self.y_test)
        plot_path = self.run_prefix.joinpath('testModel_withheldSample_ROC.png')
        ground_truth = self.y_test.values
        predictions = self.algo.predict(self.X_test)
        discrete_utils.ROC(save, plot_path, ground_truth, predictions)
        discrete_utils.precision_recall_plot(save, plot_path, ground_truth, predictions)

    def export_prediction_data(self):
        test_out = discrete_utils.export_prediction_tables(
            self.algo,
            self.y_test,
            self.X_test,
            self.IDs_test,
            self.run_prefix.joinpath('tunedModel_withheldSample_testingPredictions.csv'),
        )

        discrete_utils.export_prob_hist(
            test_out,
            self.run_prefix.joinpath('tunedModel_withheldSample_testingProbabilities'),
        )

    def additional_sumstats(self):
        print("")
        print("#"*70)
        print("Some additional summary stats logging from your application of your model to the test dataset.")
        print("")

        log_outfile = self.run_prefix.joinpath('tunedModel_validationCohort_allCasesControls_performanceMetrics.csv')
        log_table = discrete_utils.summary_stats(
            self.algo,
            self.y_test,
            self.X_test,
        )
        log_table.to_csv(log_outfile, index=False)

        print("")
        print("#"*70)
        print("")
        print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
        print("#"*70)
        print(log_table)
        print("#"*70)
        print("")
