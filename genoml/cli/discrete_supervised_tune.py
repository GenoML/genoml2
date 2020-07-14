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

from genoml.discrete import supervised


def main(run_prefix, metric_tune, max_iter, cv_count):
    # TUNING
    # Create a dialogue with the user 
    print("Here is some basic info on the command you are about to run.")
    print("CLI argument info...")
    print(f"Working with the dataset and best model corresponding to prefix {run_prefix} the timestamp from the merge is the prefix in most cases.")
    print(f"You have chosen to tune the algorithms based on {metric_tune}.")
    print(f"Your maximum number of tuning iterations is {max_iter} and if you are concerned about runtime, make this number smaller.")
    print(f"You are running {cv_count} rounds of cross-validation, and again... if you are concerned about runtime, make this number smaller.")
    print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
    print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")

    print("")

    infile_h5 = run_prefix + ".dataForML.h5"
    df = pd.read_hdf(infile_h5, key = "dataForML")

    best_algo_name_in = run_prefix + '.best_algorithm.txt'
    best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
    best_algo = str(best_algo_df.iloc[0,0])

    # Communicate to the user the best identified algorithm 
    print(f"From previous analyses in the training phase, we've determined that the best algorithm for this application is {best_algo}... so let's tune it up and see what gains we can make!")

    # Tuning 
    ## This calls on the functions made in the tune class (tuning.py) at the genoml.discrete.supervised 
    model_tune = supervised.tune(df, run_prefix, max_iter, cv_count)
    
    # Returns algo, hyperparameters, and scoring_metric
    model_tune.select_tuning_parameters(metric_tune) 
    
    # Randomized search with CV to tune
    model_tune.apply_tuning_parameters() 
    
    # Summary of the top 10 iterations of the hyperparameter tune
    model_tune.report_tune()

    # Summary of the cross-validation 
    model_tune.summarize_tune()

    # Compares tuned performance to baseline to 
    model_tune.compare_performance()

    # Export the ROC curve 
    # model_tune.ROC()

    # Export the newly tuned predictions 
    model_tune.export_tuned_data()

    # Export the probabilites 
    model_tune.export_tune_hist_prob() 

    print("")
    print("End of tuning stage with GenoML.")
    print("")
