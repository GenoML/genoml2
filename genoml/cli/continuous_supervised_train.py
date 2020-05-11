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

import sys
import pandas as pd
import numpy as np

from genoml.continuous import supervised
from genoml import utils


@utils.DescriptionLoader.function_description("cli/continuous_supervised_train")
def main(run_prefix, export_predictions, matchingCols):
    utils.DescriptionLoader.print("cli/continuous_supervised_train/info",
                                  python_version=sys.version, prefix=run_prefix)

    input_path = run_prefix + ".dataForML.h5"
    with utils.DescriptionLoader.context(
            "cli/continuous_supervised_train/input", path=input_path):
        df = pd.read_hdf(input_path, key="dataForML")
    
    if (matchingCols != None):
        print(f"Looks like you are retraining your reference file. We are using the harmonized columns you provided here: {matchingCols}")
        print(f"Note that you might have different/less features than before, given this was harmonized between training and test dataset, and might mean your model now performs worse...")

        with open(matchingCols, 'r') as matchingCols_file:
                matching_column_names_list = matchingCols_file.read().splitlines()
        
        # Keep only the columns found in the file 
        df = df[np.intersect1d(df.columns, matching_column_names_list)]
        
    model = supervised.train(df, run_prefix)
    model.summary()
    model.compete()
    model.export_model()
    model.export_predictions()
    model.save_algorithm_results(run_prefix)
    model.save_best_algorithm(run_prefix)
