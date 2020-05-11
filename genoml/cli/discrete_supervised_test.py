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
import joblib
import pandas as pd

from genoml.discrete import supervised

def main(prefix, test_prefix, refModel_prefix):
    print("")
    print("Here is some basic info on the command you are about to run.")
    print("Python version info...")
    print(sys.version)

    # Print out the chosen CLI arguments 
    print("CLI argument info...")
    print(f"You are importing this test dataset: {test_prefix}.")
    print(f"You are applying the model saved here: {refModel_prefix}.")
    print(f"The results of this test application of your model will be saved in files with the given prefix: {prefix}.")
    print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")

    print("")

    # Specify prefix and dataframe variables to be passed into class
    run_prefix = prefix
    infile_h5 = test_prefix + ".dataForML.h5"
    df = pd.read_hdf(infile_h5, key = "dataForML")

    infile_model = refModel_prefix + ".joblib"
    loaded_model = joblib.load(infile_model)

    # Pass the arguments to the class 
    test = supervised.test(df, loaded_model, run_prefix)

    # Prep and show the dataframe
    test.prep_df()

    # Export the ROC
    test.export_ROC()

    # Export the tested data 
    test.export_tested_data()

    # Export the histograms
    test.export_histograms()

    # Export the additional summary stats
    test.additional_sumstats()

    # Thank the user
    print("")
    print("Let's shut everything down, thanks for testing your model with GenoML!")
    print("")

    
