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

import random
import numpy as np
import pandas as pd
import joblib
from statsmodels.stats import outliers_influence


# Define the VIF class to be used in munging
class vif:
    def __init__(self, iterations, vif_threshold, df, chunk_size, run_prefix):
        self.iterations = iterations
        self.threshold = vif_threshold
        self.df = df
        self.id = df['ID']
        self.pheno = df['PHENO']
        self.cleaned_df = None
        self.chunk_size = chunk_size
        self.run_prefix = run_prefix

    def check_df(self):
        """
        check_df takes in dataframe as an argument and strips it of missing values and non-numerical information.
        ### Arguments:
            df {pandas dataframe} -- A dataframe 
        ### Returns:
            cleaned_df {pandas dataframe} -- A cleaned dataframe with no NA values and only numerical values 
        """

        df = self.df

        print("Stripping erroneous space, dropping non-numeric columns...")
        df.columns = df.columns.str.strip()

        print("Drop any rows where at least one element is missing...")
        # Convert any infinite values to NaN prior to dropping NAs
        df.replace([np.inf, -np.inf], np.nan)
        df.dropna(how='any', inplace=True)

        print("Keeping only numerical columns...")
        int_cols = \
            df = df._get_numeric_data()

        print("Checking datatypes...")
        data_type = df.dtypes

        # Subset df to include only relevant numerical types
        int_cols = df.select_dtypes(include=["int", "int16", "int32", "int64", "float",
                                             "float16", "float32", "float64"]).shape[1]

        original_df = df
        self.original_df = original_df

        print("Sampling 100 rows at random to reduce memory overhead...")
        cleaned_df = df.sample(n=100).copy().reset_index()
        cleaned_df.drop(columns=["index"], inplace=True)

        print("Dropping columns that are not SNPs...")
        cleaned_df.drop(columns=['PHENO'], axis=1, inplace=True)
        print("Dropped!")

        print("Cleaned!")
        self.cleaned_df = cleaned_df
        return cleaned_df

    def randomize_chunks(self):
        chunk_size = self.chunk_size
        cleaned_df = self.cleaned_df

        """
        randomize_chunks takes in a cleaned dataframe's column names, randomizes them, 
        and spits out randomized, chunked dataframes with only SNPs for the VIF calculation later
        ### Arguments:
            cleaned_df {pandas dataframe} -- A cleaned dataframe 
            chunk_size {int} -- Desired size of dataframe chunked (default=100)
        ### Returns:
            list_chunked_dfs {list dfs} -- A cleaned, randomized list of dataframes with only SNPs as columns
        """

        print("Shuffling columns...")
        col_names_list = cleaned_df.columns.values.tolist()
        col_names_shuffle = random.sample(col_names_list, len(col_names_list))
        cleaned_df = cleaned_df[col_names_shuffle]
        print("Shuffled!")

        print("Generating chunked, randomized dataframes...")
        chunked_list = [col_names_shuffle[i * chunk_size:(i + 1) * chunk_size] for i in
                        range((len(col_names_shuffle) + chunk_size - 1) // chunk_size)]
        df_list = []
        for each_list in chunked_list:
            temp_df = cleaned_df[each_list].astype(float)
            df_list.append(temp_df.copy())

        no_chunks = len(df_list)
        print(f"The number of dataframes you have moving forward is {no_chunks}")
        print("Complete!")
        return df_list

    def calculate_vif(self, df_list):
        """
        calculate_vif takes in an list of randomized dataframes and removes any variables
        that is greater than the specified threshold (default=5.0). This is to combat 
        multicolinearity between the variables. The function then returns a fully VIF-filtered
        dataframe.
        ### Arguments:
            df_list {list dfs} -- A list of cleaned, randomized pandas dataframes 
            threshold {float} -- Cut-off for dropping following the VIF calculation (default=5.0)
        ### Returns:
            glued_df {pandas df} -- A complete VIF-filtered dataframe 
        """
        threshold = self.threshold

        dropped = True
        print(f"Dropping columns with a VIF threshold greater than {threshold}")

        for df in df_list:
            while dropped:
                # Loop until all variables in dataset have a VIF less than the threshold 
                variables = df.columns
                dropped = False
                vif = []

                # Changed to look at indexing 
                # Added simple joblib parallelization
                vif = joblib.Parallel(n_jobs=5)(
                    joblib.delayed(outliers_influence.variance_inflation_factor)(df[variables].values, df.columns.get_loc(var)) for var in
                    variables)

                max_vif = max(vif)

                if np.isinf(max_vif):
                    maxloc = vif.index(max_vif)
                    print(f'Dropping "{df.columns[maxloc]}" with VIF > {threshold}')
                    dropped = True

                if max_vif > threshold:
                    maxloc = vif.index(max_vif)
                    print(f'Dropping "{df.columns[maxloc]}" with VIF = {max_vif:.2f}')
                    df.drop([df.columns.tolist()[maxloc]], axis=1, inplace=True)
                    dropped = True

        print("\nVIF calculation on all chunks complete! \n")

        print("Gluing the dataframe back together...")
        glued_df = pd.concat(df_list, axis=1)
        print("Full VIF-filtered dataframe generated!")

        return glued_df

    def iterate(self, checked):
        """
        The iterate function specifies a number of times to iterate through the shuffling 
        and VIF filtering functions 
        ### Arguments:
            number {int} -- An integer specifying the number of iterations to perform (default=5)
        ### Returns:
            features_toKeep {list} -- A list of features to keep, extracted from the final iteration
        """

        iterations = self.iterations

        for iteration in range(iterations):
            print(f"""
                \n\n
                Iteration {iteration + 1}
                \n\n
                """)
            rando = self.randomize_chunks()
            checked = self.calculate_vif(rando)

        # When done, make list of features to keep 
        features_toKeep = checked.columns.values.tolist()

        print(f"""
        \n\n
            Iterations Complete!
        \n\n
        """)
        return features_toKeep, checked

    def vif_calculations(self):
        outfile_h5 = self.run_prefix + ".dataForML.h5"
        checked = self.check_df()
        features, complete_vif = self.iterate(checked)

        # Return the original dataframe with the features to keep 
        complete_vif_original_df = self.original_df[features]
        complete_vif_original_df['ID'] = self.id
        complete_vif_original_df['PHENO'] = self.pheno
        complete_vif_original_df.to_hdf(outfile_h5, key='dataForML')

        return complete_vif_original_df
