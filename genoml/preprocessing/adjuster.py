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
import numpy as np
import statsmodels.api as sm 
import statsmodels.formula.api as smf
import statistics
import umap.umap_ as umap
from joblib import dump, load 
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

class adjuster:
    def __init__(self, run_prefix, df, target_columns, confounders, umap_reduce):
        self.run_prefix = run_prefix
        self.umap_reduce = umap_reduce
        self.target_columns = target_columns
        self.confounders = confounders
        
        df = self.run_prefix + ".dataForML.h5"
        self.munged_data = df

    def umap_reducer(self):
        target_data_df = pd.read_hdf(self.munged_data, 'dataForML')
        target_column_df = pd.read_csv(self.target_columns, names=['TARGETS'])
        confounders_df = pd.read_csv(self.confounders)

        if (self.umap_reduce == "yes"):
            IDs = confounders_df['ID']
            IDs_df = pd.DataFrame(IDs) 
            to_umap = confounders_df.drop(columns=['ID']) 

            reducer = umap.UMAP(random_state=153) 
            embedding = reducer.fit_transform(to_umap)

            embedding1 = pd.DataFrame(embedding[:,0])
            embedding2 = pd.DataFrame(embedding[:,1])

            out_data = pd.concat([IDs_df.reset_index(), embedding1.reset_index(drop=True), embedding2.reset_index(drop=True)], axis=1, ignore_index=True)
            out_data.columns = ['INDEX', 'ID', 'UMAP_embedding1', "UMAP_embedding2"]
            out_data = out_data.drop(columns=['INDEX'])

            # Plot 
            print(f"Exporting UMAP plot...")
            fig, ax = plt.subplots(figsize=(12,10))
            plt.scatter(embedding[:,0], embedding[:,1], cmap="cool")
            plt.title("Data Reduction to 2 Dimensions by UMAP", fontsize=18)
            plot_out = self.run_prefix + '.umap_plot.png'
            plt.savefig(plot_out, dpi=600)

            print(f"The UMAP plot has been exported and can be found here: {plot_out}")
            
            out_file = self.runplot_out = self.run_prefix + '.umap_data_reduction.csv'
            out_data.to_csv(out_file, index=False)

            print(f"The reduced UMAP 2 dimensions per sample .csv file can be found here: {out_file}")

            exported_reducer = reducer.fit(to_umap)
            algo_out = self.runplot_out = self.run_prefix + '.umap_clustering.joblib'
            dump(exported_reducer, algo_out)

            print(f"The UMAP .joblib  file can be found here: {algo_out}")

            confounders_df = out_data
            target_list = list(target_column_df['TARGETS'])

            confounder_list = list(confounders_df.columns[1:])
            columns_to_keep_list = list(target_data_df.columns)

            adjustments_df = target_data_df.merge(confounders_df, how='inner', on='ID', suffixes=['', '_y'])

            formula_for_confounders = ' + '.join(confounder_list)
            normalize_switch = "yes"

            if normalize_switch == "yes":
                for target in target_list:
                    current_target = str(target)
                    print(f"Normalizing the following feature: {current_target}")
                    
                    current_formula = current_target + " ~ " + formula_for_confounders
                    print(current_formula)
                    
                    target_model = smf.ols(formula=current_formula, data=adjustments_df).fit()
                    
                    if (normalize_switch == 'yes'):
                        adjustments_df['temp'] = pd.to_numeric(target_model.resid)
                        print(type(adjustments_df['temp']))
                        mean_scalar = adjustments_df['temp'].mean()
                        sd_scalar = adjustments_df['temp'].std()
                        adjustments_df[current_target] = (adjustments_df['temp'] - mean_scalar)/sd_scalar
                        adjustments_df.drop(columns=['temp'], inplace=True)
                    else:
                        adjustments_df[current_target] = pd.to_numeric(target_model.resid)

            adjusted_df = adjustments_df[columns_to_keep_list]

            outfile_h5 = self.run_prefix + ".dataForML.h5"
            adjusted_df.to_hdf(outfile_h5, key='dataForML', mode='w')

            print(f"The adjusted dataframe following UMAP reduction and normalization : {outfile_h5}")

        return adjusted_df
