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

import pathlib
import subprocess
# Import the necessary packages
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin

# Define the munging class
import genoml.dependencies


class Munging(object):
    def __init__(
        self,
        pheno_path: str,
        run_prefix="GenoML_data",
        impute_type="median",
        skip_prune="no",
        p_gwas: float = 0.001,
        addit_path: Optional[str] = None,
        gwas_path: Optional[str] = None,
        geno_path: Optional[str] = None,
        refColsHarmonize=None,
        r2_cutoff="0.5",
    ):
        self.pheno_path = pheno_path
        self.run_prefix = run_prefix

        if impute_type not in ["mean", "median"]:
            # Currently only supports mean and median
            raise KeyError(
                "The 2 types of imputation currently supportevd are 'mean' and 'median'"
            )
        self.impute_type = impute_type
        self.p_gwas = p_gwas

        self.r2 = r2_cutoff

        if skip_prune == "no":
            self.skip_prune = False
        elif skip_prune == "yes":
            self.skip_prune = True
        else:
            raise ValueError(
                f'`skip_prune` should be one of "yes" or "no", not {skip_prune}'
            )

        self.refColsHarmonize = refColsHarmonize

        # Reading in the phenotype file
        self.pheno_df = pd.read_csv(pheno_path, engine="c")

        # Raise an error and exit if the phenotype file is not properly formatted
        if not {"ID", "PHENO"}.issubset(self.pheno_df.columns):
            raise ValueError(
                "Error: It doesn't look as though your phenotype file is properly "
                "formatted.\n"
                "Did you check that the columns are 'ID' and 'PHENO' and that "
                "controls=0 and cases=1?"
            )

        # Typecase to read in the ID column as a string and the PHENO as an integer
        self.pheno_df["ID"] = self.pheno_df["ID"].astype(str)
        self.pheno_df["PHENO"] = self.pheno_df["PHENO"].astype(int)

        self.addit_path = addit_path
        self.gwas_path = gwas_path
        self.geno_path = geno_path
        if not addit_path:
            print("No additional non-genotypical features supplied.")
            # "No additional features as predictors? No problem, we'll stick to genotypes."
            self.addit_df = None
        else:
            self.addit_df = pd.read_csv(addit_path, engine="c")

        if not gwas_path:
            print("No GWAS data supplied.")
            # "So you don't want to filter on P values from external GWAS? No worries, we don't usually either (if the dataset is large enough)."
            self.gwas_df = None
        else:
            self.gwas_df = pd.read_csv(gwas_path, engine="c")

        if not geno_path:
            print(
                "So no genotypes? Okay, we'll just use additional features provided "
                "for the predictions."
            )
        else:
            print("Exporting genotype data")

        self.output_datafile = self.run_prefix + ".dataForML.h5"
        self.merged: Optional[pd.DataFrame] = None

    def plink_inputs(self):
        # Initializing some variables
        self.pheno_df.to_hdf(self.output_datafile, key="pheno", mode="w")
        raw_df = None
        addit_df = None

        if self.geno_path:
            # Shortcut for skipping crashes.
            processed_file = pathlib.Path("temp_genos.bed")
            if not processed_file.exists():
                self._run_external_plink_commands()
            g = read_plink1_bin("temp_genos.bed", ref="a0")
            g_pruned = g.drop(
                [
                    "fid",
                    "father",
                    "mother",
                    "gender",
                    "trait",
                    "chrom",
                    "cm",
                    "pos",
                    "a1",
                ]
            )

            g_pruned = g_pruned.set_index({"sample": "iid", "variant": "snp"})
            # g_pruned.values = g_pruned.values.astype("int")

            # swap pandas-plink genotype coding to match .raw format...more about that below:

            # for example, assuming C in minor allele, alleles are coded in plink .raw labels homozygous for minor allele as 2 and homozygous for major allele as 0:
            # A A   ->    0
            # A C   ->    1
            # C C   ->    2
            # 0 0   ->   NA

            # where as, read_plink1_bin flips these, with homozygous minor allele = 0 and homozygous major allele = 2
            # A A   ->    2
            # A C   ->    1
            # C C   ->    0
            # 0 0   ->   NA

            # two_idx = g_pruned.values == 2
            # zero_idx = g_pruned.values == 0
            #
            # g_pruned.values[two_idx] = 0
            # g_pruned.values[zero_idx] = 2

            g_pd = g_pruned.to_pandas()  # This is gonnabe too slow.
            g_pd.reset_index(inplace=True)
            raw_df = g_pd.rename(columns={"sample": "ID"})
            #    del raw_df.index.name
            #    del raw_df.columns.name

            # now, remove temp_genos
            # bash_rm_temp = "rm temp_genos.*"
            # print(bash_rm_temp)
            # subprocess.run(bash_rm_temp, shell=True)
            # Checking the impute flag and execute
            # Currently only supports mean and median
            raw_df = _fill_impute_na(self.impute_type, raw_df)
            raw_df.to_hdf(self.output_datafile, key="geno")
        # Checking the imputation of non-genotype features

        # Saving out the proper HDF5 file
        if self.addit_path:
            addit_df = self.munge_additional_features()
            addit_df.to_hdf(self.output_datafile, key="addit")

        # TODO: Can I get rid of this? Why are we writing then reading??
        # if self.geno_path and self.addit_path:
        #     pheno = pd.read_hdf(self.output_datafile, key="pheno")
        #     geno = pd.read_hdf(self.output_datafile, key="geno")
        #     addit = pd.read_hdf(self.output_datafile, key="addit")
        #     temp = pd.merge(pheno, addit, on="ID", how="inner")
        #     # merged = pd.merge(temp, geno, on="ID", how="inner")
        # elif self.geno_path and not self.addit_path:
        #     pheno = pd.read_hdf(self.output_datafile, key="pheno")
        #     geno = pd.read_hdf(self.output_datafile, key="geno")
        #     # merged = pd.merge(pheno, geno, on="ID", how="inner")
        # elif not self.geno_path and self.addit_path:
        #     pheno = pd.read_hdf(self.output_datafile, key="pheno")
        #     addit = pd.read_hdf(self.output_datafile, key="addit")
        #     # merged = pd.merge(pheno, addit, on="ID", how="inner")
        merged = _merge_dfs([self.pheno_df, raw_df, addit_df], col_id="ID")

        self.merged = self.harmonize_refs(merged)
        self.merged.to_hdf(self.output_datafile, key="dataForML")

        features_list = self.merged.columns.tolist()

        features_listpath = f"{self.run_prefix}.list_features.txt"
        with open(features_listpath, "w") as f:
            for feature in features_list:
                f.write("%s\n" % feature)

        print(
            f"An updated list of {len(features_list)} features, including ID and PHENO,"
            f" that is in your munged dataForML.h5 file can be found here "
            f"{features_listpath}"
            "\n"
            f"Your .dataForML file that has been fully munged can be found here: "
            f"{self.output_datafile}"
        )

        return merged

    def _run_external_plink_commands(self) -> None:
        """Runs the external plink commands from the command line."""
        if not self.geno_path:
            return
        cmds_a, cmds_b = get_bash_scripts(
            self.skip_prune, self.geno_path, self.run_prefix, self.r2
        )
        if self.gwas_path:
            p_thresh = self.p_gwas
            gwas_df_reduced = self.gwas_df[["SNP", "p"]]
            snps_to_keep = gwas_df_reduced.loc[(gwas_df_reduced["p"] <= p_thresh)]
            outfile = self.run_prefix + ".p_threshold_variants.tab"
            snps_to_keep.to_csv(outfile, index=False, sep="\t")
            print(f"Your candidate variant list is right here: {outfile}.")
            cmds = cmds_b
        else:
            cmds = cmds_a
        print(
            "A list of variants and the allele being counted in the dosages "
            "(usually the minor allele) can be found here: "
            f"{self.run_prefix}.variants_and_alleles.tab"
        )
        for cmd in cmds:
            subprocess.run(cmd, shell=True)

    def munge_additional_features(self) -> Optional[pd.DataFrame]:
        """Munges additional features and cleans up statistically insignificant data.

        * Z-Scales the features.
        * Remove any columns with a standard deviation of zero
        """
        if not self.addit_path:
            return None
        addit_df = _fill_impute_na(self.impute_type, self.addit_df)

        # Remove the ID column
        cols = [col for col in addit_df.columns if not col == "ID"]
        addit_df.drop(labels="ID", axis=0, inplace=True)

        # Remove any columns with a standard deviation of zero
        print(
            "Removing any columns that have a standard deviation of 0 prior to "
            "Z-scaling..."
        )
        std = addit_df.std()
        if any(std == 0.0):
            print(
                "\n"
                "Looks like there's at least one column with a standard deviation "
                "of 0. Let's remove that for you..."
                "\n"
            )
            addit_keep = addit_df.drop(
                std[std == 0.0].index.values, axis=1
            )
            addit_keep_list = list(addit_keep.columns.values)

            addit_df = addit_df[addit_keep_list]

            addit_keep_list.remove("ID")
            removed_list = np.setdiff1d(cols, addit_keep_list)
            for removed_column in range(len(removed_list)):
                print(f"The column {removed_list[removed_column]} was removed")

            cols = addit_keep_list

        # Z-scale the features
        print("Now Z-scaling your non-genotype features...\n")
        for col in cols:
            if (addit_df[col].min() == 0.0) and (addit_df[col].max() == 1.0):
                print(
                    f"{col} is likely a binary indicator or a proportion and will not "
                    "be scaled, just + 1 all the values of this variable and rerun to "
                    "flag this column to be scaled.",
                )
            else:
                addit_df[col] = (addit_df[col] - addit_df[col].mean()) / addit_df[
                    col
                ].std(ddof=0)

        print(
            "\n"
            "You have just Z-scaled your non-genotype features, putting everything on "
            "a numeric scale similar to genotypes.\n"
            "Now your non-genotype features might look a little closer to zero "
            "(showing the first few lines of the left-most and right-most columns)..."
        )
        print("#" * 70)
        print(addit_df.describe())
        print("#" * 70)
        return addit_df

    def harmonize_refs(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonizes data columns with an external reference file.??????????

        > Checking the reference column names flag
        > If this is a step that comes after harmonize, then a .txt file with columns
        > to keep should have been produced. This is a list of column names from the
        > reference dataset that the test dataset was harmonized against. We want to
        > compare apples to apples, so we will only keep the column names that match.
        """
        if not self.refColsHarmonize:
            return merged_df
        print(
            "\n"
            f"Looks like you are munging after the harmonization step. Great! We will "
            f"keep the columns generated from your reference dataset from that "
            f"harmonize step that was exported to this file: {self.refColsHarmonize}\n"
        )
        with open(self.refColsHarmonize, "r") as refCols_file:
            ref_column_names_list = refCols_file.read().splitlines()

        # Keep the reference columns from the test dataset if found in test data
        matching_cols = merged_df[
            np.intersect1d(merged_df.columns, ref_column_names_list)
        ]

        # Make a list of final features that will be included in the model
        # This will be used again when remunging the reference dataset
        matching_cols_list = matching_cols.columns.values.tolist()

        # Save out the final list
        intersecting_cols_outfile = f"{self.run_prefix}.finalHarmonizedCols_toKeep.txt"

        with open(intersecting_cols_outfile, "w") as f:
            for col in matching_cols_list:
                f.write(f"{col}\n")

        print(
            "A final list of harmonized columns between your reference and test "
            f"dataset has been generated here: {intersecting_cols_outfile}\n"
            "Use this to re-train your reference dataset in order to move on to "
            "testing."
        )
        return matching_cols


def get_bash_scripts(
    skip_prune: bool, geno_path: str, run_prefix: str, r2: Optional[str] = None
) -> Tuple[List, List]:
    """Gets the PLINK bash scripts to be run from CLI."""
    plink_exec = genoml.dependencies.check_plink()
    if not skip_prune:
        # Set the bashes
        bash1a = f"{plink_exec} --bfile {geno_path} --indep-pairwise 1000 50 {r2}"
        bash1b = (
            f"{plink_exec} --bfile {geno_path} --extract "
            f"{run_prefix}.p_threshold_variants.tab --indep-pairwise 1000 50 {r2}"
        )
        # may want to consider outputting temp_genos to dir in run_prefix
        bash2 = (
            f"{plink_exec} --bfile {geno_path} --extract plink.prune.in --make-bed"
            f" --out temp_genos"
        )
        bash3 = f"cut -f 2,5 temp_genos.bim > {run_prefix}.variants_and_alleles.tab"
        bash4 = "rm plink.log"
        bash5 = "rm plink.prune.*"
        #    bash6 = "rm " + self.run_prefix + ".log"
        # Set the bash command groups
        cmds_a = [bash1a, bash2, bash3, bash4, bash5]
        cmds_b = [bash1b, bash2, bash3, bash4, bash5]
    else:
        bash1a = f"{plink_exec} --bfile {geno_path}"
        bash1b = (
            f"{plink_exec} --bfile {geno_path} --extract "
            f"{run_prefix}.p_threshold_variants.tab"
        )
        # may want to consider outputting temp_genos to dir in run_prefix
        bash2 = f"{plink_exec} --bfile {geno_path} --make-bed --out temp_genos"
        bash3 = f"cut -f 2,5 temp_genos.bim > {run_prefix}.variants_and_alleles.tab"
        bash4 = "rm plink.log"

        # Set the bash command groups
        cmds_a = [bash1a, bash2, bash3, bash4]
        cmds_b = [bash1b, bash2, bash3, bash4]
    return cmds_a, cmds_b


def _fill_impute_na(impute_type: str, df: pd.DataFrame) -> pd.DataFrame:
    """Imputes the NA fields for a dataframe."""
    if impute_type.lower() == "mean":
        df = df.fillna(df.mean())
    elif impute_type.lower() == "median":
        df = df.fillna(df.median())
    print(
        "\n"
        "You have just imputed your genotype features, covering up NAs with "
        f"the column {impute_type} so that analyses don't crash due to "
        "missing data.\n"
        "Now your genotype features might look a little better (showing the "
        "first few lines of the left-most and right-most columns)..."
    )
    print("#" * 70)
    print(df.describe())
    print("#" * 70)
    print("")
    return df


def _merge_dfs(dfs: List[pd.DataFrame], col_id: str) -> pd.DataFrame:
    merged = None
    for df in dfs:
        if df is None:
            continue
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=col_id, how="inner")
    return merged
