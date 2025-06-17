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
import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path


### TODO: Add docstrings
def define_geno_bash_cmds(run_prefix, skip_prune, plink_exec, geno_path, r2, gwas_paths, pheno_path):
    tmp_prefix = str(run_prefix.joinpath("temp_genos"))
    extract_vars_path = str(run_prefix.joinpath('p_threshold_variants.tab'))
    var_alleles_path = str(run_prefix.joinpath('variants_and_alleles.tab'))
    var_path = str(run_prefix.joinpath('variants.txt'))

    # Default to using p-files if provided by the user, otherwise use b-files
    if Path(geno_path + ".pvar").exists() and Path(geno_path + ".pgen").exists() and Path(geno_path + ".psam").exists():
        print(f"A list of variants and the allele being counted in the dosages (usually the minor allele) can "
              f"be found here: {var_alleles_path}")
        tmp_ids_path = _create_ids_to_keep(geno_path, pheno_path, 2, run_prefix)
        return _define_pfile_cmds(skip_prune, plink_exec, geno_path, r2, tmp_prefix, extract_vars_path, var_alleles_path, var_path, gwas_paths, tmp_ids_path)
    
    elif Path(geno_path + ".bim").exists() and Path(geno_path + ".bed").exists() and Path(geno_path + ".fam").exists():
        print(f"A list of variants and the allele being counted in the dosages (usually the minor allele) can "
              f"be found here: {var_alleles_path}")
        tmp_ids_path = _create_ids_to_keep(geno_path, pheno_path, 1, run_prefix)
        return _define_bfile_cmds(skip_prune, plink_exec, geno_path, r2, tmp_prefix, extract_vars_path, var_alleles_path, var_path, gwas_paths, tmp_ids_path)
    
    ### TODO: Add compatibility for VCF input
    else:
        raise FileNotFoundError(f"No valid genotype files found. Options indlude:\n\t"
                                f"{geno_path} + \".pvar\", \".pgen\", AND \".psam\", OR\n\t"
                                f"{geno_path} + \".bim\",  \".bed\",  AND \".fam\"")


def gwas_filter(run_prefix, gwas_paths, p_gwas):
    if len(gwas_paths) > 0:
        snps_to_keep = []
        outfile = run_prefix.joinpath("p_threshold_variants.tab")
        for gwas_path in gwas_paths:
            gwas_df = pd.read_csv(gwas_path, sep=None)
            if "SNP" in gwas_df.columns.values[0]:
                gwas_df.rename({gwas_df.columns.values[0]:"SNP"}, axis=1, inplace=True)
            if "p" in gwas_df.columns.values[0]:
                gwas_df.rename({gwas_df.columns.values[0]:"p"}, axis=1, inplace=True)
            gwas_df_reduced = gwas_df[['SNP', 'p']]
            snps_to_keep.append(gwas_df_reduced.loc[(gwas_df_reduced['p'] <= p_gwas)])
        snps_to_keep = pd.concat(snps_to_keep)
        snps_to_keep.sort_values('p', inplace=True)
        snps_to_keep.drop_duplicates(subset=['SNP'], inplace=True)
        snps_to_keep.to_csv(outfile, index=False, sep="\t")
        print(f"Your candidate variant list is right here: {outfile}.")
    else:
        print("So you don't want to filter on P values from external GWAS? No worries, we don't usually either (if the dataset is large enough).")


def read_pheno_file(pheno_path, data_type):
    pheno_df = pd.read_csv(pheno_path, sep=None, encoding="utf-8-sig")
    try:
        if not {'ID', 'PHENO'}.issubset(pheno_df.columns):
            raise ValueError("Error: It doesn't look as though your phenotype file is properly formatted. Did you check"
                             " that the columns are 'ID' and 'PHENO' and that controls=0 and cases=1?")
    except ValueError as ve:
        print(ve)
        sys.exit()
    pheno_df['ID'] = pheno_df['ID'].astype(str)
    if data_type == "c":
        pheno_df['PHENO'] = pheno_df['PHENO'].astype(float)
    else:
        pheno_df['PHENO'] = pd.Categorical(pheno_df['PHENO']).codes
    return pheno_df


def create_geno_df(run_prefix):
    df_geno = pd.read_csv(str(run_prefix.joinpath('temp_genos.raw')), sep=r"\s+", engine='c')
    cols = df_geno.columns.values
    cols.sort()
    df_geno.drop(["FID","PAT","MAT","SEX","PHENOTYPE"], axis=1, inplace=True)
    var_ids = df_geno.columns.values[1:]
    rename_dict = dict(zip(var_ids, [var_id.split("_")[0] for var_id in var_ids]))
    rename_dict["IID"] = "ID"
    df_geno.rename(rename_dict, inplace=True, axis=1)
    df_geno.rename_axis("variant", inplace=True, axis=1)
    bash_rm_temp = f"rm {str(run_prefix.joinpath('temp_genos.*'))}"
    print(bash_rm_temp)
    subprocess.run(bash_rm_temp, shell=True)

    return df_geno


def impute_df(df, impute_type, feature_type="genotype"):
    if impute_type.lower() == "mean":
        numeric_means = df.select_dtypes(include=[np.number]).mean()
        df = df.fillna(numeric_means)
    else:
        numeric_medians = df.select_dtypes(include=[np.number]).median()
        df = df.fillna(numeric_medians)

    print("")
    print(f"You have just imputed your {feature_type} features, covering up NAs with the column {impute_type} so that analyses don't crash due to missing data.")
    print(f"Now your {feature_type} features might look a little better (showing the first few lines of the left-most and right-most columns)...")
    print("#" * 70)
    print(df.describe())
    print("#" * 70)
    print("")

    return df


def normalize_cols(df):
    # Remove the ID column
    cols = list(df.columns)
    cols.remove('ID')
    df_numeric = df[cols]

    # Remove any columns with a standard deviation of zero
    print(f"Now Z-scaling your non-genotype features...")
    print(f"Removing any columns that have a standard deviation of 0 prior to Z-scaling...")

    if any(df[cols].std() == 0.0):
        print("")
        print(f"Looks like there's at least one column with a standard deviation of 0. Let's remove that for you...")
        print("")
        addit_keep = df_numeric.drop(df_numeric.std()[df_numeric.std() == 0.0].index.values, axis=1)
        addit_keep_list = list(addit_keep.columns.values)
        df_numeric = df_numeric[addit_keep_list]
        removed_list = np.setdiff1d(cols, addit_keep_list)
        for removed_column in range(len(removed_list)):
            print(f"The column {removed_list[removed_column]} was removed")
        cols = addit_keep_list

    for col in cols:
        if (df[col].min() == 0.0) and (df[col].max() == 1.0):
            print(f"{col} is likely a binary indicator or a proportion and will not be scaled, just + 1 all the values of this variable and rerun to flag this column to be scaled.")
        else:
            df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    print("")
    print("You have just Z-scaled your non-genotype features, putting everything on a numeric scale similar to genotypes.")
    print("Now your non-genotype features might look a little closer to zero (showing the first few lines of the left-most and right-most columns)...")
    print("#" * 70)
    print(df.describe())
    print("#" * 70)

    return df


def filter_harmonize(run_prefix, merged, ref_cols_harmonize):
    if ref_cols_harmonize is not None:
        print("")
        print(f"Looks like you are munging after the harmonization step. Great! We will keep the columns generated from your reference dataset from that harmonize step that was exported to this file: {ref_cols_harmonize}")
        print("")

        # Load list of column names from harmonization step
        with open(ref_cols_harmonize, 'r') as refCols_file:
            ref_column_names_list = refCols_file.read().splitlines()

        # Keep the reference columns from the test dataset if found in test data
        merged = merged[np.intersect1d(merged.columns, ref_column_names_list)]

    # Make a list of final features that will be included in the model after re-munging the reference dataset
    matching_cols_list = merged.columns.values.tolist()
    intersecting_cols_outfile = run_prefix.joinpath(f"list_features{'_harmonized' if ref_cols_harmonize is not None else ''}.txt")
    with open(intersecting_cols_outfile, 'w') as filehandle:
        for col in matching_cols_list:
            filehandle.write('%s\n' % col)
    print(f"An updated list of {len(matching_cols_list)} features, including ID and PHENO, that is in your munged dataForML.h5 file can be found here {intersecting_cols_outfile}")

    return merged
    

def _define_bfile_cmds(
    skip_prune, plink_exec, geno_path, r2, tmp_prefix, extract_vars_path, var_alleles_path, var_path, gwas_paths, tmp_ids_path,
):

    cmds = [
        f"{plink_exec} --bfile {geno_path} --keep {tmp_ids_path} --make-bed --out {tmp_prefix}",
        f"cut -f 2,5 {tmp_prefix}.bim > {var_alleles_path}",
        f"cut -f 2 {tmp_prefix}.bim > {var_path}",
        f"{plink_exec} --bfile {tmp_prefix} --export A --export-allele {var_alleles_path} --out {tmp_prefix}",
        f"rm plink2.log",
        f"rm {tmp_ids_path}",
    ]

    if not skip_prune:
        cmds.insert(0, f"{plink_exec} --bfile {geno_path} --indep-pairwise 1000 50 {r2}")
        cmds[1] += " --extract plink2.prune.in"
        cmds.append("rm plink2.prune.*")

    if len(gwas_paths) > 0:
        cmds[0] += f" --extract {extract_vars_path}"
    
    return cmds


def _define_pfile_cmds(
    skip_prune, plink_exec, geno_path, r2, tmp_prefix, extract_vars_path, var_alleles_path, var_path, gwas_paths, tmp_ids_path,
):

    cmds = [
        f"{plink_exec} --pfile {geno_path} --keep {tmp_ids_path} --make-pgen --out {tmp_prefix}",
        f"cut -f 3,5 {tmp_prefix}.pvar > {var_alleles_path}",
        f"cut -f 3 {tmp_prefix}.pvar > {var_path}",
        f"{plink_exec} --pfile {tmp_prefix} --export A --export-allele {var_alleles_path} --out {tmp_prefix}",
        f"rm plink2.log",
        f"rm {tmp_ids_path}",
    ]

    if not skip_prune:
        cmds.insert(0, f"{plink_exec} --pfile {geno_path} --keep {tmp_ids_path} --indep-pairwise 1000 50 {r2}")
        cmds[1] += " --extract plink2.prune.in"
        cmds.append("rm plink2.prune.*")

    if len(gwas_paths) > 0:
        cmds[0] += f" --extract {extract_vars_path}"

    return cmds


def _create_ids_to_keep(geno_path, pheno_path, plink_ver, run_prefix):
    """
    Remove IDs from genotype dataset that are not in the phenotype dataset.

    Args:
        geno_path (str): Path to genotype data.
        pheno_path (str): Path to phenotype data.
        plink_ver (int): 1 if using plink1.9, or 2 if using plink2.
        run_prefix (pathlib.Path): Path to results directory.

    :return: tmp_ids_path *(str)*: \n
        Temporary path to file with IDs being kept from the genomic dataset.
    """

    if plink_ver == 1:
        file_path = f"{geno_path}.fam"
        ids_geno = pd.read_csv(file_path, sep=r"\s+", header=None)[[0, 1]]
        ids_geno.rename(columns={0:"FID", 1:"IID"}, inplace=True)
    if plink_ver == 2:
        file_path = f"{geno_path}.psam"
        df_geno = pd.read_csv(file_path, sep=r"\s+")
        list_ids = [col for col in df_geno.columns.values if col in ["FID","#FID"]] + ["IID"]
        ids_geno = df_geno[list_ids]
    ids_pheno = list(pd.read_csv(pheno_path)["ID"])
    ids_overlap = ids_geno[ids_geno["IID"].isin(ids_pheno)]
    tmp_ids_path = str(run_prefix.joinpath(".tmp_ids_keep.txt"))
    ids_overlap.to_csv(tmp_ids_path, index=False, header=False, sep="\t")
    return tmp_ids_path


def merge_addit_data(df_merged, addit_path, impute_type):
    """
    Merge additional/clinical data with phenotype data.

    Args:
        df_merged (pandas.DataFrame): Predicted phenotype probabilities.
        addit_path (str): Path to additional/clinical data.
        impute_type (str): Imputation method to use.

    :return: df_merged *(pandas.DataFrame)*: \n
        Merged phenotype and additional/clinical data.
    """

    # Process non-genotype data if provided by the user
    if addit_path is not None:
        print("Processing non-genotype data")
        df_addit = pd.read_csv(addit_path, sep=None, encoding="utf-8-sig")
        df_addit = impute_df(df_addit, impute_type, feature_type="non-genotype")
        df_addit = normalize_cols(df_addit)
        df_merged = pd.merge(df_merged, df_addit, on='ID', how='inner')
    else:
        print("No additional features as predictors? No problem, we'll stick to genotypes.")
    
    return df_merged


def merge_geno_data(df_merged, geno_path, pheno_path, impute_type, run_prefix, gwas_paths, p_gwas, skip_prune, plink_exec, r2):
    """
    Merge genotype data with phenotype and additional/clinical data.

    Args:
        df_merged (pandas.DataFrame): Predicted phenotype probabilities.
        geno_path (str): Path to genotype data.
        pheno_path (str): Path to phenotype data.
        impute_type (str): Imputation method to use.
        run_prefix (pathlib.Path): Path to results directory.
        gwas_paths (list): Paths to GWAS summary statistics datasets.
        p_gwas (float): GWAS p-value threshold.
        skip_prune (bool): True if skipping LD pruning, otherwise False.
        plink_exec (str): Path to plink2 executable.
        r2 (str): R2 cutoff for pruning.

    :return: df_merged *(pandas.DataFrame)*: \n
        Merged phenotype, additional/clinical, and genotype data.
    """

    # Process genotype data if provided by user
    if geno_path is not None:
        print("Processing genotype data")
        gwas_filter(run_prefix, gwas_paths, p_gwas)
        cmds = define_geno_bash_cmds(run_prefix, skip_prune, plink_exec, geno_path, r2, gwas_paths, pheno_path)

        print("Running the following commands:")
        for cmd in cmds:
            print(cmd)
        print()
        
        for cmd in cmds:
            subprocess.run(cmd, shell=True)
        df_geno = create_geno_df(run_prefix)
        df_geno = impute_df(df_geno, impute_type)
        df_merged = pd.merge(df_merged, df_geno, on='ID', how='inner')
    else:
        print("So no genotypes? Okay, we'll just use additional features provided for the predictions.")
    
    return df_merged


def merge_geno_harmonize_data(df_merged, geno_path, pheno_path, impute_type, run_prefix, plink_exec):
    """
    Merge genotype data with phenotype and additional/clinical data.

    Args:
        df_merged (pandas.DataFrame): Predicted phenotype probabilities.
        geno_path (str): Path to genotype data.
        pheno_path (str): Path to phenotype data.
        impute_type (str): Imputation method to use.
        run_prefix (pathlib.Path): Path to results directory.
        plink_exec (str): Path to plink2 executable.

    :return: df_merged *(pandas.DataFrame)*: \n
        Merged phenotype, additional/clinical, and genotype data.
    """

    # Process genotype data if provided by user
    if geno_path is not None:
        tmp_prefix = str(run_prefix.joinpath("temp_genos"))
        var_alleles_path = str(run_prefix.joinpath('variants_and_alleles.tab'))
        var_path = str(run_prefix.joinpath('variants.txt'))

        # Default to using p-files if provided by the user, otherwise use b-files
        if Path(geno_path + ".pvar").exists() and Path(geno_path + ".pgen").exists() and Path(geno_path + ".psam").exists():
            tmp_ids_path = _create_ids_to_keep(geno_path, pheno_path, 2, run_prefix)
            cmds = [
                f"{plink_exec} --pfile {geno_path} --keep {tmp_ids_path} --extract {var_path} --make-pgen --out {tmp_prefix}",
                f"{plink_exec} --pfile {tmp_prefix} --export A --export-allele {var_alleles_path} --out {tmp_prefix}",
            ]
        
        elif Path(geno_path + ".bim").exists() and Path(geno_path + ".bed").exists() and Path(geno_path + ".fam").exists():
            tmp_ids_path = _create_ids_to_keep(geno_path, pheno_path, 1, run_prefix)
            cmds = [
                f"{plink_exec} --bfile {geno_path} --keep {tmp_ids_path} --extract {var_path} --make-bed --out {tmp_prefix}",
                f"{plink_exec} --bfile {tmp_prefix} --export A --export-allele {var_alleles_path} --out {tmp_prefix}",
            ]
        
        else:
            raise FileNotFoundError(f"No valid genotype files found. Options indlude:\n\t"
                                    f"{geno_path} + \".pvar\", \".pgen\", AND \".psam\", OR\n\t"
                                    f"{geno_path} + \".bim\",  \".bed\",  AND \".fam\"")

        cmds.append("rm plink2.log")
        cmds.append(f"rm {tmp_ids_path}")
        print("Running the following commands:")
        for cmd in cmds:
            print(cmd)
        print()
        for cmd in cmds:
            subprocess.run(cmd, shell=True)

        df_geno = create_geno_df(run_prefix)
        df_geno = impute_df(df_geno, impute_type)
        df_merged = pd.merge(df_merged, df_geno, on='ID', how='inner')
    else:
        print("So no genotypes? Okay, we'll just use additional features provided for the predictions.")

    return df_merged


def filter_common_cols(df1, df2):
    """
    Remove non-overlapping columns from two dataframes.

    Args:
        df1 (pandas.DataFrame): First dataframe.
        df2 (pandas.DataFrame): Second dataframe.

    :return: df1 *(pandas.DataFrame)*: \n
        First dataframe with non-overlapping columns removed.
    :return: df_merged *(pandas.DataFrame)*: \n
        Second dataframe with non-overlapping columns removed.
    """

    common_cols = df1.columns.intersection(df2.columns)
    df1 = df1[common_cols]
    df2 = df2[common_cols]
    return df1, df2
