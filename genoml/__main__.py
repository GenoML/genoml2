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
import argparse
import functools
import sys
from pathlib import Path
from datetime import datetime

from genoml import utils, dependencies
import genoml.discrete.supervised.main as discrete_supervised
import genoml.multiclass.supervised.main as multiclass_supervised
import genoml.continuous.supervised.main as continuous_supervised
import genoml.preprocessing.main as preprocessing


### TODO: Add variables for loading intermediate results that were not generated within GenoML.
### TODO: Add ability to load confounders in same file with addit.


def handle_main():
    entry_points = [
        {"name": "continuous", "handler": handle_continuous,
        "description": "for processing continuous datatypes (ex: age at onset)"},
        {"name": "discrete", "handler": handle_discrete, 
        "description": "for processing discrete datatypes (ex: case vs. control status)"},
        {"name": "multiclass", "handler": handle_multiclass,
        "description": "for processing multiclass datatypes (ex: multiple cases vs. control status)"},
    ]
    handle_dispatcher(entry_points, "genoml", 0)


def handle_continuous():
    entry_points = [
        {"name": "supervised", "handler": handle_continuous_supervised,
         "description": "choice of munge, train, tune, or test your continuous dataset must be specified"},
    ]
    handle_dispatcher(entry_points, "genoml continuous", 1)


def handle_discrete():
    entry_points = [
        {"name": "supervised", "handler": handle_discrete_supervised,
         "description": "choice of munge, train, tune, or test your discrete dataset must be specified"},
    ]
    handle_dispatcher(entry_points, "genoml discrete", 1)


def handle_multiclass():
    entry_points = [
        {"name": "supervised", "handler": handle_multiclass_supervised,
         "description": "choice of munge, train, tune, or test your multiclass dataset must be specified"},
    ]
    handle_dispatcher(entry_points, "genoml multiclass", 1)


def handle_continuous_supervised():
    entry_points = [
        {"name": "munge", "handler": handle_continuous_supervised_munge,
         "description": "for supervised munging and preprocessing of your continuous dataset prior to training"},
        {"name": "harmonize", "handler": handle_continuous_supervised_harmonize,
         "description": "for supervised harmonization of new datasets with previously-munged data"},
        {"name": "train", "handler": handle_continuous_supervised_train,
         "description": "for supervised training of your munged continuous dataset by competing different algorithms"},
        {"name": "tune", "handler": handle_continuous_supervised_tune,
         "description": "for supervised tuning of your munged and trained continuous dataset"},
        {"name": "test", "handler": handle_continuous_supervised_test,
         "description": "for supervised testing of your model generated on continuous data on unseen continuous data (after its been harmonized and munged)"},
    ]
    handle_dispatcher(entry_points, "genoml continuous supervised", 2)


def handle_discrete_supervised():
    entry_points = [
        {"name": "munge", "handler": handle_discrete_supervised_munge,
         "description": "for supervised munging and preprocessing of your discrete dataset prior to training"},
        {"name": "harmonize", "handler": handle_discrete_supervised_harmonize,
         "description": "for supervised harmonization of new datasets with previously-munged data"},
        {"name": "train", "handler": handle_discrete_supervised_train,
         "description": "for supervised training of your munged discrete dataset by competing different algorithms"},
        {"name": "tune", "handler": handle_discrete_supervised_tune,
         "description": "for supervised tuning of your munged and trained discrete dataset"},
        {"name": "test", "handler": handle_discrete_supervised_test,
         "description": "for supervised testing of your model generated on discrete data on unseen discrete data (after its been harmonized and munged)"},
    ]
    handle_dispatcher(entry_points, "genoml discrete supervised", 2)


def handle_multiclass_supervised():
    entry_points = [
        {"name": "munge", "handler": handle_multiclass_supervised_munge,
         "description": "for supervised munging and preprocessing of your multiclass dataset prior to training"},
        {"name": "harmonize", "handler": handle_multiclass_supervised_harmonize,
         "description": "for supervised harmonization of new datasets with previously-munged data"},
        {"name": "train", "handler": handle_multiclass_supervised_train,
         "description": "for supervised training of your munged multiclass dataset by competing different algorithms"},
        {"name": "tune", "handler": handle_multiclass_supervised_tune,
         "description": "for supervised tuning of your munged and trained multiclass dataset"},
        {"name": "test", "handler": handle_multiclass_supervised_test,
         "description": "for supervised testing of your model generated on multiclass data on unseen multiclass data (after its been harmonized and munged)"},
    ]
    handle_dispatcher(entry_points, "genoml multiclass supervised", 2)


def handle_continuous_supervised_munge():
    handle_endpoints("genoml continuous supervised munge",
                     ["prefix", "impute_type", "geno", "pheno", "addit", "geno_test", "pheno_test", "addit_test",
                      "skip_prune", "r2_cutoff", "feature_selection", "gwas", "p", "vif", "vif_iter", "umap_reduce",
                      "adjust_data", "adjust_normalize", "target_features", "confounders", "confounders_test"],
                      functools.partial(preprocessing.munge, data_type="c"), 3)


def handle_continuous_supervised_harmonize():
    ### TODO: Allow users to harmonize from a different directory than where the training Munge data are -- define two different prefixes
    handle_endpoints("genoml continuous supervised harmonize",
                     ["prefix", "geno", "pheno", "addit", "confounders", "force_impute"],
                     functools.partial(preprocessing.harmonize, data_type="c"), 3)


def handle_continuous_supervised_train():
    handle_endpoints("genoml continuous supervised train",
                     ["prefix", "metric_max"],
                     continuous_supervised.train, 3)


def handle_continuous_supervised_tune():
    handle_endpoints("genoml continuous supervised tune",
                     ["prefix", "metric_tune", "max_tune", "n_cv"],
                     continuous_supervised.tune, 3)


def handle_continuous_supervised_test():
    handle_endpoints("genoml continuous supervised test",
                     ["prefix"],
                     continuous_supervised.test, 3)


def handle_discrete_supervised_munge():
    handle_endpoints("genoml discrete supervised munge",
                     ["prefix", "impute_type", "geno", "pheno", "addit", "geno_test", "pheno_test", "addit_test",
                      "skip_prune", "r2_cutoff", "feature_selection", "gwas", "p", "vif", "vif_iter", "umap_reduce",
                      "adjust_data", "adjust_normalize", "target_features", "confounders", "confounders_test"],
                      functools.partial(preprocessing.munge, data_type="d"), 3)

              
def handle_discrete_supervised_harmonize():       
    handle_endpoints("genoml discrete supervised harmonize",
                     ["prefix", "geno", "pheno", "addit", "confounders", "force_impute"],
                     functools.partial(preprocessing.harmonize, data_type="d"), 3)


def handle_discrete_supervised_train():
    handle_endpoints("genoml discrete supervised train",
                     ["prefix", "metric_max"],
                     discrete_supervised.train, 3)


def handle_discrete_supervised_tune():
    handle_endpoints("genoml discrete supervised tune",
                     ["prefix", "metric_tune", "max_tune", "n_cv"],
                     discrete_supervised.tune, 3)


def handle_discrete_supervised_test():
    handle_endpoints("genoml discrete supervised test",
                     ["prefix"],
                     discrete_supervised.test, 3)


def handle_multiclass_supervised_munge():
    handle_endpoints("genoml multiclass supervised munge",
                     ["prefix", "impute_type", "geno", "pheno", "addit", "geno_test", "pheno_test", "addit_test",
                      "skip_prune", "r2_cutoff", "feature_selection", "gwas", "p", "vif", "vif_iter", "umap_reduce",
                      "adjust_data", "adjust_normalize", "target_features", "confounders", "confounders_test"],
                      functools.partial(preprocessing.munge, data_type="d"), 3)


def handle_multiclass_supervised_harmonize():
    handle_endpoints("genoml multiclass supervised harmonize",
                     ["prefix", "geno", "pheno", "addit", "confounders", "force_impute"],
                     functools.partial(preprocessing.harmonize, data_type="d"), 3)


def handle_multiclass_supervised_train():
    handle_endpoints("genoml multiclass supervised train",
                     ["prefix", "metric_max"],
                     multiclass_supervised.train, 3)


def handle_multiclass_supervised_tune():
    handle_endpoints("genoml multiclass supervised tune",
                     ["prefix", "metric_tune", "max_tune", "n_cv"],
                     multiclass_supervised.tune, 3)


def handle_multiclass_supervised_test():
    handle_endpoints("genoml multiclass supervised test",
                     ["prefix"],
                     multiclass_supervised.test, 3)


def handle_dispatcher(entry_points, command_name, level):
    usage_description = f'{command_name} <command> [<args>]\n'
    for command in entry_points:
        usage_description += "   {name:15s} {description}\n".format(**command)

    parser = argparse.ArgumentParser(
        prog=command_name,
        description=command_name,
        usage=usage_description,
    )
    parser.add_argument('command', help='Subcommand to run')
    args = parser.parse_args(sys.argv[level + 1:level + 2])

    candidates = []
    for command in entry_points:
        if command["name"] == args.command:
            command["handler"]()
            return
        if command["name"].startswith(args.command):
            candidates.append(command)

    if len(candidates) == 1:
        candidates[0]["handler"]()
        return

    parser.print_usage()
    print(f'Unrecognized command: {args.command}')
    exit(1)


def add_default_flag(parser, flag_name):
    if flag_name == "prefix":
        parser.add_argument(
            '--prefix', 
            type=str, 
            default="GenoML_data",
            help="Prefix for your output build.",
        )

    elif flag_name == "metric_max":
        if "discrete" in parser.prog or "multiclass" in parser.prog:
            choices = ["AUC", "Balanced_Accuracy", "Specificity", "Sensitivity"]
            default = "AUC"
        elif "continuous" in parser.prog:
            choices = ["Explained_Variance", "Mean_Squared_Error", "Median_Absolute_Error", "R-Squared_Error"]
            default = "Explained_Variance"
        parser.add_argument(
            '--metric_max', 
            type=str, 
            default=default,
            choices=choices,
            help=f'How do you want to determine which algorithm performed the best? [default: {default}].',
        )

    elif flag_name == "verbose":
        parser.add_argument(
            '-v', 
            '--verbose', 
            action='store_true',
            default=False, 
            help="Verbose output.",
        )

    elif flag_name == "metric_tune":
        if "discrete" in parser.prog:
            choices = ["AUC", "Balanced_Accuracy"]
            default = "AUC"
        elif "multiclass" in parser.prog:
            choices = ["AUC"]
            default = "AUC"
        elif "continuous" in parser.prog:
            choices = ["Explained_Variance", "Mean_Squared_Error", "Median_Absolute_Error", "R-Squared_Error"]
            default = "Explained_Variance"
        parser.add_argument(
            '--metric_tune', 
            type=str, 
            default=default,
            choices=choices,
            help=f'Using what metric of the best algorithm do you want to tune on? [default: {default}].',
        )

    elif flag_name == "max_tune":
        parser.add_argument(
            '--max_tune', 
            type=int, 
            default=50,
            help='Max number of tuning iterations: (integer likely greater than 10). This governs the '
                 'length of tuning process, run speed and the maximum number of possible combinations of '
                 'tuning parameters [default: 50].',
        )

    elif flag_name == "n_cv":
        parser.add_argument(
            '--n_cv', 
            type=int, 
            default=5,
            help='Number of cross validations: (integer likely greater than 3). Here we set the number of '
                 'cross-validation runs for the algorithms [default: 5].')

    elif flag_name == "test_geno_prefix":
        parser.add_argument(
            '--test_geno_prefix', 
            type=str,
            default='genotype_binaries',
            help='Prefix of the genotypes for the test dataset in PLINK binary format.',
            required=True,
        )

    elif flag_name == "training_snps_alleles":
        parser.add_argument(
            '--training_snps_alleles', 
            type=str,
            default=None,
            help='File to the SNPs and alleles file generated in the training phase that we will use to compare.',
            required=True,
        )

    elif flag_name == "geno":
        parser.add_argument(
            "--geno", 
            type=str, 
            default=None,
            help="Genotype: (string file path). Path to PLINK format genotype file, everything before the "
                 "*.bed/bim/fam or *.pgen/pvar/psam [default: None].",
            required = "--addit" not in sys.argv,
        )

    elif flag_name == "pheno":
        parser.add_argument(
            "--pheno", 
            type=str, 
            default=None,
            help="Phenotype: (string file path). Path to phenotype file [default: None].",
            required=True,
        )
                            
    elif flag_name == "addit":
        parser.add_argument(
            "--addit", 
            type=str, 
            default=None,
            help="Additional: (string file path). Path to CSV format feature file [default: None].",
            required = "--geno" not in sys.argv,
        )

    elif flag_name == "geno_test":
        parser.add_argument(
            "--geno_test", 
            type=str, 
            default=None,
            help="Genotype: (string file path). Path to PLINK format genotype file for testing dataset,"
                 " everything before the *.bed/bim/fam or *.pgen/pvar/psam [default: None].",
            required = ("--addit_test" not in sys.argv) and ("--pheno_test" in sys.argv),
        )

    elif flag_name == "pheno_test":
        parser.add_argument(
            "--pheno_test", 
            type=str, 
            default=None,
            help="Phenotype: (string file path). Path to phenotype file for testing dataset [default: None].",
            required = ("geno_test" in sys.argv) or ("addit_test" in sys.argv),
        )
                            
    elif flag_name == "addit_test":
        parser.add_argument(
            "--addit_test", 
            type=str, 
            default=None,
            help="Additional: (string file path). Path to CSV format feature file "
                 "for testing dataset [default: None].",
            required = ("--geno_test" not in sys.argv) and ("--pheno_test" in sys.argv),
        )

    elif flag_name == "skip_prune":
        parser.add_argument(
            "--skip_prune", 
            action="store_true",
            help="Skip LD pruning.",
        )

    elif flag_name == "r2_cutoff":
        parser.add_argument(
            "--r2_cutoff", 
            type=str, 
            default="0.5",
            help="How strict would you like your pruning? [default: 0.5].",
            choices=["0.1", "0.2", "0.3", "0.4", "0.5"], 
        )

    elif flag_name == "gwas":
        parser.add_argument(
            "--gwas", 
            action='append', 
            type=str, 
            default=[],
            help="GWAS summary stats: (string file path). Path to CSV format external GWAS summary "
                 "statistics containing at least the columns \"SNP\" and \"p\" in the header (default: []).",
        )

    elif flag_name == "p":
        parser.add_argument(
            "--p", 
            type=float, 
            default=0.001,
            help="P threshold for GWAS: (some value between 0-1). P value to filter your SNP data on (default: 0.001).",
        )

    elif flag_name == "vif":
        parser.add_argument(
            "--vif", 
            type=int, 
            default=0,
            help="Variance Inflation Factor (VIF): (integer). This is the VIF threshold for pruning "
                 "non-genotype features. We recommend a value of 5-10. The default of 0 means no VIF "
                 "filtering will be done. [default: 0].",
        )

    elif flag_name == "vif_iter":
        parser.add_argument(
            "--vif_iter", 
            type=int, 
            default=0,
            help="Iterator: (integer). How many iterations of VIF pruning of features do you want to run. "
                 "To save time VIF is run in randomly assorted chunks of 1000 features per iteration. [default: 0].",
        )

    elif flag_name == "impute_type":
        parser.add_argument(
            "--impute_type", 
            type=str, 
            default="median",
            help="Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].",
            choices=["median", "mean"],
        )

    elif flag_name == "force_impute":
        parser.add_argument(
            "--force_impute", 
            action="store_true",
            help="If harmonizing, add in missing columns using the average value from munging.",
        )

    elif flag_name == "feature_selection":
        parser.add_argument(
            '--feature_selection', 
            type=int, 
            default=0,
            help='Run a quick tree-based feature selection routine prior to anything else, here you '
                 'input the integer number of estimators needed, we suggest >= 50. The default of 0 '
                 'will skip this functionality. This will also output a reduced dataset for analyses '
                 'in addition to feature ranks. [default: 0]',
        )
    
    elif flag_name == "umap_reduce":
        parser.add_argument(
            '--umap_reduce', 
            action="store_true",
            help='Would you like to reduce your dimensions with UMAP? Must be run with --confounders flag if yes.',
        )

    elif flag_name == "adjust_data":
        parser.add_argument(
            '--adjust_data', 
            action="store_true",
            help='Would you like to adjust features and/or confounders in your data?',
        )

    elif flag_name == "adjust_normalize":
        parser.add_argument(
            '--adjust_normalize', 
            action="store_true",
            help='Would you like to normalize the features and/or confounders you are adjusting for in your data?', 
        )

    elif flag_name == "target_features":
        parser.add_argument(
            '--target_features', 
            type=str, 
            default=None,
            help='For adjusting data. A .txt file, one column, with a list of features to adjust (no header). '
                 'These should correspond to features in the munged dataset', 
            required='--adjust_data' in sys.argv,
        )

    elif flag_name == "confounders":
        parser.add_argument(
            '--confounders', 
            type=str, 
            default=None,
            help='For adjusting data. A .csv of confounders to adjust for with ID column and header. Numeric, with no ' 
                 'missing data and the ID column is mandatory', 
            required='--adjust_data' in sys.argv,
        )

    elif flag_name == "confounders_test":
        parser.add_argument(
            '--confounders_test', 
            type=str, 
            default=None,
            help='For adjusting data. A .csv of confounders to adjust for with ID column and header. Numeric, with no ' 
                 'missing data and the ID column is mandatory', 
            required=('--adjust_data' in sys.argv) and ("--pheno_test" in sys.argv),
        )

    else:
        raise Exception(f"Unknown flag: {flag_name}")


def handle_endpoints(command_name, flag_names, endpoint, level):
    parser = argparse.ArgumentParser(prog=command_name)
    for name in flag_names:
        add_default_flag(parser, name)

    add_default_flag(parser, "verbose")

    args, unknown = parser.parse_known_args(sys.argv[level + 1:])
    if unknown:
        parser.print_usage()
        print(f'Unrecognized arguments: {unknown[0]}')
        exit(1)

    utils.ContextScope._verbose = args.verbose
    args = [args.__dict__[name] for name in flag_names]
    args_string = ", ".join([f"{name}: {value if value else '[None]'}" for name, value in zip(flag_names, args)])

    with utils.ContextScope(f"Running {command_name}",
                            description="Args= " + args_string,
                            error=""):
        dependencies.check_dependencies()

        ### TODO: Add check here for if the directory exists
        with open(Path(args[0]).joinpath("log.txt"), "a") as f:
            f.write(datetime.now().astimezone().strftime("%A, %B %d, %Y at %I:%M:%S %p %Z") + "\n")
            f.write(" ".join(sys.argv) + "\n\n")
        endpoint(*args)


if __name__ == "__main__":
    handle_main()
