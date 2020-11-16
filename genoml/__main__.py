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


import argparse
import functools
import sys

from genoml import utils
from genoml import dependencies
from genoml.cli import continuous_supervised_train, continuous_supervised_tune, \
    continuous_supervised_test, discrete_supervised_train, \
    discrete_supervised_tune, discrete_supervised_test, munging, harmonizing


def handle_main():
    entry_points = [
        {"name": "continuous", "handler": handle_continuous,
        "description": "for processing continuous datatypes (ex: age at onset)"},
        {"name": "discrete", "handler": handle_discrete, 
        "description": "for processing discrete datatypes (ex: case vs. control status)"},
        {"name": "harmonize", "handler": handle_harmonize,
        "description": "for harmonizing incoming test datasets to use the same SNPs and reference alleles prior to munging, training, and testing"},
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


def handle_continuous_supervised():
    entry_points = [
        {"name": "munge", "handler": handle_continuous_supervised_munge,
         "description": "for supervised munging and preprocessing of your continuous dataset prior to training"},
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
        {"name": "train", "handler": handle_discrete_supervised_train,
         "description": "for supervised training of your munged discrete dataset by competing different algorithms"},
        {"name": "tune", "handler": handle_discrete_supervised_tune,
         "description": "for supervised tuning of your munged and trained discrete dataset"},
        {"name": "test", "handler": handle_discrete_supervised_test,
         "description": "for supervised testing of your model generated on discrete data on unseen discrete data (after its been harmonized and munged)"},
    ]
    handle_dispatcher(entry_points, "genoml discrete supervised", 2)


def handle_harmonize():
    handle_endpoints("genoml harmonize",
                     ["test_geno_prefix", "test_prefix", "ref_model_prefix",
                      "training_snps_alleles"],
                     harmonizing.main, 1)


def handle_continuous_supervised_munge():
    handle_endpoints("genoml discrete supervised munge",
                     ["prefix", "impute", "geno", "skip_prune", "r2_cutoff", "pheno", "addit",
                      "feature_selection", "gwas", "p", "vif", "iter",
                      "ref_cols_harmonize", "umap_reduce", "adjust_data",
                      "adjust_normalize","target_features", "confounders"],
                      functools.partial(munging.main, data_type="c"), 3)


def handle_continuous_supervised_train():
    handle_endpoints("genoml continuous supervised train",
                     ["prefix", "export_predictions", "matching_columns"],
                     continuous_supervised_train.main, 3)


def handle_continuous_supervised_tune():
    handle_endpoints("genoml continuous supervised tune",
                     ["prefix", "max_tune", "n_cv", "matching_columns"],
                     continuous_supervised_tune.main, 3)


def handle_continuous_supervised_test():
    handle_endpoints("genoml continuous supervised test",
                     ["prefix", "test_prefix", "ref_model_prefix"],
                     continuous_supervised_test.main, 3)


def handle_discrete_supervised_munge():
    handle_endpoints("genoml discrete supervised munge",
                     ["prefix", "impute", "geno", "skip_prune", "r2_cutoff", "pheno", "addit",
                      "feature_selection", "gwas", "p", "vif", "iter",
                      "ref_cols_harmonize", "umap_reduce", "adjust_data",
                      "adjust_normalize","target_features", "confounders"],
                      functools.partial(munging.main, data_type="d"), 3)


def handle_discrete_supervised_train():
    handle_endpoints("genoml discrete supervised train",
                     ["prefix", "metric_max", "prob_hist", "auc",
                      "matching_columns"],
                     discrete_supervised_train.main, 3)


def handle_discrete_supervised_tune():
    handle_endpoints("genoml discrete supervised tune",
                     ["prefix", "metric_tune", "max_tune", "n_cv", "matching_columns"],
                     discrete_supervised_tune.main, 3)


def handle_discrete_supervised_test():
    handle_endpoints("genoml discrete supervised test",
                     ["prefix", "test_prefix", "ref_model_prefix"],
                     discrete_supervised_test.main, 3)


def handle_dispatcher(entry_points, command_name, level):
    usage_description = f'{command_name} <command> [<args>]\n'
    for command in entry_points:
        usage_description += "   {name:15s} {description}\n".format(**command)

    parser = argparse.ArgumentParser(prog=command_name,
                                     description=f'{command_name}',
                                     usage=usage_description
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
        parser.add_argument('--prefix', type=str, default="GenoML_data",
                            help="Prefix for your output build.")

    elif flag_name == "metric_max":
        parser.add_argument('--metric_max', type=str, default='AUC',
                            choices=['AUC', "Balanced_Accuracy", "Specificity",
                                     "Sensitivity"],
                            help='How do you want to determine which algorithm'
                                 ' performed the best? [default: AUC].')

    elif flag_name == "verbose":
        parser.add_argument('-v', '--verbose', action='store_true',
                            default=False, help="Verbose output.")

    elif flag_name == "matching_columns":
        parser.add_argument('--matching_columns', type=str, default=None,
                            help="This is the list of intersecting columns "
                                 "between reference and testing datasets with "
                                 "the suffix *_finalHarmonizedCols_toKeep.txt")

    elif flag_name == "prob_hist":
        parser.add_argument('--prob_hist', type=bool, default=False)

    elif flag_name == "auc":
        parser.add_argument('--auc', type=bool, default=False)

    elif flag_name == "export_predictions":
        parser.add_argument('--export_predictions', type=bool, default=False)

    elif flag_name == "metric_tune":
        parser.add_argument('--metric_tune', type=str, default='AUC',
                            choices=['AUC', "Balanced_Accuracy"],
                            help='Using what metric of the best algorithm do '
                                 'you want to tune on? [default: AUC].')

    elif flag_name == "max_tune":
        parser.add_argument('--max_tune', type=int, default=50,
                            help='Max number of tuning iterations: (integer '
                                 'likely greater than 10). This governs the '
                                 'length of tuning process, run speed and the '
                                 'maximum number of possible combinations of '
                                 'tuning parameters [default: 50].')

    elif flag_name == "n_cv":
        parser.add_argument('--n_cv', type=int, default=5,
                            help='Number of cross validations: (integer likely '
                                 'greater than 3). Here we set the number of '
                                 'cross-validation runs for the algorithms '
                                 '[default: 5].')

    elif flag_name == "test_prefix":
        parser.add_argument('--test_prefix', type=str, default='GenoML_data',
                            help='Prefix for the dataset you would like to '
                                 'test against your reference model. '
                                 'Remember, the model will not function well '
                                 'if it does not include the same features, '
                                 'and these features should be on the same '
                                 'numeric scale, you can leave off the '
                                 '\'.dataForML.h5\' suffix.')

    elif flag_name == "ref_model_prefix":
        parser.add_argument('--ref_model_prefix', type=str,
                            default='GenoML_model',
                            help='Prefix of your reference model file, '
                                 'you can leave off the \'.joblib\' suffix.')

    elif flag_name == "test_geno_prefix":
        parser.add_argument('--test_geno_prefix', type=str,
                            default='genotype_binaries',
                            help='Prefix of the genotypes for the test '
                                 'dataset in PLINK binary format.',
                            required=True)

    elif flag_name == "training_snps_alleles":
        parser.add_argument('--training_snps_alleles', type=str,
                            default=None,
                            help='File to the SNPs and alleles file generated '
                                 'in the training phase that we will use to '
                                 'compare.',
                            required=True)

    elif flag_name == "pheno":
        parser.add_argument("--pheno", type=str, default=None,
                            help="Phenotype: (string file path). Path to CSV "
                                 "phenotype file [default: None].",
                            required=True)

    elif flag_name == "geno":
        parser.add_argument("--geno", type=str, default=None,
                            help="Genotype: (string file path). Path to PLINK "
                                 "format genotype file, everything before the "
                                 "*.bed/bim/fam [default: None].")

    elif flag_name == "skip_prune":
        parser.add_argument("--skip_prune", type=str, default="no",
                            help="[default: no].",
                            choices=["no", "yes"], required=False)

    elif flag_name == "r2_cutoff":
        parser.add_argument("--r2_cutoff", type=str, default="0.5",
                            help="How strict would you like your pruning? [default: 0.5].",
                            choices=["0.1", "0.2", "0.3", "0.4", "0.5"], required=False)
                            
    elif flag_name == "addit":
        parser.add_argument("--addit", type=str, default=None,
                            help="Additional: (string file path). Path to CSV "
                                 "format feature file [default: None].")

    elif flag_name == "gwas":
        parser.add_argument("--gwas", type=str, default=None,
                            help="GWAS summary stats: (string file path). "
                                 "Path to CSV format external GWAS summary "
                                 "statistics containing at least the columns "
                                 "SNP and P in the header [default: None].")

    elif flag_name == "p":
        parser.add_argument("--p", type=float, default=0.001,
                            help="P threshold for GWAS: (some value between "
                                 "0-1). P value to filter your SNP data on ["
                                 "default: 0.001].")

    elif flag_name == "vif":
        parser.add_argument("--vif", type=int, default=0,
                            help="Variance Inflation Factor (VIF): (integer). "
                                 "This is the VIF threshold for pruning "
                                 "non-genotype features. We recommend a value "
                                 "of 5-10. The default of 0 means no VIF "
                                 "filtering will be done. [default: 0].")

    elif flag_name == "iter":
        parser.add_argument("--iter", type=int, default=0,
                            help="Iterator: (integer). How many iterations of "
                                 "VIF pruning of features do you want to run. "
                                 "To save time VIF is run in randomly "
                                 "assorted chunks of 1000 features per "
                                 "iteration. The default of 1 means only one "
                                 "pass through the data. [default: 1].")

    elif flag_name == "impute":
        parser.add_argument("--impute", type=str, default="median",
                            help="Imputation: (mean, median). Governs "
                                 "secondary imputation and data "
                                 "transformation [default: median].",
                            choices=["median", "mean"])

    elif flag_name == "feature_selection":
        parser.add_argument('--feature_selection', type=int, default=0,
                            help='Run a quick tree-based feature selection '
                                 'routine prior to anything else, here you '
                                 'input the integer number of estimators '
                                 'needed, we suggest >= 50. The default of 0 '
                                 'will skip this functionality. This will '
                                 'also output a reduced dataset for analyses '
                                 'in addition to feature ranks. [default: 0]')

    elif flag_name == "ref_cols_harmonize":
        parser.add_argument('--ref_cols_harmonize', type=str, default=None,
                            help='Are you now munging a test dataset '
                                 'following the harmonize step? Here you '
                                 'input the path to the to the '
                                 '*_refColsHarmonize_toKeep.txt file '
                                 'generated at that step.')
    
    elif flag_name == "umap_reduce":
        parser.add_argument('--umap_reduce', type=str, default="no",
                            help = 'Would you like to reduce your dimensions with UMAP? [default: no]. Must be run with --confounders flag if yes.', 
                            choices=["no", "yes"], required='--confounders' in sys.argv and '--adjust_data' in sys.argv and '--adjust_normalize' in sys.argv)

    elif flag_name == "adjust_data":
        parser.add_argument('--adjust_data', type=str, default="no",
                            help = 'Would you like to adjust features and/or confounders in your data? [default: no]', 
                            choices=["yes", "no"], required='--adjust_normalize' in sys.argv and '--target_features' in sys.argv or '--confounders' in sys.argv)

    elif flag_name == "adjust_normalize":
        parser.add_argument('--adjust_normalize', type=str, default="yes",
                            help = 'Would you like to normalize the features and/or confounders you are adjusting for in your data? [default: yes]', 
                            choices=["no", "yes"], required='--adjust_data' in sys.argv and '--target_features' in sys.argv or '--confounders' in sys.argv)

    elif flag_name == "target_features":
        parser.add_argument('--target_features', type=str, default=None,
                            help = 'For adjusting data. A .txt file, one column, with a list of features '
                            'to adjust (no header). These should correspond to features '
                            'in the munged dataset', required='--adjust_data' in sys.argv and '--adjust_normalize' in sys.argv)

    elif flag_name == "confounders":
        parser.add_argument('--confounders', type=str, default=None,
                            help = 'For adjusting data. A .csv of confounders to adjust for with ID column and header.'
                            'Numeric, with no missing data and the ID column'
                            'is mandatory', required='--adjust_data' in sys.argv and '--adjust_normalize' in sys.argv)

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
        endpoint(*args)


if __name__ == "__main__":
    handle_main()