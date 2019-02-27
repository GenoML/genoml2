#! /usr/bin/env python -u
# coding=utf-8
import csv
import os
import time

import numpy as np
from docopt import docopt

from genoml import __version__

from genoml.steps import PhenoScale


class Options:
    _options = {}

    def __init__(self, commandline_args_file=None):
        self._options = self.parse_args(commandline_args_file)

    @staticmethod
    def parse_args(commandline_args_file):
        arg_file = os.path.join(os.path.dirname(__file__), "misc", commandline_args_file)
        with open(arg_file, "r") as fp:
            return docopt(fp.read(), version=__version__)

    @property
    def pheno_scale(self):
        """this function checks phenotype input, determining the phenotype format.
        Extract the pheno column and find the number of phenotype classes"""
        # TODO(ffaghri1): Add return hint

        with open(self.pheno_file, 'r') as infile:
            column3 = [cols[2] for cols in csv.reader(infile, delimiter="\t")]
            column3 = column3[1:]  # remove header

        column3 = np.array(column3)
        uniq_pheno = np.unique(column3).size

        if uniq_pheno == 1:
            raise ValueError('check_pheno: phenotype file has only one class.')
        elif uniq_pheno == 2:
            # phenotype file is discrete with two classes
            return PhenoScale.DISCRETE
        else:
            # phenotype file is continuous with uniq_pheno classes
            return PhenoScale.CONTINUOUS

    @property
    def geno_prefix(self):
        return self._options['--geno-prefix']

    @property
    def pheno_file(self):
        return self._options['--pheno-file']

    @property
    def cov_file(self):
        return self._options['--cov-file']

    @property
    def gwas_file(self):
        return self._options['--gwas-file']

    @property
    def herit(self):
        return self._options['--herit']

    @property
    def addit_file(self):
        return self._options['--addit-file']

    @property
    def temp_dir(self):
        return self._options['--temp-dir']

    @property
    def n_cores(self):
        return self._options['--n-cores']

    @property
    def train_speed(self):
        return self._options['--train-speed']

    @property
    def cv_reps(self):
        return self._options['--cv-reps']

    @property
    def impute_data(self):
        return self._options['--impute-data']

    @property
    def grid_search(self):
        return self._options['--grid-search']

    @property
    def prune_prefix(self):
        return self._options['--prune-prefix']

    @property
    def best_model_name(self):
        return self._options['--best-model-name']

    @property
    def valid_geno_dir(self):
        return self._options['--valid-geno-prefix']

    @property
    def valid_pheno_file(self):
        return self._options['--valid-pheno-file']

    @property
    def valid_cov_file(self):
        return self._options['--valid-cov-file']

    @property
    def valid_addit_file(self):
        return self._options['--valid-addit-file']

    @property
    def run_id(self):
        return time.strftime("%Y%m%d-%H%M%S")

    @property
    def prefix(self):
        if '--prune-prefix' not in self._options or self._options['--prune-prefix'] is None:
            return os.path.join(self.temp_dir, self.run_id)
        else:
            return self.prune_prefix

    # TODO: temp for paths to external, fix this by using rpy2 and integration
    @property
    def genoml_env(self):
        return os.environ.copy()

    # @property
    # def genoml_env["PATH"](self):
    #     return "/Users/faraz/Downloads/plink_mac_20181202:" + genoml_env["PATH"]  # plink

    @property
    def other_packages(self):
        return os.path.join(os.path.dirname(__file__), "misc", "R")

    @property
    def PLINK(self):
        return os.path.join(self.other_packages, "plink")

    @property
    def GCTA64(self):
        return os.path.join(self.other_packages, "gcta64_mac")

    @property
    def FILTER_SBLUP(self):
        return os.path.join(self.other_packages, "filterSblup.R")

    @property
    def PRSICE_R(self):
        return os.path.join(self.other_packages, "PRSice.R")

    @property
    def PRSICE_LINUX(self):
        return os.path.join(self.other_packages, "PRSice_linux")

    @property
    def MERGE(self):
        return os.path.join(self.other_packages, "mergeForGenoML.R")

    @property
    def TRAIN_DISC(self):
        return os.path.join(self.other_packages, "trainDisc.R")

    @property
    def TRAIN_CONT(self):
        return os.path.join(self.other_packages, "trainCont.R")

    @property
    def TUNE_DISC(self):
        return os.path.join(self.other_packages, "tuneDisc.R")

    @property
    def TUNE_CONT(self):
        return os.path.join(self.other_packages, "tuneCont.R")

    @property
    def VALIDATE_DISC(self):
        return os.path.join(self.other_packages, "validateDisc.R")

    @property
    def VALIDATE_CONT(self):
        return os.path.join(self.other_packages, "validateCont.R")

    @property
    def SCALE_VAR_DOSES_TRAIN(self):
        return os.path.join(self.other_packages, "scaleVariantDoses_training.R")

    @property
    def SCALE_VAR_DOSES_VALID(self):
        return os.path.join(self.other_packages, "scaleVariantDoses_validation.R")

    @property
    def CHECK_PRS(self):
        return os.path.join(self.other_packages, "checkPrs.R")

    @property
    def CHECK_TRAINING(self):
        return os.path.join(self.other_packages, "checkTraining.R")

    @property
    def CHECK_VALIDATION(self):
        return os.path.join(self.other_packages, "checkValidation.R")

    @property
    def PREP_SBLUP_SCALE(self):
        return os.path.join(self.other_packages, "prepSblupToScale.R")

    def is_data_prune(self):
        return self._options['data-prune']

    def is_model_train(self):
        return self._options['model-train']

    def is_model_tune(self):
        return self._options['model-tune']

    def is_model_validate(self):
        return self._options['model-validate']

    @property
    def model_file(self):
        return self._options['--model-file']

    @property
    def valid_dir(self):
        return self._options['--valid-dir']
