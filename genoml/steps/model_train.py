#! /usr/bin/env python -u
# coding=utf-8

from genoml.steps import StepBase, PhenoScale

__author__ = 'Sayed Hadi Hashemi'


class ModelTrainStep(StepBase):
    """
    Trains a machine learning model. Works based on DISC and CONT phenotypes.
    """

    def process(self):
        script_file = self._opt.TRAIN_DISC if self._opt.pheno_scale == PhenoScale.DISCRETE else self._opt.TRAIN_CONT
        self.execute_command([
            self._dependecies["R"],
            script_file,
            self._opt.prefix,
            self._opt.n_cores,
            self._opt.train_speed,
            self._opt.cv_reps,
            self._opt.grid_search,
            self._opt.impute_data
        ], "R")

        # check_training
        self.execute_command([
            self._dependecies["R"],
            self._opt.CHECK_TRAINING,
            self._opt.prefix,
            self._opt.pheno_file,
        ], "check_training")
