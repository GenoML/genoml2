#! /usr/bin/env python -u
# coding=utf-8

from genoml.steps import StepBase, PhenoScale

__author__ = 'Sayed Hadi Hashemi'


class ModelTuneStep(StepBase):
    """Performs secondary tunning for the ML model"""

    def process(self):
        script_file = self._opt.TUNE_DISC if self._opt.pheno_scale == PhenoScale.DISCRETE else self._opt.TUNE_CONT
        self.execute_command([
            self._dependecies["R"],
            script_file,
            self._opt.prefix,
            self._opt.n_cores,
            self._opt.cv_reps,
            self._opt.grid_search,
            self._opt.impute_data,
            self._opt.best_model_name
        ], name="R")
