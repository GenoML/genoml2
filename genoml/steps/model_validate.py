#! /usr/bin/env python -u
# coding=utf-8
import subprocess
from shutil import copyfile

from genoml.steps import PhenoScale, StepBase

__author__ = 'Sayed Hadi Hashemi'


class ModelValidateStep(StepBase):
    """performs validation with existing data"""
    _valid_prefix = None

    def _reduce_validate(self):
        self.execute_command([
            self._dependecies["Plink"],
            "--bfile", self._opt.valid_geno_dir,
            "--extract", f"{self._opt.prune_prefix}.reduced_genos_snpList",
            "--recode", "A",
            "--out", f"{self._valid_prefix}.reduced_genos"
        ], name="Plink")

    def _merge(self):
        self.merge_reduced()

    def _main(self):
        if self._opt.pheno_scale == PhenoScale.DISCRETE:
            script_name = self._opt.VALIDATE_DISC if self._opt.pheno_scale == PhenoScale.DISCRETE \
                else self._opt.VALIDATE_CONT
            self.execute_command([
                self._dependecies["R"],
                script_name,
                self._valid_prefix,
                self._opt.n_cores,
                self._opt.impute_data,
                self._opt.prune_prefix #todo: new best_model
            ], name="VALIDATE_CONT, please make sure you have included .cov and .addit validation files, if used for "
                    "training.")

    def process(self):
        self._valid_prefix = f"{self._opt.prune_prefix}_validation"
        self.model_validate()

    def merge_reduced(self):
        self.execute_command([
            self._dependecies["R"],
            self._opt.MERGE,
            self._opt.valid_geno_dir,
            self._opt.valid_pheno_file,
            self.xna(self._opt.valid_cov_file),
            self.xna(self._opt.valid_addit_file),
            self._valid_prefix
        ], name="R")

    @staticmethod
    def xna(s):
        return s if s is not None else "NA"

    # todo: new
    def model_validate(self):
        """this function performs validation with existing data"""
        print("validate")

        # check if GWAS is present (meaning it was also present in training), otherwise the Prune option has been used
        # TODO: find a way to ensure user is providing GWAS for validation, in case it was used during training
        if self._opt.gwas_file is None:
            # we need to specify the forced allele here from the training set genotype file, this pulls the allele to
            # force
            # TODO: refactor to a Python code
            self.cut_column(self._opt.geno_prefix + ".bim",
                            "2,5",
                            self._opt.prune_prefix + ".allelesToForce")

            # plink
            self.execute_command([
                self._dependecies["Plink"],
                "--bfile", self._opt.valid_geno_dir,
                "--extract", self._opt.prune_prefix + '.reduced_genos_snpList',
                "--recode", "A",
                "--recode-allele", self._opt.prune_prefix + '.allelesToForce',
                "--out", self._valid_prefix + '.reduced_genos'
            ], name="model_validate")
        else:  # gwas_file is not None
            # plink
            self.execute_command([
                self._dependecies["Plink"],
                "--bfile", self._opt.valid_geno_dir,
                "--extract", self._opt.prune_prefix + '.reduced_genos_snpList',
                "--recode", "A",
                "--recode-allele", self._opt.prune_prefix + '.variantWeightings',
                "--out", self._valid_prefix + '.reduced_genos'
            ], name="model_validate")

            # copy
            copyfile(self._opt.prune_prefix + ".temp.snpsToPull2", self._valid_prefix + ".temp.snpsToPull2")

            self.execute_command([
                self._dependecies["R"],
                self._opt.SCALE_VAR_DOSES_VALID,
                self._opt.prune_prefix,
                self._opt.gwas_file,
                self._opt.valid_geno_dir,
                self._opt.geno_prefix
            ], name="validate")

        self.merge_reduced()
        self._main()

        self.execute_command([
            self._dependecies["R"],
            self._opt.CHECK_VALIDATION,
            self._valid_prefix,
            self._opt.valid_pheno_file
        ], name="CHECK_VALIDATION")
