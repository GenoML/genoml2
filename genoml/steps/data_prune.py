#! /usr/bin/env python -u
# coding=utf-8
import csv
import glob
import os
import subprocess
import sys

from pandas_plink import read_plink

from genoml.steps import StepBase, PhenoScale

__author__ = 'Sayed Hadi Hashemi'


class DataPruneStep(StepBase):
    herit = None
    num_snps = None
    num_samples = None

    def process(self):
        self.check_inputs()
        self.check_geno()
        self.reduce()
        self.merge_reduced()
        # print("Pruning stage prefix (use if for the next stage): " + self._opt.prefix)

    def reduce(self):
        # main for prune()
        if self._opt.geno_prefix and self._opt.pheno_file:
            if not (self._opt.cov_file or self._opt.gwas_file or self.herit):
                self.reduce_prune()
            elif self._opt.gwas_file and not self.herit:
                self.reduce_prsice()
                self.scale_variant()
            elif self._opt.gwas_file and self.herit and not self._opt.cov_file:
                self.reduce_sblup()
                self.scale_variant()
            else:
                raise RuntimeError("ISSUE: all files present: geno, pheno, gwas, cov, herit. "
                                   "Need to decide between cov and herit for method.")

    def scale_variant(self):
        self.execute_command([
            self._dependecies["R"],
            self._opt.SCALE_VAR_DOSES_TRAIN,
            self._opt.prefix,
            self._opt.gwas_file,
            self._opt.geno_prefix
        ], name="scaleVariantDoses")

    def reduce_prune(self):
        self.execute_command([
            self._dependecies["Plink"],
            "--bfile", self._opt.geno_prefix,
            "--indep-pairwise", "10000",
            "1", "0.1",
            "--out", self._opt.prefix + '.temp'
        ], name="prune")

        self.execute_command([
            self._dependecies["Plink"],
            "--bfile", self._opt.geno_prefix,
            "--extract", self._opt.prefix + '.temp.prune.in',
            "--recode", "A",
            "--out", self._opt.prefix + '.reduced_genos'
        ], name="Plink")

        with open(self._opt.prefix + ".reduced_genos_snpList", "w") as fp:
            subprocess.check_call(["cut", "-f", "1", self._opt.prefix + ".temp.prune.in"], stdout=fp)

    def reduce_prsice(self):
        print("prsice;")

        if self._opt.pheno_scale == PhenoScale.DISCRETE and not self._opt.cov_file:
            print("prsice, disc, cov=na")
            self.execute_command([
                self._dependecies["R"],
                self._opt.PRSICE_R,
                "--binary-target", "T",
                "--prsice", self._dependecies["PRSice"],
                "-n", self._opt.n_cores,
                "--out", self._opt.prefix + ".temp",
                "--pheno-file", self._opt.pheno_file,
                "-t", self._opt.geno_prefix,
                "-b", self._opt.gwas_file,
                "--print-snp",
                "--score", "std",
                "--perm", "10000",
                "--bar-levels",
                "5E-8,4E-8,3E-8,2E-8,1E-8,9E-7,8E-7,7E-7,6E-7,5E-7,4E-7,3E-7,2E-7,1E-7,9E-6,8E-6,7E-6,6E-6,5E-6,4E-6,"
                "3E-6,2E-6,1E-6,9E-5,8E-5,7E-5,6E-5,5E-5,4E-5,3E-5,2E-5,1E-5,9E-4,8E-4,7E-4,6E-4,5E-4,4E-4,3E-4,2E-4,"
                "1E-4,9E-3,8E-3,7E-3,6E-3,5E-3,4E-3,3E-3,2E-3,1E-3,9E-2,8E-2,7E-2,6E-2,5E-2",
                "--no-full",
                "--fastscore",
                "--binary-target", "T",
                "--beta",
                "--snp", "SNP",
                "--A1", "A1",
                "--A2", "A2",
                "--stat", "b",
                "--se", "se",
                "--pvalue", "p"], name="Prune")

        elif self._opt.pheno_scale == PhenoScale.DISCRETE and self._opt.cov_file:
            self.execute_command([
                self._dependecies["R"],
                self._opt.PRSICE_R,
                "--binary-target", "T",
                "--prsice", self._dependecies["PRSice"],
                "-n", self._opt.n_cores,
                "--out", self._opt.prefix + ".temp",
                "--pheno-file", self._opt.pheno_file,
                "--cov-file", self._opt.cov_file,
                "-t", self._opt.geno_prefix,
                "-b", self._opt.gwas_file,
                "--print-snp",
                "--score", "std",
                "--perm", "10000",
                "--bar-levels",
                "5E-8,4E-8,3E-8,2E-8,1E-8,9E-7,8E-7,7E-7,6E-7,5E-7,4E-7,3E-7,2E-7,1E-7,9E-6,8E-6,7E-6,6E-6,5E-6,4E-6,"
                "3E-6,2E-6,1E-6,9E-5,8E-5,7E-5,6E-5,5E-5,4E-5,3E-5,2E-5,1E-5,9E-4,8E-4,7E-4,6E-4,5E-4,4E-4,3E-4,2E-4,"
                "1E-4,9E-3,8E-3,7E-3,6E-3,5E-3,4E-3,3E-3,2E-3,1E-3,9E-2,8E-2,7E-2,6E-2,5E-2",
                "--no-full",
                "--fastscore",
                "--binary-target", "T",
                "--beta",
                "--snp", "SNP",
                "--A1", "A1",
                "--A2", "A2",
                "--stat", "b",
                "--se", "se",
                "--pvalue", "p"
            ], name="Prune")
        elif self._opt.pheno_scale == PhenoScale.CONTINUOUS and not self._opt.cov_file:
            print("prsice, cont, cov=na")
            self.execute_command([
                self._dependecies["R"],
                self._opt.PRSICE_R,
                "--prsice", self._opt.PRSICE,
                "-n", self._opt.n_cores,
                "--out", self._opt.prefix + ".temp",
                "--pheno-file", self._opt.pheno_file,
                "-t", self._opt.geno_prefix,
                "-b", self._opt.gwas_file,
                "--print-snp",
                "--score", "std",
                "--perm", "10000",
                "--bar-levels",
                "5E-8,4E-8,3E-8,2E-8,1E-8,9E-7,8E-7,7E-7,6E-7,5E-7,4E-7,3E-7,2E-7,1E-7,9E-6,8E-6,7E-6,6E-6,5E-6,4E-6,"
                "3E-6,2E-6,1E-6,9E-5,8E-5,7E-5,6E-5,5E-5,4E-5,3E-5,2E-5,1E-5,9E-4,8E-4,7E-4,6E-4,5E-4,4E-4,3E-4,2E-4,"
                "1E-4,9E-3,8E-3,7E-3,6E-3,5E-3,4E-3,3E-3,2E-3,1E-3,9E-2,8E-2,7E-2,6E-2,5E-2",
                "--no-full",
                "--fastscore",
                "--binary-target", "F",
                "--beta",
                "--snp", "SNP",
                "--A1", "A1",
                "--A2", "A2",
                "--stat", "b",
                "--se", "se",
                "--pvalue", "p"
            ], name="Prune")
        elif self._opt.pheno_scale == PhenoScale.CONTINUOUS and self._opt.cov_file:
            print("prsice, cont, cov!=na")
            self.execute_command([
                self._dependecies["R"],
                self._opt.PRSICE_R,
                "--prsice", self._opt.PRSICE,
                "-n", self._opt.n_cores,
                "--out", self._opt.prefix + ".temp",
                "--pheno-file", self._opt.pheno_file,
                "--cov-file", self._opt.cov_file,
                "-t", self._opt.geno_prefix,
                "-b", self._opt.gwas_file,
                "--print-snp",
                "--score", "std",
                "--perm", "10000",
                "--bar-levels",
                "5E-8,4E-8,3E-8,2E-8,1E-8,9E-7,8E-7,7E-7,6E-7,5E-7,4E-7,3E-7,2E-7,1E-7,9E-6,8E-6,7E-6,6E-6,5E-6,4E-6,"
                "3E-6,2E-6,1E-6,9E-5,8E-5,7E-5,6E-5,5E-5,4E-5,3E-5,2E-5,1E-5,9E-4,8E-4,7E-4,6E-4,5E-4,4E-4,3E-4,2E-4,"
                "1E-4,9E-3,8E-3,7E-3,6E-3,5E-3,4E-3,3E-3,2E-3,1E-3,9E-2,8E-2,7E-2,6E-2,5E-2",
                "--no-full",
                "--fastscore",
                "--binary-target", "F",
                "--beta",
                "--snp", "SNP",
                "--A1", "A1",
                "--A2", "A2",
                "--stat", "b",
                "--se", "se",
                "--pvalue", "p"
            ], name="Prune")
        else:
            raise RuntimeError("ISSUE: reduce_prsice not a possible condition present.")

        self.cut_column(self._opt.prefix + '.temp.snp',
                        "2",
                        self._opt.prefix + ".temp.snpsToPull")

        # checking threshold from summary: thresh=$(awk 'NR == 2 {print $3}' $prefix.temp.summary)
        with open(self._opt.prefix + ".temp.summary", 'r') as infile:
            column3 = [cols[2] for cols in csv.reader(infile, delimiter="\t")]
            thresh = column3[1]

        # plink
        self.execute_command([
            self._dependecies["Plink"],
            "--bfile", self._opt.geno_prefix,
            "--extract", self._opt.prefix + '.temp.snpsToPull',
            "--clump", self._opt.gwas_file,
            "--clump-p1", thresh,
            "--clump-p2", thresh,
            "--clump-snp-field", "SNP",
            "--clump-field", "p",
            "--clump-r2", "0.1",
            "--clump-kb", "250",
            "--out", self._opt.prefix + '.tempClumps'
        ], name="Prune")

        # TODO: refarctor to a Python code
        self.cut_column(self._opt.prefix + '.tempClumps.clumped',
                        "2",
                        self._opt.prefix + ".temp.snpsToPull2")

        # plink
        self.execute_command([
            self._dependecies["Plink"],
            "--bfile", self._opt.geno_prefix,
            "--extract", self._opt.prefix + '.temp.snpsToPull2',
            "--recode", "A",
            "--out", self._opt.prefix + '.reduced_genos'
        ], name="Prune")

        # TODO: refarctor to a Python code
        self.cut_column(self._opt.prefix + '.temp.snpsToPull2',
                        "1",
                        self._opt.prefix + ".reduced_genos_snpList")

        # checkPrs: only with PRSice, generating more files after the prune
        self.execute_command([
            self._dependecies["R"],
            self._opt.CHECK_PRS, self._opt.prefix, self._opt.pheno_file
        ], name="prune checkPrs")

    def reduce_sblup(self):
        sblup_lambda = self.num_snps * ((1 / self.herit) - 1)

        self.execute_command([
            self._dependecies["Plink"],
            "--bfile", self._opt.geno_prefix,
            "--pheno", self._opt.pheno_file,
            "--make-bed",
            "--out", self._opt.prefix + '.forSblup'
        ], name="Plink")

        self.execute_command([
            self._dependecies["GCTA"],
            "--bfile", self._opt.prefix + '.forSblup', "--cojo-file", self._opt.gwas_file,
            "--cojo-sblup",
            str(sblup_lambda),
            "--cojo-wind", "10000", "--thread-num", self._opt.n_cores, "--out", self._opt.prefix + '.temp'],
            name="GCTA")

        self.execute_command([
            self._dependecies["R"],
            self._opt.FILTER_SBLUP,
            self._opt.prefix], name="R"
        )

        self.execute_command([
            self._dependecies["Plink"],
            "--bfile", self._opt.prefix + '.forSblup',
            "--extract", self._opt.prefix + '.sblupToPull',
            "--indep-pairwise", "10000", "1", "0.1",
            "--out", self._opt.prefix + '.pruning'],
            name="Plink")

        self.execute_command([
            self._dependecies["Plink"],
            "--bfile", self._opt.prefix + '.forSblup',
            "--extract", self._opt.prefix + '.pruning.prune.in',
            "--recode", "A",
            "--out", self._opt.prefix + '.reduced_genos'
        ], name="Plink")

        with open(self._opt.prefix + ".reduced_genos_snpList", "w") as fp:
            subprocess.check_call(["cut", "-f", "1", self._opt.prefix + '.pruning.prune.in'], stdout=fp)

        self.execute_command([
            self._dependecies["R"],
            self._opt.PREP_SBLUP_SCALE,
            self._opt.prefix,
            self._opt.gwas_file,
            self._opt.geno_prefix
        ], name="prune PREP_SBLUP_SCALE")

        sblup_temp_files = glob.glob(self._opt.prefix + '.forSblup.*')
        for filename in sblup_temp_files:
            os.remove(filename)

    def check_inputs(self):
        for ext in ["bed", "bim", "fam"]:
            filename = f"{self._opt.geno_prefix}.{ext}"
            if not os.path.isfile(filename):
                raise RuntimeError(f"{filename} is not found. Exiting program.")

        if not os.path.isfile(self._opt.pheno_file) or not self._opt.pheno_file.endswith('.pheno'):
            raise RuntimeError("No .pheno file found. Exiting program.")

        if self._opt.cov_file is not None:
            if not os.path.isfile(self._opt.cov_file) or not self._opt.cov_file.endswith('.cov'):
                raise RuntimeError("No .cov file found. Exiting program.")

        if self._opt.gwas_file is not None:
            if not os.path.isfile(self._opt.gwas_file):
                raise RuntimeError("No GWAS file found. Exiting program.")

        if self._opt.herit is not None:
            herit = float(self._opt.herit)
            self.herit = herit
            if not 0 < herit <= 1:
                raise RuntimeError("--herit is not in the range of (0,1].")

        if self._opt.addit_file is not None:
            if not os.path.isfile(self._opt.addit_file) or not self._opt.addit_file.endswith('.addit'):
                raise RuntimeError("No .addit file found. Exiting program.")

        if not os.path.exists(os.path.dirname(self._opt.prefix)):
            os.makedirs(os.path.dirname(self._opt.prefix))

    def check_geno(self):
        """this function checks genotype input, determining the format."""
        (bim, fam, bed) = read_plink(self._opt.geno_prefix)
        self.num_snps = bim.shape[0]
        self.num_samples = fam.iid.unique().size

    def merge_reduced(self):
        self.execute_command([
            self._dependecies["R"],
            self._opt.MERGE,
            self._opt.geno_prefix,
            self._opt.pheno_file,
            self.xna(self._opt.cov_file),
            self.xna(self._opt.addit_file),
            self._opt.prefix
        ], name="R")

    @staticmethod
    def xna(s):
        return s if s is not None else "NA"
