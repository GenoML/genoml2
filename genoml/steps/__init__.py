#! /usr/bin/env python -u
# coding=utf-8
import subprocess
from enum import Enum

from genoml.utils import ColoredBox, ContextScope, DescriptionLoader

__author__ = 'Sayed Hadi Hashemi'


class PhenoScale(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class StepBase:
    _opt = {}
    _dependecies = {}

    def set_environment(self, options, dependencies):
        self._opt = options
        self._dependecies = dependencies
        ContextScope.set_verbose(self._opt.verbose > 0)

    def execute_command(self, args, name="", output=None):
        with ColoredBox(ColoredBox.BLUE):
            default_output = None if self._opt.verbose > 2 else subprocess.DEVNULL
            output = default_output if output is None else output
            try:
                subprocess.check_call(
                    args,
                    stdout=output,
                    stderr=default_output,
                )
            except subprocess.CalledProcessError:
                print("Running: ", args)
                raise EnvironmentError(f"{name} fail")

    def cut_column(self, input_file, columns, output_file):
        with open(output_file, "w") as fp:
            self.execute_command([
                "cut",
                "-f", columns,
                input_file,
            ], name="Cut", output=fp)
