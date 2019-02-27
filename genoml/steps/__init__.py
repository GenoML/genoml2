#! /usr/bin/env python -u
# coding=utf-8
import subprocess
from enum import Enum

from genoml.utils import ColoredBox

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

    @staticmethod
    def execute_command(args, name="", output=None):
        with ColoredBox(ColoredBox.BLUE):
            output = None if output is None else output
            try:
                subprocess.check_call(
                    args,
                    stdout=output,
                    # env=genoml_env
                )
            except subprocess.CalledProcessError:
                print("Running: ", args)
                raise EnvironmentError(f"{name} fail")
            finally:
                print("\x1b[0m\n")

    @staticmethod
    def cut_column(input_file, columns, output_file):
        with open(output_file, "w") as fp:
            StepBase.execute_command([
                "cut",
                "-f", columns,
                input_file,
            ], name="Cut", output=fp)
