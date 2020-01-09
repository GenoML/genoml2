#! /usr/bin/env python -u
# coding=utf-8
import os

from genoml.steps import StepBase

__author__ = 'Sayed Hadi Hashemi'


class Experimental(StepBase):
    def process(self):
        base_path = self._opt.experimental_path
        for i, pkg in enumerate(os.listdir(base_path)):
            full_path = os.path.abspath(os.path.join(base_path, pkg))
            if not os.path.isdir(full_path):
                continue
            print(">> {:2}: {}".format(i, full_path))
            readme_path = os.path.join(full_path, "README.md")
            if os.path.exists(readme_path):
                with open(readme_path) as fp:
                    print(fp.read().strip())
            print()
