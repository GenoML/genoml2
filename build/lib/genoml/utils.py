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

import json
import os
import time
import traceback

__author__ = 'Sayed Hadi Hashemi'

import textwrap


class ColoredBox:
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    RESET = 39

    def __init__(self, color=None):
        if color is None:
            color = self.GREEN
        self.__color = color

    def __enter__(self):
        print('\033[{}m'.format(self.__color), end="")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("\x1b[0m", end="")

    @classmethod
    def wrap(cls, text, color):
        return '\033[{}m'.format(color) + text + "\x1b[0m"


class ContextScope:
    indent = 0
    _verbose = False

    def __init__(self, title, description, error, start=True, end=False,
                 **kwargs):
        self._title = title.format(**kwargs)
        self._description = description.format(**kwargs)
        self._error = error.format(**kwargs)
        self._start = start
        self._end = end

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            if self._end:
                print(
                    "{}{}: {}".format(
                        self.get_prefix(ColoredBox.GREEN),
                        ColoredBox.wrap(self._title, ColoredBox.GREEN),
                        ColoredBox.wrap('[Done]', ColoredBox.GREEN)))
            self.remove_indent()
        else:
            print("{}{}: {}".format(
                self.get_prefix(ColoredBox.RED), self._title,
                ColoredBox.wrap('[Failed]', ColoredBox.RED)))
            print("{}".format(self.indent_text(self._error)))
            self.remove_indent()
            traceback.print_exception(exc_type, exc_val, exc_tb)
            exit(1)

    def __enter__(self):
        self.add_indent()
        if self._start:
            print()
            print("{}{}".format(self.get_prefix(ColoredBox.BLUE),
                                ColoredBox.wrap(self._title, ColoredBox.BLUE)))
        if self._verbose and self._description:
            print("{}".format(self._description))

    @classmethod
    def add_indent(cls):
        cls.indent += 1

    @classmethod
    def remove_indent(cls):
        cls.indent -= 1

    @classmethod
    def get_prefix(cls, color=None):
        indent_size = 4
        # text = "=" * (cls.indent * 4) + "> "
        text = "---> " * cls.indent
        if color:
            text = ColoredBox.wrap(text, color)
        return text

    @classmethod
    def indent_text(cls, text):
        WIDTH = 70
        indent = max(0, len(cls.get_prefix()) - 2)
        width = WIDTH - indent
        ret = textwrap.fill(text, width)
        ret = textwrap.indent(ret, " " * indent)
        return ret

    @classmethod
    def set_verbose(cls, verbose):
        cls._verbose = verbose


def function_description(**dkwargs):
    def wrap(func):
        def func_wrapper(*args, **kwargs):
            with ContextScope(**dkwargs):
                return func(*args, **kwargs)

        return func_wrapper

    return wrap


class DescriptionLoader:
    _descriptions = None

    @classmethod
    def _load(cls):
        description_file = os.path.join(os.path.dirname(__file__),
                                        "misc", "descriptions.json")
        with open(description_file) as fp:
            cls._descriptions = json.load(fp)

    @classmethod
    def function_description(cls, key, **kwargs):
        dkwargs = cls.get(key)
        return function_description(**dkwargs, **kwargs)

    @classmethod
    def get(cls, key):
        if cls._descriptions is None:
            cls._load()
        return cls._descriptions[key]

    @classmethod
    def context(cls, key, **kwargs):
        dkwargs = cls.get(key)
        return ContextScope(**dkwargs, **kwargs)

    @classmethod
    def print(cls, key, **kwargs):
        dkwargs = cls.get(key)
        with ContextScope(**dkwargs, **kwargs):
            pass


class Timer:
    def __init__(self):
        self.start = None
        self.end = None

    def start_timer(self):
        self.start = time.time()

    def __enter__(self):
        self.start_timer()
        return self

    def __exit__(self, *args):
        self.stop_timer()

    def stop_timer(self):
        self.end = time.time()

    def elapsed(self):
        return self.end - self.start
