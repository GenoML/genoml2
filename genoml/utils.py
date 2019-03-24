#! /usr/bin/env python -u
# coding=utf-8
import json
import os
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

    def __init__(self, title, description, error):
        self._title = title
        self._description = description
        self._error = error

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            print(f"{self.get_prefix(ColoredBox.GREEN)}{self._title}: {ColoredBox.wrap('[Done]', ColoredBox.GREEN)}")
            print()
            self.remove_indent()
        else:
            print(f"{self.get_prefix(ColoredBox.RED)}{self._title}: {ColoredBox.wrap('[Failed]', ColoredBox.RED)}")
            print(f"{self.indent_text(self._error)}\n")
            self.remove_indent()
            # print(f"{exc_val} {exc_tb}\n")
            traceback.print_exception(exc_type, exc_val, exc_tb)
            # raise exc_val
            exit(1)

    def __enter__(self):
        self.add_indent()
        print(f"{self.get_prefix(ColoredBox.BLUE)}{self._title}")
        if self._verbose:
            # print(f"{self.indent_text(self._description)}\n")
            print(f"{self._description}\n")

    @classmethod
    def add_indent(cls):
        cls.indent += 1

    @classmethod
    def remove_indent(cls):
        cls.indent -= 1

    @classmethod
    def get_prefix(cls, color=None):
        text = "=" * (cls.indent * 4) + "> "
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


def function_description(title, description, error):
    def wrap(func):
        def func_wrapper(*args, **kwargs):
            with ContextScope(title, description, error):
                return func(*args, **kwargs)

        return func_wrapper

    return wrap


class DescriptionLoader:
    _descriptions = None

    @classmethod
    def _load(cls):
        description_file = os.path.join(os.path.dirname(__file__), "misc", "descriptions.json")
        with open(description_file) as fp:
            cls._descriptions = json.load(fp)

    @classmethod
    def function_description(cls, key):
        title, description, error = cls.get(key)
        return function_description(title, description, error)

    @classmethod
    def get(cls, key):
        if cls._descriptions is None:
            cls._load()
        title = cls._descriptions[key]["title"]
        description = cls._descriptions[key]["description"]
        error = cls._descriptions[key]["error"]
        return title, description, error

    @classmethod
    def context(cls, key):
        title, description, error = cls.get(key)
        return ContextScope(title, description, error)


