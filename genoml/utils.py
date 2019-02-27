#! /usr/bin/env python -u
# coding=utf-8

__author__ = 'Sayed Hadi Hashemi'


# def run_r_script(script_name, *arg):

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
        print('\033[{}m\n'.format(self.__color))

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("\x1b[0m\n")

