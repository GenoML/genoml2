#! /usr/bin/env python -u
# coding=utf-8
import os
import platform
import stat
import subprocess
import zipfile
import logging

from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from io import BytesIO
from pathlib import Path
import requests

__author__ = 'Sayed Hadi Hashemi'

# __executable_folder = os.path.join(os.path.dirname(__file__), "misc", "executables")
__executable_folder = os.path.join(str(Path.home()), ".genoml", "misc", "executables")


def check_r_packages():
    for name in __R_PACKAGES:
        try:
            importr(name)
        except Exception:
            try:
                utils = rpackages.importr('utils')
                utils.chooseCRANmirror(ind=1)
                utils.install_packages(name, repos="https://cloud.r-project.org")
            except Exception:
                raise EnvironmentError(f"Missing R Package: {name}")


def check_exec(exec_path, *args, absolute_path=False):
    if not absolute_path:
        binary_path = os.path.join(__executable_folder, exec_path)
    else:
        binary_path = exec_path
    if not os.path.exists(binary_path):
        return False

    _ = subprocess.run([binary_path, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def install_exec(url, exec_path):
    r = requests.get(url, verify=False, stream=True)
    r.raw.decode_content = True
    buffer = BytesIO()
    buffer.write(r.content)
    with zipfile.ZipFile(buffer, "r") as fp:
        fp.extractall(__executable_folder)

    binary_path = os.path.join(__executable_folder, exec_path)
    os.chmod(binary_path, stat.S_IEXEC)


def check_package(name):
    platform_system = platform.system()

    if name not in __DEPENDENCIES:
        raise EnvironmentError(f"Unknown package: {name}")

    if platform_system not in __DEPENDENCIES[name]:
        raise EnvironmentError(f"Unknown supported OK: {platform_system}")

    entry = __DEPENDENCIES[name][platform_system]

    binary_name = entry["binary"]
    args = entry["version_args"]
    url = entry["url"]

    if check_exec(binary_name, *args):
        logging.debug(f"{name} is found")
        return os.path.join(__executable_folder, binary_name)

    logging.warning(f"Installing {name}")
    install_exec(url, binary_name)
    if not check_exec(binary_name, *args):
        logging.warning(f"Failed to run {name} after installation")
        raise EnvironmentError(f"Can not install {name}")
    else:
        return os.path.join(__executable_folder, binary_name)


def check_R():
    r_path = subprocess.check_output("which Rscript", shell=True).decode().strip()
    if not check_exec(r_path, "--version", absolute_path=True):
        raise EnvironmentError("R is not installed")

    check_r_packages()
    # args = [
    #     r_path,
    #     os.path.abspath(os.path.join(__executable_folder, "..", "R", "dependencies.R"))
    # ]
    #
    # subprocess.check_call(args, stdout=subprocess.PIPE)

    return r_path


def check_dependencies():
    global __DEPENDENCIES
    ret = {}
    for package, data in __DEPENDENCIES.items():
        if "checker" in data:
            ret[package] = data["checker"]()

    return ret


__DEPENDENCIES = {
    "PRSice": {
        "checker": lambda: check_package("PRSice"),
        "Darwin": {
            "binary": "PRSice_mac",
            "version_args": ["-v"],
            "url": "https://github.com/choishingwan/PRSice/releases/download/2.1.9/PRSice_mac.zip"
        },
        "Linux": {
            "binary": "PRSice_linux",
            "version_args": ["-v"],
            "url": "https://github.com/choishingwan/PRSice/releases/download/2.1.9/PRSice_linux.zip"
        }
    },
    "GCTA": {
        "checker": lambda: check_package("GCTA"),
        "Darwin": {
            "binary": "gcta_1.91.7beta_mac/bin/gcta64",
            "version_args": ["-v"],
            "url": "https://cnsgenomics.com/software/gcta/gcta_1.91.7beta_mac.zip"
        },
        "Linux": {
            "binary": "gcta_1.91.7beta/gcta64",
            "version_args": ["-v"],
            "url": "https://cnsgenomics.com/software/gcta/gcta_1.91.7beta.zip"
        }
    },
    "Plink": {
        "checker": lambda: check_package("Plink"),
        "Darwin": {
            "binary": "plink",
            "version_args": ["--version"],
            "url": "http://s3.amazonaws.com/plink1-assets/plink_mac_20181202.zip"
        },
        "Linux": {
            "binary": "plink",
            "version_args": ["--version"],
            "url": "http://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20181202.zip"
        }
    },
    "R": {
        "checker": check_R,
    },
}

__R_PACKAGES = [
    "caret", "lattice", "ggplot2", "rBayesianOptimization", "plotROC", "doParallel", "randomForest", "xgboost", "e1071",
]
