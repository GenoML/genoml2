#! /usr/bin/env python -u
# coding=utf-8
import io
import logging
import os
import pathlib
import platform
import requests
import stat
import subprocess
import zipfile

from genoml.utils import DescriptionLoader


def __get_executable_folder():
    key = "GENOML_DEP_DIR"
    if key in os.environ:
        return os.path.abspath(os.environ.get(key))
    else:
        return os.path.join(str(pathlib.Path.home()), ".genoml", "misc", "executables")


__executable_folder = __get_executable_folder()


def __check_exec(exec_path, *args, absolute_path=False):
    if not absolute_path:
        binary_path = os.path.join(__executable_folder, exec_path)
    else:
        binary_path = exec_path
    if not os.path.exists(binary_path):
        return False

    _ = subprocess.run([binary_path, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def __install_exec(url, exec_path):
    r = requests.get(url, verify=False, stream=True)
    r.raw.decode_content = True
    buffer = io.BytesIO()
    buffer.write(r.content)
    with zipfile.ZipFile(buffer, "r") as fp:
        fp.extractall(__executable_folder)

    binary_path = os.path.join(__executable_folder, exec_path)
    os.chmod(binary_path, stat.S_IEXEC)


def __check_package(name):
    platform_system = platform.system()

    if name not in __DEPENDENCIES:
        raise EnvironmentError("Unknown package: {}".format(name))

    if platform_system not in __DEPENDENCIES[name]:
        raise EnvironmentError("Unknown supported OK: {}".format(platform_system))

    entry = __DEPENDENCIES[name][platform_system]

    binary_name = entry["binary"]
    args = entry["version_args"]
    url = entry["url"]

    if __check_exec(binary_name, *args):
        logging.debug("{} is found".format(name))
        return os.path.join(__executable_folder, binary_name)

    logging.warning("Installing {}".format(name))
    __install_exec(url, binary_name)
    if not __check_exec(binary_name, *args):
        logging.warning("Failed to run {} after installation".format(name))
        raise EnvironmentError("Can not install {}".format(name))
    else:
        return os.path.join(__executable_folder, binary_name)


@DescriptionLoader.function_description("check_dependencies")
def check_dependencies():
    global __DEPENDENCIES
    ret = {}
    for package, data in __DEPENDENCIES.items():
        if "checker" in data:
            ret[package] = data["checker"]()

    return ret


@DescriptionLoader.function_description("check_dependencies_Plink")
def check_plink():
    return __check_package("Plink")


__DEPENDENCIES = {
    "Plink": {
        "checker": check_plink,
        "Darwin": {
            "binary": "plink",
            "version_args": ["--version"],
            "url": "http://s3.amazonaws.com/plink1-assets/plink_mac_20200219.zip"
        },
        "Linux": {
            "binary": "plink",
            "version_args": ["--version"],
            "url": "http://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20200219.zip"
        }
    },
}
