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

import io
import logging
import os
import pathlib
import platform
import requests
import stat
import subprocess
import zipfile

from genoml import utils


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

    _ = subprocess.run([binary_path, *args], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
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
        raise EnvironmentError("Unsupported OS: {}".format(platform_system))

    entry = __DEPENDENCIES[name][platform_system]
    binary_name = entry["binary"]
    args = entry["version_args"]
    url = entry["url"]

    # Special case: override URL based on macOS architecture
    if name == "Plink" and platform_system == "Darwin":
        arch = platform.machine()
        if arch == "arm64":
            url = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_mac_arm64_20250609.zip"
        elif arch == "x86_64":
            url = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_mac_avx2_20250609.zip"
        else:
            url = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_mac_20250609.zip"

    if __check_exec(binary_name, *args):
        logging.debug(f"{name} is already installed.")
        return os.path.join(__executable_folder, binary_name)

    logging.warning(f"Installing {name} from {url}")
    __install_exec(url, binary_name)

    if not __check_exec(binary_name, *args):
        logging.warning(f"Failed to run {name} after installation")
        raise EnvironmentError(f"Cannot install {name}")
    else:
        return os.path.join(__executable_folder, binary_name)


@utils.DescriptionLoader.function_description("check_dependencies")
def check_dependencies():
    global __DEPENDENCIES
    ret = {}
    for package, data in __DEPENDENCIES.items():
        if "checker" in data:
            with utils.DescriptionLoader.context(f"check_dependencies_{package}"):
                ret[package] = data["checker"]()
    return ret


def check_plink():
    return __check_package("Plink")


__DEPENDENCIES = {
    "Plink": {
        "checker": check_plink,
        "Darwin": {
            "binary": "plink2",
            "version_args": ["--version"],
            "url": "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_mac_20250609.zip"  # fallback
        },
        "Linux": {
            "binary": "plink2",
            "version_args": ["--version"],
            "url": "http://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_latest.zip"
        }
    },
}
