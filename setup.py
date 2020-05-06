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

import setuptools

with open('requirements.txt') as file:
    requires = [line.strip() for line in file if not line.startswith('#')]


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genoml", 
    version="2.0.0",
    author="Mike A. Nalls",
    author_email="mike@datatecnica.com",
    description="GenoML is an automated machine learning tool that optimizes basic machine learning pipelines for genomic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://genoml.github.io/", 
    entry_points = {
        'console_scripts': 
        ['GenoML=genoml.GenoML:main',
        'GenoMLMunging=genoml.GenoMLMunging:main',
        'GenoMLHarmonizing=genoml.GenoMLHarmonizing:main'],
    },
    packages=setuptools.find_packages(),
    install_requires=requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={'genoml': ['misc/*']},
)
