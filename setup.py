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
    name="genoml2",
    version="1.0.0-beta.7",
    maintainer="The GenoML Development Team",
    maintainer_email="genoml@googlegroups.com",
    description="GenoML is an automated machine learning tool that optimizes"
                " basic machine learning pipelines for genomic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://genoml.github.io/",
    download_url="https://github.com/GenoML/genoml2/archive/v1.0.0-beta.7.tar.gz",
    entry_points={
        'console_scripts':
            ['genoml=genoml.__main__:handle_main'],
    },
    packages=setuptools.find_packages(),
    install_requires=[
        'joblib', 
        'matplotlib',
        'numpy',
        'tables',
        'pandas',
        'pandas_plink',
        'requests',
        'scikit-learn',
        'scipy',
        'seaborn',
        'statsmodels',
        'xgboost',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={'genoml': ['misc/*']},
)
