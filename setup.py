import setuptools

with open('requirements.txt') as file:
    requires = [line.strip() for line in file if not line.startswith('#')]


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genoml", 
    version="0.0.1",
    author="Mike A. Nalls",
    author_email="mike@datatecnica.com",
    description="GenoML is an automated machine learning tool that optimizes basic machine learning pipelines for genomic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://genoml.github.io/", #TODO: Change to GitHub repository?
    entry_points = {
        'console_scripts': 
        ['GenoML=genoml.GenoML:main',
        'GenoMLMunging=genoml.GenoMLMunging:main'],
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
