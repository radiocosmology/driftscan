import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

import versioneer

drift_data = {"drift.telescope": ["gmrtpositions.dat"]}

# Load the requirements list
with open("requirements.txt", "r") as fh:
    requires = fh.readlines()

# Enable OpenMP support if available
if sys.platform == "darwin":
    compile_args = []
    link_args = []
else:
    compile_args = ["-fopenmp"]
    link_args = ["-fopenmp"]

# Cython module for fast operations
fast_ext = Extension(
    "drift.util._fast_tools",
    ["drift/util/_fast_tools.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="driftscan",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    ext_modules=cythonize([fast_ext]),
    install_requires=requires,
    python_requires=">=3.9",
    package_data=drift_data,
    entry_points="""
        [console_scripts]
        drift-makeproducts=drift.scripts.makeproducts:cli
        drift-runpipeline=drift.scripts.runpipeline:cli
    """,
    # metadata for upload to PyPI
    author="J. Richard Shaw",
    author_email="richard@phas.ubc.ca",
    description="Transit telescope analysis with the m-mode formalism",
    license="GPL v3.0",
    url="http://github.com/radiocosmology/driftscan",
)
