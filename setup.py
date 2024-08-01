"""Build cython extensions.

The full project config can be found in `pyproject.toml`. `setup.py` is still
required to build cython extensions.
"""

import os
import re
import sysconfig

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# Enable OpenMP support if available
if re.search("gcc", sysconfig.get_config_var("CC")) is None:
    print("Not using OpenMP")
    omp_args = []
else:
    omp_args = ["-fopenmp"]

# Cython module for fast operations
extensions = [
    Extension(
        "drift.util._fast_tools",
        ["drift/util/_fast_tools.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=omp_args,
        extra_link_args=omp_args,
    ),
]

setup(
    name="driftscan",
    ext_modules=cythonize(extensions),
)
