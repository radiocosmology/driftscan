"""Build cython extensions.

The full project config can be found in `pyproject.toml`. `setup.py` is still
required to build cython extensions.
"""

import os
import platform
import re
import tempfile
import sysconfig
import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# Subset of `-ffast-math` flags which preserve IEEE compliance
FAST_MATH_ARGS = ["-O3", "-fno-math-errno", "-fno-trapping-math"]
# Additional compiler flags
COMPILE_FLAGS = ["-flto", *FAST_MATH_ARGS]


def get_mcpu_flag():
    """Try to figure out the relevant cpu flags."""
    system = platform.system().lower()

    if system != "darwin":
        # Making no assumptions about other systems
        return "-march=native" if system == "linux" else ""

    try:
        # Get CPU model: "Apple M3 Pro", etc.
        cpu_brand = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode()
            .strip()
            .lower()
        )
    except:  # noqa: E722
        return ""

    # See if we can match against specifc Metal generations
    match = re.match(r"apple m\d", cpu_brand)

    if match:
        cpu = match[0].lower().strip().replace(" ", "-")

        return f"-mcpu={cpu}"

    return "-mcpu=apple-m1"


def _compiler_supports_openmp():
    """Test openmp support by trying to compile a minimal piece of code."""
    cc = os.environ.get("CC", sysconfig.get_config_var("CC"))
    if cc is None:
        return False

    test_code = r"""
    #include <omp.h>
    int main(void) {
        return omp_get_max_threads();
    }
    """

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "test.c")
        exe = os.path.join(d, "test")
        with open(src, "w") as f:
            f.write(test_code)

        cmd = [cc, src, "-fopenmp", "-o", exe]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except Exception:  # noqa: BLE001
            return False

    return True


if not os.environ.get("DRIFT_NO_OPENMP"):
    if _compiler_supports_openmp():
        # OpenMP flags are required by both the compiler and the linker
        COMPILE_FLAGS.append("-fopenmp")
    else:
        cc = os.environ.get("CC", sysconfig.get_config_var("CC"))
        print(
            f"Compiler `{cc}` does not support OpenMP. "
            "If an OpenMP-supporting compiler is available, "
            "add it to your PATH or use `CC=<compiler> pip install ...` "
            "Alternatively, to suppress this warning, set the environment "
            "variable DRIFT_NO_OPENMP to any truth-like value."
        )

if "-fopenmp" not in COMPILE_FLAGS and platform.system().lower() == "darwin":
    COMPILE_FLAGS.extend(("-framework", "Accelerate"))

COMPILE_FLAGS.append(get_mcpu_flag())

# Cython module for fast operations
extensions = [
    Extension(
        "drift.util._fast_tools",
        ["drift/util/_fast_tools.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=COMPILE_FLAGS,
        extra_link_args=COMPILE_FLAGS,
    ),
]

setup(
    name="driftscan",
    ext_modules=cythonize(extensions),
)
