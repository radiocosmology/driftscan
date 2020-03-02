# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from future.utils import bytes_to_native_str
from setuptools import setup, find_packages

import versioneer

drift_data = {
    # TODO: Py3 remove this hack needed to work around a setuptools bug
    bytes_to_native_str(b"drift.telescope"): ["gmrtpositions.dat"]
}

setup(
    name="driftscan",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=[
        "numpy>=1.7",
        "scipy",
        "healpy>=1.8",
        "h5py",
        "caput>=0.3",
        "click",
        "cora",
    ],
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
