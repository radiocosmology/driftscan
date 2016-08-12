from setuptools import setup, find_packages

import drift

setup(
    name='driftscan',
    version=drift.__version__,

    packages=find_packages(),
    requires=['numpy', 'scipy', 'healpy', 'h5py', 'caput>=0.3.0', 'cora'],
    package_data={'drift.telescope': ['gmrtpositions.dat'] },
    scripts=['scripts/drift-makeproducts', 'scripts/drift-runpipeline'],

    # metadata for upload to PyPI
    author="J. Richard Shaw",
    author_email="richard@phas.ubc.ca",
    description="Transit telescope analysis with the m-mode formalism",
    license="GPL v3.0",
    url="http://github.com/radiocosmology/driftscan"
)
