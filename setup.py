from setuptools import setup, find_packages

setup(
    name = 'driftscan',
    version = 0.1,

    packages = find_packages(),
    requires = ['numpy', 'scipy', 'healpy', 'h5py', 'caput', 'cora'],
    package_data = {'drift.telescope' : ['gmrtpositions.dat'] },
    scripts = ['scripts/drift-makeproducts', 'scripts/drift-runpipeline'],

    # metadata for upload to PyPI
    author = "J. Richard Shaw",
    author_email = "jrs65@cita.utoronto.ca",
    description = "Transit telescope analysis with the m-mode formalism",
    license = "GPL v3.0",
    url = "http://github.com/CITA/driftscan"
)
