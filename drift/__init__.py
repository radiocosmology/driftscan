"""
driftscan - transit radio interferometertry analysis

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    core
    pipeline
    scripts
    telescope
    util
"""

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
