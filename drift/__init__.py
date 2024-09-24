"""Modelling for transit radio telescopes.

The existing code is mostly focussed on interferometers but can also be used
for multi-beam transit telescopes.

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

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("driftscan")
except PackageNotFoundError:
    # package is not installed
    pass

del version, PackageNotFoundError
