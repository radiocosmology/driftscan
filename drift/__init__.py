"""
Modelling for transit radio telescopes.

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

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
