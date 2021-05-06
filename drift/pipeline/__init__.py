"""
A very simple pipeline for analysis of noiseless simulation data.

For most uses you should consider using
`draco <https://github.com/radiocosmology/draco/>`_ which is much more flexible
and sophisticated.

.. autosummary::
    :toctree:

    pipeline
    timestream
"""

import warnings

warnings.warn(
    "This pipeline code is deprecated and will be removed. Use draco instead.",
    DeprecationWarning,
)
