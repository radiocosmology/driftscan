"""Helpers for parameterised construction of array layouts.

The classes provide parameterised array layouts made use of by the
:py:class:`..core.DishArrayMixin` mixin. These can be specified in the
`layout_spec` section of the configuration file. If the
:py:class:`drift.core.telescope.TransitTelescope` is a polarised telescope,
each station will be assumed to have an "X" and "Y" polarised feed.

For example:

.. code-block:: yaml

    # In a drift-makeproducts configuration file:

    telescope:

      # Must be a TransitTelescope (sub)-class with DishArrayMixin mixin.
      type: PolarisedDishArray

      # Any TransitTelescope / DishArrayMixin Parameters
      ...

      layout_spec:
        # 10 x 10 array with 6.5 feed separation in the EW direction and
        # 8.5 m feed separation in the NS direction.
        type: grid
        grid_ew: 10
        grid_ns: 10
        spacing_ew: 6.5 # metres
        spacing_ns: 8.5 # metres


Other examples:

.. code-block:: yaml

    layout_spec:
      # Array layout specified by data files with two whitespace separated
      # columns of EW and NS feed separation (in that order).
      # Expects a list (possibly of length one) of files that will be
      # concatenated.
      type: file
      filenames:
        - /path/to/feed_positions_1.dat
        - /path/to/feed_positions_1.dat

.. code-block:: yaml

    layout_spec:
      # Array layout specified by data files with two whitespace separated
      # columns of EW and NS dimensionless template separations.
      # The actual feed separation will be determined by the spacing_ew and
      # spacing_ns parameters multiplied by the template separations.
      # Useful for trying the same layout with different spacings.
      type: file
      filenames:
        - /path/to/feed_positions_template.dat
      spacing_ew: 10 # metres
      spacing_ns: 20 # metres

Currently supported `layout_spec` types are:

- `grid` provided by :py:class:`GridLayout`
- `file` provided by :py:class:`SimpleLayoutFile`

See their class and base class definitions for more parameter options.

"""

from typing import Optional

import numpy as np
from caput import config


class GridLayout(config.Reader):
    """A simple grid layout specified by grid dimensions and spacing.

    Attributes
    ----------
    grid_ew: :py:class:`caput.config.Property(proptype=int)`
        Number of columns of dishes along the EW direction
    grid_ns: :py:class:`caput.config.Property(proptype=int)`
        Number of rows of dishes along the NS direction
    spacing_ew: :py:class:`caput.config.Property(proptype=float)`
        Spacing between the EW columns in metres.
        Default: 6 m
    spacing_ns: :py:class:`caput.config.Property(proptype=float)`
        Spacing between the NS rows in metres.
        Default: 6 m
    """

    grid_ew = config.Property(proptype=int, default=3)
    grid_ns = config.Property(proptype=int, default=3)
    spacing_ew = config.Property(proptype=float, default=6.0)
    spacing_ns = config.Property(proptype=float, default=6.0)

    def __init__(self, expand_pol: Optional[bool] = False):
        self.expand_pol = expand_pol

    def _finalise_config(self):
        n_dish = self.grid_ew * self.grid_ns
        d_ew = -self.spacing_ew * (np.arange(n_dish) % self.grid_ew)
        d_ns = self.spacing_ns * (np.arange(n_dish) // self.grid_ew)

        self.feedpositions = np.column_stack([d_ew, d_ns])
        self.polarisation = None

        if self.expand_pol:
            self.polarisation = np.tile(["X", "Y"], len(self.feedpositions))
            self.feedpositions = np.repeat(self.feedpositions, 2, axis=0)


class SimpleLayoutFile(config.Reader):
    """Read in an array layout from a list of files. The element positions
    from the files will be concatenated.

    File should be arranged such that is can be read in with:

    .. code-block:: python

        ew, ns = np.loadtxt(filename, unpack=True)

    Ie. whitespace delimited data with the EW offset positions in the first column
    and NS offset positions in the second column. Units are assumed metres.

    Attributes
    ----------
    filenames: :py:class:`caput.config.Property(proptype=list)`
        List of filenames to read array layout from.
    spacing_ew, spacing_ns: :py:class:`caput.config.Property(proptype=float)`
        Multiplicative factor for EW and NS dish positions respectively.
        Useful if locations in the files are a template with the spacing factored
        out so that is can be changed with these parameters.
        Default: 1 (Assumes positions in file have the desired spacing in metres)
    """

    filenames = config.list_type(type_=str, default=[])
    spacing_ew = config.Property(proptype=float, default=1.0)
    spacing_ns = config.Property(proptype=float, default=1.0)

    def __init__(self, expand_pol: Optional[bool] = False):
        self.expand_pol = expand_pol

    def _finalise_config(self):
        d_ew, d_ns = [], []

        for filename in self.filenames:
            curr_d_ew, curr_d_ns = np.loadtxt(filename, unpack=True)
            d_ew.extend(curr_d_ew)
            d_ns.extend(curr_d_ns)
        d_ew = np.array(d_ew)
        d_ns = np.array(d_ns)

        self.feedpositions = np.column_stack(
            [self.spacing_ew * d_ew, self.spacing_ns * d_ns]
        )
        self.polarisation = None

        if self.expand_pol:
            self.polarisation = np.tile(["X", "Y"], len(self.feedpositions))
            self.feedpositions = np.repeat(self.feedpositions, 2, axis=0)


AVAILABLE_LAYOUTS = {
    "file": SimpleLayoutFile,
    "grid": GridLayout,
}
