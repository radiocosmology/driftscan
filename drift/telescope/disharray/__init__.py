"""
Implementations of customisable, dish-array transit telescopes with
support for multi-pointed surveys. Mixin classes for adding this
functionality to generic :py:class:`drift.core.telescope.TransitTelescope`'s
are provided.

Mixins
======
- :py:class:`.core.DishArrayMixin`
- :py:class:`.core.MultiElevationSurvey`

Concrete Implementations
========================
- :py:class:`.core.DishArray`
- :py:class:`.core.PolarisedDishArray`
- :py:class:`.core.PolarisedDishArraySurvey`
- :py:class:`.core.UnpolarisedDishArray`
- :py:class:`.core.UnpolarisedDishArraySurvey`

.. note::

    For subclassing, it is recommended to generate your own subclass of a
    :py:class:`drift.core.telescope.TransitTelescope` and add the mixins
    if desired. Directly subclassing the implementations provided may lead to
    unexpected behaviour. In particular, the :py:class:`.core.MultiElevationSurvey` mixin
    works by reconstructing a telescope object using its own configuration and the
    class directly above it in it's class hierarchy. Unexpected
    behaviour might occur if the mixin is not at the bottom of the class hierarchy.

.. autosummary::
    :toctree:

    core
    beams
    layouts

Example Usage
=============

For a concrete example of a 10x10 disharray, multi-pointed survey:

.. code-block:: yaml

    # In a drift-makeproducts configuration file:

    telescope:

      # drift.telescope.disharray.core.PolarisedDishArraySurvey
      type: PolarisedDishArraySurvey

      # Configuration options inherited from
      # drift.core.telescope.TransitTelescope

      freq_lower: 600
      freq_upper: 700
      freq_mode: edge
      num_freq: 64

      ndays: 120 # Note for the multi-elevation survey this is now ndays per pointing.
      tsys_flat: 50
      maxlength: 50

      # Configuration sections for functionality provided by
      # drift.telescope.disharray.core.MultiElevationSurvey Mixin
      #
      # This simulates 7 elevation pointings from -10 to 10 degrees
      # off the telescope zenith (or fiducial) pointing by replicating
      # baselines for each pointing and adjusting the primary beams
      elevation_start: -10
      elevation_stop: 10
      npointings: 7

      # Configuration parameters and sections for functionality provided by
      # drift.telescope.disharray.core.DishArrayMixin Mixin

      # Set array location
      latitude: -30
      longitude: 0
      altitude: 1000

      # Minimum inter-feed spacing in metres for m, l limit calculations
      min_u: 6.5
      min_v: 8.5

      beam_spec:
        # Gaussian beam with FWHM = lambda/(6 m)
        # (See other options and examples in docs for
        # drift.telescope.disharray.beams)
        type: gaussian
        diameter: 6 # effective dish diameter in metres

      layout_spec:
        # 10 x 10 array with 6.5 feed separation in the EW direction and
        # 8.5 m feed separation in the NS direction.
        # (See other options and examples in docs for
        # drift.telescope.disharray.layouts)
        type: grid
        grid_ew: 10
        grid_ns: 10
        spacing_ew: 6.5 # metres
        spacing_ns: 8.5 # metres

"""
