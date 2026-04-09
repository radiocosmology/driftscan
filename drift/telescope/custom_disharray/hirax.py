"""Implementations of simple models of the HIRAX telescope. Identical
to telescope objects in :py:mod:`drift.telescope.custom_disharray.PolarisedDishArray` and
:py:mod:`drift.telescope.custom_disharray.PolarisedDishArraySurvey` but with
the following HIRAX defaults:

    * Frequency channels set to the HIRAX band (400-800 MHz with 1024 channels)
    * Observatory location set to the (approximate) location of the HIRAX array.
    * A grid array layout with 6.5 m EW spacing and 8.5 m NS spacing for the default
      array in the case of :py:class:`HIRAX` and :py:class:`HIRAXSurvey` or the hex-tile
      array layout in the case of :py:class:`HIRAXHexTile` and :py:class:`HIRAXHexTileSurvey`.

Note that these defaults represent a large telescope object to simulate. For smaller
scale runs use :py:class:`drift.core.telescope.TransitTelescope` configuration
parameters such as `maxlength` to cut down the baselines considered. Additionally
using broader frequency channels over a sub-band is often useful.

"""

from pkg_resources import resource_filename
import abc
from caput import config

from drift.core.telescope import PolarisedTelescope
from .core import MultiElevationSurvey, CustomDishArray

# These are approximate for the Swartfontein site.
HIRAX_LATITUDE = -30.69638889  # 30°41'47.0"S
HIRAX_LONGITUDE = 21.57222222  # 21°34'20.0"E
HIRAX_ALTITUDE = 1113.0  # m


class _HIRAXDefaults(CustomDishArray, config.Reader, metaclass=abc.ABCMeta):
    """Mixin for a HIRAX-like telescope. :py:class:`.core.CustomDishArray` but
    with defaults mentioned above.
    """

    freq_start = config.Property(proptype=float, default=400)
    freq_end = config.Property(proptype=float, default=800)
    num_freq = config.Property(proptype=int, default=1024)

    latitude = config.Property(proptype=float, default=HIRAX_LATITUDE)
    longitude = config.Property(proptype=float, default=HIRAX_LONGITUDE)
    altitude = config.Property(proptype=float, default=HIRAX_ALTITUDE)

    layout_spec = config.Property(
        proptype=dict,
        default={
            "type": "grid",
            "grid_ew": 32,
            "grid_ns": 32,
            "spacing_ew": 6.5,
            "spacing_ns": 8.5,
        },
    )


class HIRAX(_HIRAXDefaults, PolarisedTelescope):
    """Single pointing HIRAX telescope."""

    pass


class HIRAXSurvey(MultiElevationSurvey, HIRAX):
    """A multi-pointed HIRAX survey."""

    pass


class _HIRAXHexTile(_HIRAXDefaults, config.Reader, metaclass=abc.ABCMeta):
    """Mixin for HIRAX defaults but with hex-tile array layout."""

    layout_spec = config.Property(
        proptype=dict,
        default={
            "type": "file",
            "filenames": [
                resource_filename(
                    "drift.telescope.custom_disharray",
                    "data/hirax_hextile_template_1024.dat",
                )
            ],
            "spacing_ew": 6.5,
            "spacing_ns": 8.5,
        },
    )


class HIRAXHexTile(_HIRAXHexTile, PolarisedTelescope):
    """Single pointing HIRAX telescope using the Hex-tile array layout
    (reference to be added).
    """

    pass


class HIRAXHexTileSurvey(MultiElevationSurvey, HIRAXHexTile):
    """A multi-pointed HIRAX survey  using the Hex-tile array layout
    (reference to be added)."""

    pass
