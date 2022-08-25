import os.path

import numpy as np
from scipy.special import jn

from caput import config

from cora.util import coord
from drift.core import telescope


def jinc(x):
    return 0.5 * (jn(0, x) + jn(2, x))


def beam_circular(angpos, zenith, uv_diameter):
    """Beam pattern for a circular dish.

    Parameters
    ----------
    angpos : np.ndarray
        Array of angular positions
    zenith : np.ndarray
        Co-ordinates of the zenith.
    uv_diameter : scalar
        Diameter of the dish (in units of wavelength).

    Returns
    -------
    beam : np.ndarray
        Beam pattern at each position in angpos.
    """

    x = (1.0 - coord.sph_dot(angpos, zenith) ** 2) ** 0.5 * np.pi * uv_diameter

    return 2 * jinc(x)


class GmrtArray(telescope.TransitTelescope):
    """A Telescope describing an interferometric array of dishes.

    Attributes
    ----------
    gridu, gridv : integer
        Number of dishes in u and v directions.
    dish_width : scalar
        Width of the dish in metres.
    """

    fwhm = 3.1  # degrees

    freq_lower = 139.33
    freq_upper = 156.00
    num_freq = 64

    _pos_file = os.path.dirname(__file__) + "/gmrtpositions.dat"
    _compact = True

    _bc_freq = None
    _bc_nside = None

    _positions = None

    pointing = config.Property(proptype=float, default=0.0)

    dish_width = 45.0

    tsys_flat = 582.0

    minlength = 0.0
    maxlength = 600.0

    def __init__(self, pointing=0.0):
        super(GmrtArray, self).__init__(latitude=19.09, longitude=74.05)

        self._positions = np.loadtxt(self._pos_file)
        # self._positions = self._positions[np.where((self._positions**2).sum(axis=1)**0.5 < 1000)]
        self.pointing = pointing

    @property
    def u_width(self):
        return self.dish_width

    @property
    def v_width(self):
        return self.dish_width

    def beam(self, feed, freq):
        """Beam for a particular feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            A Healpix map (of size self._nside) of the beam. Potentially
            complex.
        """

        if self._bc_freq != freq or self._bc_nside != self._nside:
            sigma = (
                np.radians(self.fwhm)
                / (8.0 * np.log(2.0)) ** 0.5
                / (self.frequencies[freq] / 150.0)
            )

            pointing = np.array(
                [np.pi / 2.0 - np.radians(self.pointing), self.zenith[1]]
            )

            x2 = (1.0 - coord.sph_dot(self._angpos, pointing) ** 2) / (4 * sigma**2)
            self._bc_map = np.exp(-x2)

            self._bc_freq = freq
            self._bc_nside = self._nside

        return self._bc_map

    beamx = beam
    beamy = beam

    @property
    def _single_feedpositions(self):
        """The set of feed positions in the CMU telescope.

        Returns
        -------
        feedpositions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """
        if self._positions is None:
            self._positions = np.loadtxt(self._pos_file)

        return self._positions


class GmrtUnpolarised(GmrtArray, telescope.SimpleUnpolarisedTelescope):
    """Unpolarised GMRT class."""

    pass
