import numpy as np

from scipy.special import jn

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


class DishArray(telescope.TransitTelescope):
    """A Telescope describing an interferometric array of dishes.

    Attributes
    ----------
    gridu, gridv : integer
        Number of dishes in u and v directions.
    dish_width : scalar
        Width of the dish in metres.
    """

    dish_width = 3.5

    gridu = 4
    gridv = 4

    freq_lower = 1000
    freq_upper = 1200
    num_freq = 100

    _bc_freq = None
    _bc_nside = None

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
            self._bc_map = beam_circular(
                self._angpos, self.zenith, self.dish_width / self.wavelengths[freq]
            )

            self._bc_freq = freq
            self._bc_nside = self._nside

        return self._bc_map

    beamx = beam
    beamy = beam

    @property
    def feedpositions(self):
        """The set of feed positions in the CMU telescope.

        Returns
        -------
        feedpositions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """
        pos = np.zeros((self.gridu, self.gridv, 2))

        for i in range(self.gridu):
            for j in range(self.gridv):
                pos[i, j, 0] = i * self.dish_width
                pos[i, j, 1] = j * self.dish_width

        return pos.reshape((self.gridu * self.gridv, 2))

    def _get_unique(self, feedpairs):
        """Calculate the unique baseline pairs.

        Pairs are considered identical if they have the same baseline
        separation,

        Parameters
        ----------
        fpairs : np.ndarray
            An array of all the feed pairs, packed as [[i1, i2, ...], [j1, j2, ...] ].

        Returns
        -------
        baselines : np.ndarray
            An array of all the unique pairs. Packed as [ [i1, i2, ...], [j1, j2, ...]].
        redundancy : np.ndarray
            For each unique pair, give the number of equivalent pairs.
        """
        # Calculate separation of all pairs, and map into a half plane (so
        # baselines and their negative are identical).
        bl1 = self.feedpositions[feedpairs[0]] - self.feedpositions[feedpairs[1]]
        bl1 = telescope.map_half_plane(bl1)

        # Turn separation into a complex number and find unique elements
        ub, ind, inv = np.unique(
            bl1[..., 0] + 1.0j * bl1[..., 1], return_index=True, return_inverse=True
        )

        # Bin to find redundancy of each pair
        redundancy = np.bincount(inv)

        # Construct array of pairs
        upairs = feedpairs[:, ind]

        return upairs, redundancy


# Commented out, because: The class CMUTelescope does not exist. If you know what that's supposed to be, please fix it.
# class UnpolarisedDishArray(CMUTelescope, telescope.UnpolarisedTelescope):
#     """Unpolarised dish array class."""
#
#     pass
