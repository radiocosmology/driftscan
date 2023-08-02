import numpy as np
from scipy.special import jn

from caput import config

from cora.util import coord, units
from drift.core import telescope
from drift.util import util


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


def gaussian_beam(angpos, pointing, fwhm):
    sigma = np.radians(fwhm) / (8.0 * np.log(2.0)) ** 0.5
    x2 = (1.0 - coord.sph_dot(angpos, pointing) ** 2) / (4 * sigma**2)

    return np.exp(-x2)


class FocalPlaneArray(telescope.UnpolarisedTelescope):
    beam_num_u = config.Property(proptype=int, default=10)
    beam_num_v = config.Property(proptype=int, default=10)

    beam_spacing_u = config.Property(proptype=float, default=0.1)
    beam_spacing_v = config.Property(proptype=float, default=0.1)

    beam_size = config.Property(proptype=float, default=0.1)
    beam_pivot = config.Property(proptype=float, default=400.0)

    beam_freq_scale = config.Property(proptype=bool, default=True)

    square_beam = config.Property(proptype=bool, default=False)

    @property
    def beam_pointings(self):
        pnt_u = self.beam_spacing_u * (
            np.arange(self.beam_num_u) - (self.beam_num_u - 1) / 2.0
        )
        pnt_v = self.beam_spacing_v * (
            np.arange(self.beam_num_v) - (self.beam_num_v - 1) / 2.0
        )

        pnt_u = np.radians(pnt_u) + self.zenith[1]
        pnt_v = np.radians(pnt_v) + self.zenith[0]

        pnt = np.zeros((self.beam_num_u, self.beam_num_v, 2))
        pnt[:, :, 1] = pnt_u[:, np.newaxis]
        pnt[:, :, 0] = pnt_v[np.newaxis, :]

        return pnt.reshape(-1, 2)

    # == Methods for calculating the unique baselines ===

    @util.cache_last
    def beam_gaussian(self, feed, freq):
        pointing = self.beam_pointings[feed]
        if self.beam_freq_scale:
            fwhm = self.beam_size * self.frequencies[freq] / self.beam_pivot
        else:
            fwhm = self.beam_size

        return gaussian_beam(self._angpos, pointing, fwhm)

    @util.cache_last
    def beam_square(self, feed, freq):
        pointing = self.beam_pointings[feed]
        bdist = self._angpos - pointing[np.newaxis, :]
        bdist = np.abs(
            np.where(
                (bdist[:, 1] < np.pi)[:, np.newaxis],
                bdist,
                bdist - np.array([0, 2 * np.pi])[np.newaxis, :],
            )
        ) / np.radians(self.beam_size)
        # bdist = np.abs(np.where((bdist[:, 1] < np.pi)[:, np.newaxis], bdist, bdist - np.array([0, 2*np.pi])[np.newaxis, :])) / np.radians(self.beam_size)
        beam = np.logical_and(bdist[:, 0] < 0.5, bdist[:, 1] < 0.5).astype(np.float64)

        return beam

    def beam(self, feed, freq):
        if self.square_beam:
            return self.beam_square(feed, freq)
        else:
            return self.beam_gaussian(feed, freq)

    @property
    def dish_width(self):
        lpivot = units.c / self.beam_pivot * 1e-6
        return lpivot / np.radians(self.beam_size)

    @property
    def u_width(self):
        return self.dish_width

    @property
    def v_width(self):
        return self.dish_width

    @property
    def nfeed(self):
        return self.beam_num_u * self.beam_num_v

    @property
    def feedpositions(self):
        """Feed positions (all zero in FPA)."""
        return np.zeros([self.nfeed, 2])

    def _unique_beams(self):
        beam_mask = np.identity(self.nfeed, dtype=bool)
        beam_map = telescope._remap_keyarray(
            np.diag(np.arange(self.nfeed)), mask=beam_mask
        )

        return beam_map, beam_mask
