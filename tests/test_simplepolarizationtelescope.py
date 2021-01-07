"""Tests for SimplePolarizationTelescope."""

from drift.core.telescope import SimplePolarisedTelescope

import numpy as np
import pytest


class DummyTelescope(SimplePolarisedTelescope):
    """A dummy implementation of SimplePolarisedTelescope in order to test the abstract class."""

    @property
    def _single_feedpositions(self):
        """
        Feed positions relative to an arbitrary point (in m).

        These are all 1m. This also sets the number of feeds.

        Returns
        -------
        np.ndarray : Array of feedpositions.

        """
        return np.ones((10, 2))

    @property
    def beamx(self, feed, freq):
        """Beam for the X polarisation feed.

        These are all 0.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            Healpix maps (of size [self._nside, 2]) of the field pattern in the
            theta and phi directions.

        """
        return np.zeros((10, 2))

    @property
    def beamy(self, feed, freq):
        """Beam for the y polarisation feed.

        These are all 0.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            Healpix maps (of size [self._nside, 2]) of the field pattern in the
            theta and phi directions.

        """
        return np.zeros((10, 2))

    @property
    def u_width(self):
        """
        Approximate physical width (in the u-direction) of the dish/telescope etc.

        For calculating the maximum (l,m).

        Returns
        -------
        float : width in u-direction.

        """
        return 5.5

    @property
    def v_width(self):
        """
        Approximate physical length (in the v-direction) of the dish/telescope etc.

        For calculating the maximum (l,m).

        Returns
        -------
        float : width in v-direction.

        """
        return 10.1


@pytest.fixture(scope="module")
def spt():
    """
    Fixture to supply a SimplePolarizedTelescope to the tests.

    Returns
    -------
    DummyTelescope : a dummy implementation of SimplePolarizedTelescope.

    """
    return DummyTelescope()


def test_polarisation_map(spt):
    """Test that the polarization map is ['x','x','x',...,'y','y','y']."""
    expected_pol_map = np.concatenate(
        (np.full(spt.nfeed // 2, "X"), np.full(spt.nfeed // 2, "Y"))
    )
    assert np.all(spt.polarisation == expected_pol_map)
