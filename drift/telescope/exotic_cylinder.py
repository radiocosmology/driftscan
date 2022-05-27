import numpy as np

from caput import config
from caput.cache import cached_property

from drift.telescope import cylinder, cylbeam


class RandomCylinder(cylinder.UnpolarisedCylinderTelescope):

    pos_sigma = 0.5

    def feed_positions_cylinder(self, cylinder_index):

        pos = super(RandomCylinder, self).feed_positions_cylinder(cylinder_index)

        rs = np.random.get_state()
        np.random.seed(cylinder_index)

        p1 = np.sort(
            pos[:, 1]
            + self.pos_sigma
            * self.feed_spacing
            * np.random.standard_normal(pos.shape[0])
        )

        np.random.set_state(rs)

        pos[:, 1] = p1
        return pos


class GradientCylinder(cylinder.UnpolarisedCylinderTelescope):

    min_spacing = config.Property(proptype=float, default=-1.0)
    max_spacing = config.Property(proptype=float, default=20.0)

    def feed_positions_cylinder(self, cylinder_index):

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        nf = self.num_feeds

        # Parameters for gradient feedspacing
        a = self.wavelengths[-1] / 2.0 if self.min_spacing < 0.0 else self.min_spacing
        # b = 2 * (sp - a) / nf
        b = 2.0 * (self.max_spacing - a * (nf - 1)) / (nf - 1) ** 2.0

        pos = np.empty([nf, 2], dtype=np.float64)

        i = np.arange(nf)

        pos[:, 0] = cylinder_index * self.cylinder_spacing
        pos[:, 1] = a * i + 0.5 * b * i ** 2

        return pos


class CylinderExtra(cylinder.UnpolarisedCylinderTelescope):

    extra_feeds = config.Property(proptype=np.array, default=[])

    def feed_positions_cylinder(self, cylinder_index):

        pos = super(CylinderExtra, self).feed_positions_cylinder(cylinder_index)

        nextra = self.extra_feeds.shape[0]

        pos2 = np.zeros((pos.shape[0] + nextra, 2), dtype=np.float64)

        pos2[nextra:] = pos

        pos2[:nextra, 0] = cylinder_index * self.cylinder_spacing
        pos2[:nextra, 1] = self.extra_feeds

        return pos2


class CylinderPerturbed(cylinder.PolarisedCylinderTelescope):
    """A base for a polarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.

    Methods
    -------
    beamx, beamy : methods
        (abstract methods) Routines giving the field pattern for the x and y feeds.
    """

    npert = 2

    @property
    def beamclass(self):
        """Simple beam mode of dual polarisation feeds."""
        nsfeed = self._single_feedpositions.shape[0]

        # Evens are X feed for each pert, Odds are Y feed
        beamclass = [bc * np.ones(nsfeed) for bc in range(2 * self.npert)]

        return np.concatenate(beamclass).astype(np.int64)

    @property
    def feedpositions(self):
        beampos = [self._single_feedpositions for bc in range(2 * self.npert)]
        return np.concatenate(beampos)

    def beamx(self, feed, freq, angpos=None):
        """Beam for the x polarisation feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.
        angpos : np.ndarray[nposition, 2], optional
            Angular position on the sky (in radians). If not provided, default to the
            _angpos class attribute.

        Returns
        -------
        beam : np.ndarray
            Healpix maps (of size [self._nside, 2]) of the field pattern in the
            theta and phi directions.
        """
        beampert = int(self.beamclass[feed] // 2)

        if beampert == 0:

            return cylbeam.beam_x(
                self._angpos if angpos is None else angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_e,
                self.fwhm_h,
            )

        elif beampert == 1:

            beam0 = cylbeam.beam_x(
                self._angpos if angpos is None else angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_e,
                self.fwhm_h,
            )

            beam1 = cylbeam.beam_x(
                self._angpos if angpos is None else angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_e * 1.01,
                self.fwhm_h,
            )

            dbeam = (beam1 - beam0) / (0.01 * self.fwhm_e)

            return dbeam

    def beamy(self, feed, freq, angpos=None):
        """Beam for the x polarisation feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.
        angpos : np.ndarray[nposition, 2], optional
            Angular position on the sky (in radians). If not provided, default to the
            _angpos class attribute.

        Returns
        -------
        beam : np.ndarray
            Healpix maps (of size [self._nside, 2]) of the field pattern in the
            theta and phi directions.
        """

        beampert = int(self.beamclass[feed] // 2)

        if beampert == 0:

            return cylbeam.beam_y(
                self._angpos if angpos is None else angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_e,
                self.fwhm_h,
            )

        elif beampert == 1:

            beam0 = cylbeam.beam_y(
                self._angpos if angpos is None else angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_e,
                self.fwhm_h,
            )

            beam1 = cylbeam.beam_y(
                self._angpos if angpos is None else angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_e * 1.01,
                self.fwhm_h,
            )

            dbeam = (beam1 - beam0) / (0.01 * self.fwhm_e)

            return dbeam


class CylinderShift(cylinder.UnpolarisedCylinderTelescope):

    shift = config.Property(proptype=float, default=0.0)

    def feed_positions_cylinder(self, cylinder_index):

        pos = super(CylinderExtra, self).feed_positions_cylinder(cylinder_index)

        nextra = self.extra_feeds.shape[0]

        pos2 = np.zeros((pos.shape[0] + nextra, 2), dtype=np.float64)

        pos2[nextra:] = pos

        pos2[:nextra, 0] = cylinder_index * self.cylinder_spacing
        pos2[:nextra, 1] = self.extra_feeds

        return pos2


class PolarisedCylinderFitBeam(cylinder.PolarisedCylinderTelescope):
    """Polarised cylinder telescope with freq-dependent primary beam widths, fit to
    more detailed model of CHIME primary beam.

    This code has been adapted from ch_pipeline.core.telescope.CHIMEFitBeam.
    """

    _exwidth = [0.7]
    _eywidth = (
        3.0
        / (2 * np.pi)
        * np.array([1.15310483e-07, -2.30462590e-04, 1.50451290e-01, -3.07440520e01])
    )

    _hywidth = [1.2]
    _hxwidth = (
        3.0
        / (2 * np.pi)
        * np.array([2.97495306e-07, -6.00582101e-04, 3.99949759e-01, -8.66733249e01])
    )

    # Tweak the following two properties to change the beam width
    @cached_property
    def fwhm_ex(self):
        """Full width half max of the E-plane antenna beam for X polarization."""

        return np.polyval(np.array(self._exwidth) * 2.0 * np.pi / 3.0, self.frequencies)

    @cached_property
    def fwhm_hx(self):
        """Full width half max of the H-plane antenna beam for X polarization."""

        return np.polyval(np.array(self._hxwidth) * 2.0 * np.pi / 3.0, self.frequencies)

    @cached_property
    def fwhm_ey(self):
        """Full width half max of the E-plane antenna beam for Y polarization."""

        return np.polyval(np.array(self._eywidth) * 2.0 * np.pi / 3.0, self.frequencies)

    @cached_property
    def fwhm_hy(self):
        """Full width half max of the H-plane antenna beam for Y polarization."""

        return np.polyval(np.array(self._hywidth) * 2.0 * np.pi / 3.0, self.frequencies)

    # @util.cache_last
    def beamx(self, feed, freq, angpos=None):

        return cylbeam.beam_x(
            self._angpos if angpos is None else angpos,
            self.zenith,
            self.cylinder_width / self.wavelengths[freq],
            self.fwhm_ex[freq],
            self.fwhm_hx[freq],
        )

    # @util.cache_last
    def beamy(self, feed, freq, angpos=None):

        return cylbeam.beam_y(
            self._angpos if angpos is None else angpos,
            self.zenith,
            self.cylinder_width / self.wavelengths[freq],
            self.fwhm_ey[freq],
            self.fwhm_hy[freq],
        )
