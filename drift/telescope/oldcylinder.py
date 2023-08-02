import numpy as np

from caput import config

from cora.util import coord

from drift.core import telescope, visibility


class CylinderTelescope(telescope.TransitTelescope):
    """Common functionality for all Cylinder Telescopes.

    Attributes
    ----------
    num_cylinders : integer
        The number of cylinders.
    num_feeds : integer
        Number of regularly spaced feeds along each cylinder.
    cylinder_width : scalar
        Width in metres.
    feed_spacing : scalar
        Gap between feeds in metres.
    in_cylinder : boolean
        Include in cylinder correlations?
    touching : boolean
        Are the cylinders touching (no spacing between them)?
    cylspacing : scalar
        If not `touching` this is the spacing in metres.
    """

    num_cylinders = config.Property(proptype=int, default=2)
    num_feeds = config.Property(proptype=int, default=6)

    cylinder_width = config.Property(proptype=float, default=20.0)
    feed_spacing = config.Property(proptype=float, default=0.5)

    in_cylinder = config.Property(proptype=bool, default=True)

    touching = config.Property(proptype=bool, default=True)
    cylspacing = config.Property(proptype=float, default=0.0)

    non_commensurate = config.Property(proptype=bool, default=False)

    ## u-width property override
    @property
    def u_width(self):
        return self.cylinder_width

    ## v-width property override
    @property
    def v_width(self):
        return 0.0

    def _unique_baselines(self):
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

        base_map, base_mask = super(CylinderTelescope, self)._unique_baselines()

        if not self.in_cylinder:
            # Construct array of indices
            fshape = [self.nfeed, self.nfeed]
            f_ind = np.indices(fshape)

            # Construct array of baseline separations in complex representation
            bl1 = self.feedpositions[f_ind[0]] - self.feedpositions[f_ind[1]]

            ic_mask = np.where(
                bl1[..., 0] != 0.0,
                np.ones(fshape, dtype=bool),
                np.zeros(fshape, dtype=bool),
            )
            base_mask = np.logical_and(base_mask, ic_mask)
            base_map = telescope._remap_keyarray(base_map, base_mask)

        return base_map, base_mask

    @property
    def _single_feedpositions(self):
        """The set of feed positions on *all* cylinders.

        Returns
        -------
        feedpositions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """
        fplist = [self.feed_positions_cylinder(i) for i in range(self.num_cylinders)]

        return np.vstack(fplist)

    @property
    def cylinder_spacing(self):
        if self.touching:
            return self.cylinder_width
        else:
            if self.cylspacing is None:
                raise Exception("Need to set cylinder spacing if not touching.")
            return self.cylspacing

    def feed_positions_cylinder(self, cylinder_index):
        """Get the feed positions on the specified cylinder.

        Parameters
        ----------
        cylinder_index : integer
            The cylinder index, an integer from 0 to self.num_cylinders.

        Returns
        -------
        feed_positions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        nf = self.num_feeds
        sp = self.feed_spacing
        if self.non_commensurate:
            nf = self.num_feeds - cylinder_index
            sp = self.feed_spacing / (nf - 1.0) * nf

        pos = np.empty([nf, 2], dtype=np.float64)

        pos[:, 0] = cylinder_index * self.cylinder_spacing
        pos[:, 1] = np.arange(nf) * sp

        return pos


class UnpolarisedCylinderTelescope(
    CylinderTelescope, telescope.SimpleUnpolarisedTelescope
):
    """A complete class for an Unpolarised Cylinder telescope."""

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

        return visibility.cylinder_beam(
            self._angpos, self.zenith, self.cylinder_width / self.wavelengths[freq]
        )


class PolarisedCylinderTelescope(CylinderTelescope, telescope.SimplePolarisedTelescope):
    """A complete class for an Unpolarised Cylinder telescope."""

    # Change the illuminated width in X and Y
    illumination_x = config.Property(proptype=float, default=1.0)
    illumination_y = config.Property(proptype=float, default=1.0)

    ortho_pol = config.Property(proptype=bool, default=True)

    # @util.cache_last
    def beamx(self, feed, freq):
        bpat = visibility.cylinder_beam(
            self._angpos,
            self.zenith,
            self.illumination_x * self.cylinder_width / self.wavelengths[freq],
        )

        bm = np.zeros_like(self._angpos)
        if self.ortho_pol:
            bm[:, 1] = bpat
        else:
            thatz, phatz = coord.thetaphi_plane_cart(self.zenith)
            thatp, phatp = coord.thetaphi_plane_cart(self._angpos)
            bm[:, 0] = np.dot(thatp, phatz) * bpat
            bm[:, 1] = np.dot(phatp, phatz) * bpat

        return bm

    # @util.cache_last
    def beamy(self, feed, freq):
        bpat = visibility.cylinder_beam(
            self._angpos,
            self.zenith,
            self.illumination_y * self.cylinder_width / self.wavelengths[freq],
        )

        bm = np.zeros_like(self._angpos)
        if self.ortho_pol:
            bm[:, 0] = bpat
        else:
            thatz, phatz = coord.thetaphi_plane_cart(self.zenith)
            thatp, phatp = coord.thetaphi_plane_cart(self._angpos)
            bm[:, 0] = np.dot(thatp, thatz) * bpat
            bm[:, 1] = np.dot(phatp, thatz) * bpat

        return bm
