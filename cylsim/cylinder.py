
import numpy as np

import telescope
import visibility




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

    num_cylinders = 2
    num_feeds = 6

    cylinder_width = 20.0
    feed_spacing = 0.5

    in_cylinder = True

    touching = True
    cylspacing = None

    non_commensurate = False

    ## u-width property override
    @property
    def u_width(self):
        return self.cylinder_width

    ## v-width property override
    @property
    def v_width(self):
        return 0.0


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
        ub, ind, inv = np.unique(np.around(bl1[..., 0] + 1.0J * bl1[..., 1], 5),
                                 return_index=True, return_inverse=True)

        # Bin to find redundancy of each pair
        redundancy = np.bincount(inv)

        # Construct array of pairs
        upairs = feedpairs[:,ind]

        if not self.in_cylinder:
            mask = np.where(bl1[ind, 0] != 0.0)
            upairs = upairs[:, mask][:, 0, ...]
            redundancy = redundancy[mask]
        
        return upairs, redundancy



    @property
    def feedpositions(self):
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

        pos[:,0] = cylinder_index * self.cylinder_spacing
        pos[:,1] = np.arange(nf) * sp

        return pos


    
    _bc_freq = None
    _bc_nside = None

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
            self._bc_map = visibility.cylinder_beam(self._angpos, self.zenith,
                                                    self.cylinder_width / self.wavelengths[freq])

            self._bc_freq = freq
            self._bc_nside = self._nside

        return self._bc_map
            
    beamx = beam
    beamy = beam


class UnpolarisedCylinderTelescope(CylinderTelescope, telescope.UnpolarisedTelescope):
    """A complete class for an Unpolarised Cylinder telescope.
    """
    pass




class PolarisedCylinderTelescope(CylinderTelescope, telescope.PolarisedTelescope):
    """A complete class for an Unpolarised Cylinder telescope.
    """
    pass



class CylBT(UnpolarisedCylinderTelescope):
    """A cylinder class which ignores large baseline correlations.
    
    Attributes
    ----------
    maxlength : scalar
        Maximum baseline length to include (in metres).
    """
    minlength = 0.0
    maxlength = 20.0

    def _get_unique(self, feedpairs):

        upairs, redundancy = UnpolarisedCylinderTelescope._get_unique(self, feedpairs)

        bl1 = self.feedpositions[upairs[0]] - self.feedpositions[upairs[1]]

        blength = np.hypot(bl1[:,0], bl1[:,1])

        mask = np.where(np.logical_and(blength >= self.minlength, blength < self.maxlength))

        return upairs[:,mask][:, 0, ...], redundancy[mask]
