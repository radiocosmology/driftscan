
import numpy as np

import telescope
import visibility




class CylinderTelescope(telescope.TransitTelescope):
    """Common functionality for all Cylinder Telescopes.
    """

    num_cylinders = 2
    num_feeds = 6

    cylinder_width = 20.0
    feed_spacing = 0.5

    in_cylinder = True

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
        ub, ind, inv = np.unique(bl1[..., 0] + 1.0J * bl1[..., 1], return_index=True, return_inverse=True)

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

        
        pos = np.empty([self.num_feeds, 2], dtype=np.float64)

        pos[:,0] = cylinder_index * self.cylinder_width
        pos[:,1] = np.arange(self.num_feeds) * self.feed_spacing

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

