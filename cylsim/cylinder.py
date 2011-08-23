import numpy as np

class CylinderTelescope(object):


    num_cylinders = 2
    num_feeds = 20

    cylinder_width = 20.0
    feed_spacing = 0.5

    feedx = np.array([1.0, 0.0])
    feedy = np.array([0.0, 1.0])

    freq_lower = 400.0
    freq_upper = 800.0

    num_freq = 20


    def __init__(self, latitude=45, longitude=0):
        """Initialise a cylinder object.
        
        Parameters
        ----------
        latitude, longitude : scalar
        Position on the Earths surface of the telescope (in degrees).
        """

        self.zenith = np.array([np.pi / 2.0 - np.radians(latitude),
                                np.remainder(np.radians(longitude), 2*np.pi)])


    def baselines(self):
        """Calculate all the unique baselines and their redundancies.

        Returns
        -------
        baselines : np.ndarray
            An array of all the baselines. Packed as [ [u1, v1], [u2, v2], ...]

        redundancy : np.ndarray
            For each baseline give the number of pairs if feeds, that have it.
        """

        feed_pos = self.feed_positions()
        
        bl1 = feed_pos[np.newaxis,:,:] - feed_pos[:,np.newaxis,:]
        bl2 = bl1[np.triu_indices(feed_pos.shape[0], 1)]

        bl3, ind = np.unique(bl2[...,0] + 1.0J * bl2[...,1], return_inverse=True)

        baselines = np.empty([bl3.shape[0], 2], dtype=np.float64)
        baselines[:,0] = bl3.real
        baselines[:,1] = bl3.imag

        redundancy = np.bincount(ind)

        return baselines, redundancy
        

    def feed_positions(self):

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
        
        
        
        
