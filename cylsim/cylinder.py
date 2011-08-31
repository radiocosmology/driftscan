import numpy as np

import hputil
import visibility

class CylinderTelescope(object):


    num_cylinders = 2
    num_feeds = 10

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

        self._init_trans(2)



    _baselines = None

    @property
    def baselines(self):

        if self._baselines == None:
            self.calculate_baselines()

        return self._baselines


    def calculate_baselines(self):
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

        self._baselines = baselines
        self._redundancy = redundancy



    _frequencies = None

    @property
    def frequencies(self):

        if self._frequencies== None:
            self.calculate_frequencies()

        return self._frequencies

    def calculate_frequencies(self):

        self._frequencies = np.linspace(self.freq_lower, self.freq_upper, self.num_freq)


        

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
        
    def _init_trans(self, nside):

        # Angular positions in healpix map of nside
        self._nside = nside
        self._angpos = hputil.ang_positions(nside)

        # The horizon function
        self._horizon = visibility.horizon(self._angpos, self.zenith)

        # Differentiate beams
        self._beamx = visibility.cylinder_beam(self._angpos, self.zenith, self.cylinder_width)
        self._beamy = visibility.cylinder_beam(self._angpos, self.zenith, self.cylinder_width)

        # Polarisation projections of feed pairs
        self._pIQUxx = visibility.pol_IQU(self._angpos, self.zenith, self.feedx, self.feedx)
        self._pIQUxy = visibility.pol_IQU(self._angpos, self.zenith, self.feedx, self.feedy)
        self._pIQUyy = visibility.pol_IQU(self._angpos, self.zenith, self.feedy, self.feedy)

        # Multiplied pairs
        self._mIQUxx = self._horizon * self._beamx * self._beamx * self._pIQUxx
        self._mIQUxy = self._horizon * self._beamx * self._beamy * self._pIQUxy
        self._mIQUyy = self._horizon * self._beamy * self._beamy * self._pIQUyy

    def _best_nside(self, lmax):
        nside = int(2**(np.ceil(np.log((lmax + 1) / 3.0) / np.log(2.0))))
        return nside

    #@profile
    def _transfer_single(self, uv, lmax, lside):

        if self._nside != self._best_nside(lmax):
            self._init_trans(self._best_nside(lmax))

        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        cvIQUxx = self._mIQUxx * fringe
        cvIQUxy = self._mIQUxy * fringe
        cvIQUyy = self._mIQUyy * fringe

        ### If beams ever become complex need to do yx combination.
        btransxx = hputil.sphtrans_complex_pol(cvIQUxx, centered = False,
                                               lmax = int(lmax), lside=lside)
        btransxy = hputil.sphtrans_complex_pol(cvIQUxy, centered = False,
                                               lmax = int(lmax), lside=lside)
        btransyy = hputil.sphtrans_complex_pol(cvIQUyy, centered = False,
                                               lmax = int(lmax), lside=lside)

        return [btransxx, btransxy, btransyy]

    #@profile
    def transfer_matrices(self, bl_indices, f_indices):
        import pdb
        from progressbar import ProgressBar

        progress = ProgressBar()

        if np.shape(bl_indices) != np.shape(f_indices):
            raise Exception("Index arrays must be the same shape.")

        if np.min(bl_indices) < 0 or np.max(bl_indices) >= self.baselines.shape[0]:
            raise Exception("Baseline indices aren't valid")

        if np.min(f_indices) < 0 or np.max(f_indices) >= self.frequencies.shape[0]:
            raise Exception("Frequency indices aren't valid")

        wavelength = 3e2 / self.frequencies[np.array(f_indices)]
        uv_arr = self.baselines[np.array(bl_indices)] / wavelength[...,np.newaxis]  # Replace with constant c

        mmax = np.ceil(2 * np.pi * (uv_arr[...,1] + (self.cylinder_width / wavelength)))
        lmax = np.ceil((mmax**2 + (2*np.pi*uv_arr[...,0])**2)**0.5)

        all_lmax = lmax.max()

        tarray = np.zeros(np.shape(bl_indices) + (3, 3, all_lmax+1, 2*all_lmax+1) ,dtype=np.complex128)
        print "Size: %i elements. Memory %f GB." % (tarray.size, tarray.size * 8.0 / 2**30)

        i_arr = np.argsort(lmax.flat)

        for i in progress(range(i_arr.size)):
            ind = np.unravel_index(i_arr[i], lmax.shape)
            trans = self._transfer_single(uv_arr[ind], lmax[ind], all_lmax)
            #tarray[ind] = np.array(trans)
            tarray[ind,0,0] = trans[0][0]
            tarray[ind,0,1] = trans[0][1]
            tarray[ind,0,2] = trans[0][2]

            tarray[ind,1,0] = trans[1][0]
            tarray[ind,1,1] = trans[1][1]
            tarray[ind,1,2] = trans[1][2]

            tarray[ind,2,0] = trans[2][0]
            tarray[ind,2,1] = trans[2][1]
            tarray[ind,2,2] = trans[2][2]

        return tarray


    def transfer_for_freq(self, freq):

        if freq < 0 or freq >= self.num_freq:
            raise Exception("Frequency index not valid.")

        bi = np.arange(self.baselines.shape[0])
        fi = freq * np.ones_like(bi)

        return self.transfer_matrices(bi, fi)

        

        
        

        
