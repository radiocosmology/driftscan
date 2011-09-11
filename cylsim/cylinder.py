import numpy as np

import hputil
import visibility

class CylinderTelescope(object):


    num_cylinders = 2
    num_feeds = 6

    cylinder_width = 20.0
    feed_spacing = 0.5

    feedx = np.array([1.0, 0.0])
    feedy = np.array([0.0, 1.0])

    freq_lower = 400.0
    freq_upper = 800.0

    num_freq = 50

    accuracy_boost = 1

    print_progress = False

    _extdelkeys = []


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



    def __getstate__(self):

        state = self.__dict__.copy()

        delkeys = ['_baselines', '_redundancy', '_frequencies'] + self._extdelkeys

        for key in delkeys:
            if key in state:
                del state[key]

        return state
            
    

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

        if self._frequencies == None:
            self.calculate_frequencies()

        return self._frequencies

    def calculate_frequencies(self):

        self._frequencies = np.linspace(self.freq_lower, self.freq_upper, self.num_freq)


    _redundancy = None

    @property
    def redundancy(self):

        if self._redundancy == None:
            self.calculate_baselines()

        return self._redundancy
        

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


    def _best_nside(self, lmax):
        nside = int(2**(self.accuracy_boost + np.ceil(np.log((lmax + 1) / 3.0) / np.log(2.0))))
        return nside


    def max_lm(self, freq_index = None):
        """Get the maximum (l,m) that the telescope is sensitive to.
        
        Parameters
        ----------
        freq_index : integer, optional
            The frequency to calculate the maximum's at. If None (default), use
            the maximum frequency, and hence the maximum (l,m) at any frequency.

        Returns
        -------
        lmax, mmax : integer
        """
        
        if freq_index == None:
            freq = self.frequencies.max()
        else:
            freq = self.frequencies[freq_index]

        wavelength = 3e2 / freq

        umax = (np.abs(self.baselines[:,0]).max() + self.cylinder_width) / wavelength
        vmax = np.abs(self.baselines[:,1]).max()  / wavelength

        mmax = np.ceil(2 * np.pi * umax)
        lmax = np.ceil((mmax**2 + (2*np.pi*vmax)**2)**0.5)

        return int(lmax), int(mmax)
        

        
    def transfer_matrices(self, bl_indices, f_indices, mfirst = False, global_lmax = True):

        ## Setup a progress bar if required.
        progress = lambda x: x
        if self.print_progress:
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

        mmax = np.ceil(2 * np.pi * (uv_arr[...,0] + (self.cylinder_width / wavelength)))
        lmax = np.ceil((mmax**2 + (2*np.pi*uv_arr[...,1])**2)**0.5)

        if global_lmax:
            all_lmax, all_mmax = self.max_lm()
        else:
            all_lmax = lmax.max()

        tarray = self._make_matrix_array(np.shape(bl_indices), mfirst, all_lmax)

        print "Size: %i elements. Memory %f GB." % (tarray.size, 2*tarray.size * 8.0 / 2**30)

        i_arr = np.argsort(lmax.flat)

        for i in progress(range(i_arr.size)):
            ind = np.unravel_index(i_arr[i], lmax.shape)
            trans = self._transfer_single(uv_arr[ind], lmax[ind], all_lmax)

            self._copy_single_into_array(tarray, trans, ind, mfirst)

        return tarray


    def transfer_for_frequency(self, freq, mfirst = True):

        if freq < 0 or freq >= self.num_freq:
            raise Exception("Frequency index not valid.")

        bi = np.arange(self.baselines.shape[0])
        fi = freq * np.ones_like(bi)

        return self.transfer_matrices(bi, fi, mfirst = mfirst)


    def transfer_for_baseline(self, baseline, mfirst = True):

        if baseline < 0 or baseline >= self.baselines.shape[0]:
            raise Exception("Frequency index not valid.")

        fi = np.arange(self.num_freq)
        bi = baseline * np.ones_like(fi)

        return self.transfer_matrices(bi, fi, mfirst = mfirst)


class PolarisedCylinderTelescope(CylinderTelescope):


    feedx = np.array([1.0, 0.0])
    feedy = np.array([0.0, 1.0])

    cylinder_xy_ratio = 1.0


    ## Extra fields to remove when pickling.
    _extdelkeys = ['_nside', '_angpos', '_horizon', '_beamx', '_beamy',
                   '_pIQUxx', '_pIQUxy', '_pIQUyy', '_mIQUxx', '_mIQUxy', '_mIQUyy']
        
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


    def _make_matrix_array(self, shape, mfirst, lmax):

        if mfirst:
            tarray = np.zeros((2*lmax+1,) + shape + (3, 3, lmax+1) ,dtype=np.complex128)
        else:
            tarray = np.zeros(shape + (3, 3, lmax+1, 2*lmax+1), dtype=np.complex128)

        return tarray


    def _copy_single_into_array(self, tarray, tsingle, ind, mfirst):
        for pi in range(3):
            for pj in range(3):
                islice = (((slice(None),) + ind + (pi,pj) + (slice(None),))
                          if mfirst else (ind + (pi,pj) + (slice(None),slice(None))))
                tarray[islice] = tsingle[pi][pj].T if mfirst else tsingle[pi][pj]


        
        


class UnpolarisedCylinderTelescope(CylinderTelescope):


    ## Extra fields to remove when pickling.
    _extdelkeys = ['_nside', '_angpos', '_horizon', '_beam', '_mul']

    def _init_trans(self, nside):

        # Angular positions in healpix map of nside
        self._nside = nside
        self._angpos = hputil.ang_positions(nside)

        # The horizon function
        self._horizon = visibility.horizon(self._angpos, self.zenith)

        # Beam
        self._beam = visibility.cylinder_beam(self._angpos, self.zenith, self.cylinder_width)

        # Multiplied quantity
        self._mul = self._horizon * self._beam


    def _transfer_single(self, uv, lmax, lside):

        if self._nside != self._best_nside(lmax):
            self._init_trans(self._best_nside(lmax))

        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        cvis = self._mul * fringe

        btrans = hputil.sphtrans_complex(cvis, centered = False, lmax = int(lmax), lside=lside)

        return btrans


    def _make_matrix_array(self, shape, mfirst, lmax):

        if mfirst:
            tarray = np.zeros((2*lmax+1,) + shape + (lmax+1,), dtype=np.complex128)
        else:
            tarray = np.zeros(shape + (lmax+1, 2*lmax+1), dtype=np.complex128)

        return tarray

    def _copy_single_into_array(self, tarray, tsingle, ind, mfirst):

        islice = (slice(None),) + ind + (slice(None),) if mfirst else ind
        tarray[islice] = tsingle.T if mfirst else tsingle


        
        

        
        
