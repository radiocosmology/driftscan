import abc

import numpy as np

from cosmoutils import hputil, units
import visibility



def in_range(arr, min, max):
	"""Check if array entries are within the given range.
	
	Parameters
	----------
	arr : np.ndarray
		Array to check.
	min, max : scalar or np.ndarray
		Minimum and maximum values to test against. Values can be in arrays
		broadcastable against `arr`.
	
	Returns
	-------
	val : boolean
		True if all entries are within range.
	"""
    return (arr >= min).all() and (arr < max).all()

def out_of_range(arr, min, max):
    return not in_range(arr, min, max)


def map_half_plane(arr):
    arr = np.where((arr[:,0] < 0.0)[:,np.newaxis], -arr, arr)
    arr = np.where(np.logical_and(arr[:,0] == 0.0, arr[:,1] < 1)[:,np.newaxis], -arr, arr)
    
    return arr


_horizon_const = 0
def max_lm(baselines, wavelengths, uwidth, vwidth = 0.0):
    """Get the maximum (l,m) that a baseline is sensitive to.

    Parameters
    ----------
    baselines : np.ndarray
        An array of baselines.
    wavelengths : np.ndarray
        An array of frequencies.
    width : np.ndarray
        Width of the receiver in the u-direction.

    Returns
    -------
    lmax, mmax : array_like
    """

    umax = (np.abs(baselines[:,0]) + uwidth) / wavelengths
    vmax = (np.abs(baselines[:,1]) + vwidth)  / wavelengths

    mmax = np.ceil(2 * np.pi * umax).astype(np.int64) + _horizon_const
    lmax = np.ceil((mmax**2 + (2*np.pi*vmax)**2)**0.5).astype(np.int64) + _horizon_const

    return lmax, mmax



class TransitTelescope(object):
    """Base class for simulating any transit interferometer.
    
    This is an abstract class, and several methods must be implemented before it
    is usable. These are:

    * `feedpositions` - a property which contains the positions of all the feeds
    * `_get_unique` -  calculates which baselines are identical
    * `_transfer_single` - calculate the beam transfer for a single baseline+freq
    * `_make_matrix_array` - makes an array of the right size to hold the
      transfer functions
    * `_copy_transfer_into_single` - copy a single transfer matrix into a
      collection.

    The last two are required for supporting polarised beam functions.
    
    Properties
    ----------
    freq_lower, freq_higher : scalar
        The center of the lowest and highest frequency bands.
    num_freq : scalar
        The number of frequency bands (only use for setting up the frequency
        binning). Generally using `nfreq` is preferred.
    tsys_flat : scalar
        The system temperature (in K). Override `tsys` for anything more
        sophisticated.
    """
    __metaclass__ = abc.ABCMeta  # Enforce Abstract class

    
    freq_lower = 400.0
    freq_upper = 800.0

    num_freq = 50

    _progress = lambda x: x

    accuracy_boost = 1


    def __init__(self, latitude=45, longitude=0):
        """Initialise a telescope object.
        
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

        #delkeys = ['_baselines', '_redundancy', '_frequencies'] + self._extdelkeys

        for key in self.__dict__:
            #if (key in delkeys) or (key[0] == "_"):
            if (key[0] == "_"):
                del state[key]

        return state
            
    

    #========= Properties related to baselines =========

    _baselines = None

    @property
    def baselines(self):
        """The unique baselines in the telescope."""
        if self._baselines == None:
            self.calculate_baselines()

        return self._baselines


    _redundancy = None

    @property
    def redundancy(self):
        """The redundancy of each baseline (corresponds to entries in
        cyl.baselines)."""
        if self._redundancy == None:
            self.calculate_baselines()

        return self._redundancy

    @property
    def nbase(self):
        """The number of unique baselines."""
        return self.baselines.shape[0]

    
    _feedpairs = None
    
    @property
    def feedpairs(self):
        """An (nbase,2) array of the feed pairs corresponding to each baseline."""
        if self._feedpairs == None:
            self.calculate_baselines()
        return self._feedpairs
    
    #===================================================



    #======== Properties related to frequencies ========
    
    _frequencies = None

    @property
    def frequencies(self):
        """The centre of each frequency band (in MHz)."""
        if self._frequencies == None:
            self.calculate_frequencies()

        return self._frequencies

    def calculate_frequencies(self):

        self._frequencies = np.linspace(self.freq_lower, self.freq_upper, self.num_freq)


    @property
    def wavelengths(self):
        """The central wavelength of each frequency band (in metres)."""
        return units.c / (1e6 * self.frequencies)

    @property
    def nfreq(self):
        """The number of frequency bins."""
        return self.frequencies.shape[0]

    #===================================================



    
    #======== Properties related to the feeds ==========

    @property
    def nfeed(self):
        """The number of feeds."""
        return self.feedpositions.shape[0]

    #===================================================


    #======= Properties related to polarisation ========
    
    @property
    def num_pol_telescope(self):
        """The number of polarisation combinations on the telescope. Should be
        one of 1 (unpolarised), 3 (XX, YY, XY=YX), and (XX, YY, XY, YX)."""
        return self._npol_tel_

    @property
    def num_pol_sky(self):
        """The number of polarisation combinations on the sky that we are
        considering. Should be either 1 (T=I only) or 3 (T, Q, U
        sensitivity). Circular polarisation (stokes V) is ignored."""
        return self._npol_sky_

    #===================================================




    #===== Properties related to harmonic spread =======

    @property
    def lmax(self):
        """The maximum l the telescope is sensitive to."""
        lmax, mmax = max_lm(self.baselines, self.wavelengths[-1], self.u_width, self.v_width)
        return lmax.max()

    @property
    def mmax(self):
        """The maximum m the telescope is sensitive to."""
        lmax, mmax = max_lm(self.baselines, self.wavelengths[-1], self.u_width, self.v_width)
        return mmax.max()

    #===================================================

    

    #== Methods for calculating the unique baselines ===
    
    def calculate_baselines(self):
        """Calculate all the unique baselines and their redundancies, and set
        the internal state of the object.
        """

        # Form list of all feed pairs
        fpairs = np.indices((self.nfeed, self.nfeed))[(slice(None),) + np.triu_indices(self.nfeed, 1)]

        # Get unique pairs
        upairs, self._redundancy = self._get_unique(fpairs)
        
        self._baselines = self.feedpositions[upairs[0]] - self.feedpositions[upairs[1]]  # Should this be the negative?

        self._feedpairs = upairs.T

    #===================================================




    #==== Methods for calculating Transfer matrices ====
        
    def transfer_matrices(self, bl_indices, f_indices, global_lmax = True):
        """Calculate the spherical harmonic transfer matrices for baseline and
        frequency combinations.

        Parameters
        ----------
        bl_indices : array_like
            Indices of baselines to calculate.
        f_indices : array_like
            Indices of frequencies to calculate. Must be broadcastable against
            `bl_indices`.
        global_lmax : boolean, optional
            If set (default), the output size `lside` in (l,m) is big enough to
            hold the maximum for the entire telescope. If not set it is only big
            enough for the requested set.

        Returns
        -------
        transfer : np.ndarray, dtype=np.complex128
            An array containing the transfer functions. The shape is somewhat
            complicated, the first indices correspond to the broadcast size of
            `bl_indices` and `f_indices`, then there may be some polarisation
            indices, then finally the (l,m) indices, range (lside, 2*lside-1).
        """

        # Broadcast arrays against each other
        bl_indices, f_indices = np.broadcast_arrays(bl_indices, f_indices)

        ## Check indices are all in range
        if out_of_range(bl_indices, 0, self.nbase):
            raise Exception("Baseline indices aren't valid")

        if out_of_range(f_indices, 0, self.nfreq):
            raise Exception("Frequency indices aren't valid")

        # Fetch the set of lmax's for the baselines (in order to reduce time
        # regenerating Healpix maps)
        lmax, mmax = max_lm(self.baselines[bl_indices], self.wavelengths[f_indices], self.u_width, self.v_width)

        # Set the size of the (l,m) array to write into
        lside = self.lmax if global_lmax else lmax.max()

        # Generate the array for the Transfer functions

        tarray = np.zeros(bl_indices.shape + (self.num_pol_telescope, self.num_pol_sky, lside+1, 2*lside+1),
                          dtype=np.complex128)

        print "Size: %i elements. Memory %f GB." % (tarray.size, 2*tarray.size * 8.0 / 2**30)

        # Sort the baselines by ascending lmax and iterate through in that
        # order, calculating the transfer matrices
        i_arr = np.argsort(lmax.flat)
        
        for iflat in np.argsort(lmax.flat):
            ind = np.unravel_index(iflat, lmax.shape)
            trans = self._transfer_single(bl_indices[ind], f_indices[ind], lmax[ind], lside)

            ## Iterate over pol combinations and copy into transfer array
            for pi in range(self.num_pol_telescope):
                for pj in range(self.num_pol_sky):
                    islice = (ind + (pi,pj) + (slice(None),slice(None)))
                    tarray[islice] = trans[pi][pj]

        return tarray


    def transfer_for_frequency(self, freq):
        """Fetch all transfer matrices for a given frequency.
        
        Parameters
        ----------
        freq : integer
            The frequency index.

        Returns
        -------
        transfer : np.ndarray
            The transfer matrices. Packed as in `TransitTelescope.transfer_matrices`.
        """
        bi = np.arange(self.nbase)
        fi = freq * np.ones_like(bi)

        return self.transfer_matrices(bi, fi)


    def transfer_for_baseline(self, baseline):
        """Fetch all transfer matrices for a given baseline.
        
        Parameters
        ----------
        baseline : integer
            The baseline index.

        Returns
        -------
        transfer : np.ndarray
            The transfer matrices. Packed as in `TransitTelescope.transfer_matrices`.
        """
        fi = np.arange(self.nfreq)
        bi = baseline * np.ones_like(fi)

        return self.transfer_matrices(bi, fi)

    #===================================================



    #======== Noise properties of the telescope ========

    tsys_flat = 50.0 # Kelvin
    ndays = 732.5 # 2 years in sidereal days
    
    def tsys(self, f_indices = None):
        """The system temperature.

        Currenty has a flat T_sys across the whole bandwidth. Override for
        anything more complicated.

        Parameters
        ----------
        f_indices : array_like
            Indices of frequencies to get T_sys at.

        Returns
        -------
        tsys : array_like
            System temperature at requested frequencies.
        """
        if f_indices == None:
            freq = self.frequencies
        else:
            freq = self.frequencies[f_indices]
        return np.ones_like(freq) * self.tsys_flat


    def noisepower(self, bl_indices, f_indices, ndays = None):
        """Calculate the instrumental noise power spectrum.

        Assume we are still within the regime where the power spectrum is white
        in `m` modes.
        
        Parameters
        ----------
        bl_indices : array_like
            Indices of baselines to calculate.
        f_indices : array_like
            Indices of frequencies to calculate. Must be broadcastable against
            `bl_indices`.
        ndays : integer
            The number of sidereal days observed.

        Returns
        -------
        noise_ps : np.ndarray
            The noise power spectrum.
        """

        ndays = self.ndays if not ndays else ndays # Set to value if not set.
        
        # Broadcast arrays against each other
        bl_indices, f_indices = np.broadcast_arrays(bl_indices, f_indices)

        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        delnu = units.t_sidereal * bw / (2*np.pi)
        noisepower = self.tsys(f_indices)**2 / (2 * np.pi * delnu * ndays)
        noisebase = noisepower / self.redundancy[bl_indices]

        return noisebase
        
    #===================================================


    _nside = None

    def _init_trans(self, nside):
        ## Internal function for generating some common Healpix maps (position,
        ## horizon). These should need to be generated only when nside changes.
        
        # Angular positions in healpix map of nside
        self._nside = nside
        self._angpos = hputil.ang_positions(nside)

        # The horizon function
        self._horizon = visibility.horizon(self._angpos, self.zenith)




    #===================================================
    #================ ABSTRACT METHODS =================
    #===================================================


    # Implement to specify feed positions in the telescope.
    @abc.abstractproperty
    def feedpositions(self):
        """An (nfeed,2) array of the feed positions relative to an arbitary point (in m)"""
        return
    
    # Implement to specify feed positions in the telescope.
    @abc.abstractproperty
    def u_width(self):
        """The approximate physical width (in the u-direction) of the dish/telescope etc, for
        calculating the maximum (l,m)."""
        return

    # Implement to specify feed positions in the telescope.
    @abc.abstractproperty
    def v_width(self):
        """The approximate physical length (in the v-direction) of the dish/telescope etc, for
        calculating the maximum (l,m)."""
        return


    # Implement to determine which baselines are unique
    @abc.abstractmethod
    def _get_unique(self, fpairs):
        """Calculate the unique baseline pairs.

        **Abstract method** must be implemented.
        
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
        return


    # The work method which does the bulk of calculating all the transfer matrices.
    @abc.abstractmethod
    def _transfer_single(self, bl_index, f_index, lmax, lside):
        """Calculate transfer matrix for a single baseline+frequency.
        
        **Abstract method** must be implemented.

        Parameters
        ----------
        bl_index : integer
            The index of the baseline to calculate.
        f_index : integer
            The index of the frequency to calculate.
        lmax : integer
            The maximum *l* we are interested in. Determines accuracy of
            spherical harmonic transforms.
        lside : integer
            The size of array to embed the transfer matrix within.

        Returns
        -------
        transfer : np.ndarray
            The transfer matrix, an array of shape (pol_indices, lside,
            2*lside-1). Where the `pol_indices` are usually only present if
            considering the polarised case.
        """
        return


    #===================================================
    #============== END ABSTRACT METHODS ===============
    #===================================================
    





class UnpolarisedTelescope(TransitTelescope):
    """A base for an unpolarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the `beam` function.
    """
    __metaclass__ = abc.ABCMeta

    _npol_tel_ = 1
    _npol_sky_ = 1
    
    @abc.abstractmethod
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
        return
    

    #===== Implementations of abstract functions =======

    def _transfer_single(self, bl_index, f_index, lmax, lside):
        
        if self._nside != hputil.nside_for_lmax(lmax, accuracy_boost=self.accuracy_boost):
            self._init_trans(hputil.nside_for_lmax(lmax, accuracy_boost=self.accuracy_boost))

        # Get beam maps for each feed.
        feedi, feedj = self.feedpairs[bl_index]
        beami, beamj = self.beam(feedi, f_index), self.beam(feedj, f_index)

        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        # Beam solid angle (integrate over beam^2 - equal area pixels)
        omega_A = (np.abs(beami) * np.abs(beamj) * self._horizon).sum() * (4*np.pi / beami.size)

        # Calculate the complex visibility
        cvis = self._horizon * fringe * beami * beamj / omega_A

        # Perform the harmonic transform to get the transfer matrix.
        btrans = hputil.sphtrans_complex(cvis, centered = False, lmax = lmax, lside=lside)

        return [ [ btrans ]]


    #===================================================

    def noisepower(self, bl_indices, f_indices, ndays = None):
        """Calculate the instrumental noise power spectrum.

        Assume we are still within the regime where the power spectrum is white
        in `m` modes. 
        
        Parameters
        ----------
        bl_indices : array_like
            Indices of baselines to calculate.
        f_indices : array_like
            Indices of frequencies to calculate. Must be broadcastable against
            `bl_indices`.
        ndays : integer
            The number of sidereal days observed.

        Returns
        -------
        noise_ps : np.ndarray
            The noise power spectrum.
        """

        bnoise = TransitTelescope.noisepower(self, bl_indices, f_indices, ndays)

        return bnoise[..., np.newaxis] * 0.5 # Correction for unpolarisedness




class PolarisedTelescope(TransitTelescope):
    """A base for a polarised telescope.
    
    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.
    
    Properties
    ----------
    feedx, feedy : np.ndarray
        Two element vectors giving the orientation of the x, y polarisations in
        the UV plane. The directions are assumed to be the same over all feeds.
    """
    __metaclass__ = abc.ABCMeta
    
    feedx = np.array([1.0, 0.0])
    feedy = np.array([0.0, 1.0])

    _npol_tel_ = 3
    _npol_sky_ = 3

    
    def _init_trans(self, nside):
        ## Override _init_trans to generate the polarisation projections.
        
        TransitTelescope._init_trans(self, nside)

        # Polarisation projections of feed pairs
        self._pIQUxx = visibility.pol_IQU(self._angpos, self.zenith, self.feedx, self.feedx)
        self._pIQUxy = visibility.pol_IQU(self._angpos, self.zenith, self.feedx, self.feedy)
        self._pIQUyy = visibility.pol_IQU(self._angpos, self.zenith, self.feedy, self.feedy)

        # Multiplied pairs
        self._mIQUxx = self._horizon * self._pIQUxx
        self._mIQUxy = self._horizon * self._pIQUxy
        self._mIQUyy = self._horizon * self._pIQUyy


    #===== Implementations of abstract functions =======

    def _transfer_single(self, bl_index, f_index, lmax, lside):

        if self._nside != hputil.nside_for_lmax(lmax):
            self._init_trans(hputil.nside_for_lmax(lmax))

        # Get beam maps for each feed.
        feedi, feedj = self.feedpairs[bl_index]
        beamix, beamjx = self.beamx(feedi, f_index), self.beamx(feedj, f_index)
        beamiy, beamjy = self.beamy(feedi, f_index), self.beamy(feedj, f_index)

        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        omega_A_xx = (np.abs(beamix) * np.abs(beamjx) * self._horizon).sum() * (4*np.pi / beamix.size)
        omega_A_xy = (np.abs(beamix) * np.abs(beamjy) * self._horizon).sum() * (4*np.pi / beamix.size)
        omega_A_yy = np.abs((beamiy) * np.abs(beamjy) * self._horizon).sum() * (4*np.pi / beamiy.size)

        cvIQUxx = self._mIQUxx * fringe * beamix * beamjx / omega_A_xx
        cvIQUxy = self._mIQUxy * fringe * beamix * beamjy / omega_A_xy
        cvIQUyy = self._mIQUyy * fringe * beamiy * beamjy / omega_A_yy

        ### If beams ever become complex need to do yx combination.
        btransxx = hputil.sphtrans_complex_pol(cvIQUxx, centered = False,
                                               lmax = int(lmax), lside=lside)
        btransxy = hputil.sphtrans_complex_pol(cvIQUxy, centered = False,
                                               lmax = int(lmax), lside=lside)
        btransyy = hputil.sphtrans_complex_pol(cvIQUyy, centered = False,
                                               lmax = int(lmax), lside=lside)

        return [btransxx, btransxy, btransyy]


    #===================================================

    def noisepower(self, bl_indices, f_indices, ndays=None):
        """Calculate the instrumental noise power spectrum.

        Assume we are still within the regime where the power spectrum is white
        in `m` modes. 
        
        Parameters
        ----------
        bl_indices : array_like
            Indices of baselines to calculate.
        f_indices : array_like
            Indices of frequencies to calculate. Must be broadcastable against
            `bl_indices`.
        ndays : integer
            The number of sidereal days observed.

        Returns
        -------
        noise_ps : np.ndarray
            The noise power spectrum.
        """

        bnoise = TransitTelescope.noisepower(self, bl_indices, f_indices, ndays)

        return bnoise[..., np.newaxis] * np.array([1.0, 0.5, 1.0])
        
    #===================================================


