import abc

import numpy as np

from cosmoutils import hputil, units

from drift.core import visibility
from drift.util import util, config



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
    arr = np.where(np.logical_and(arr[:,0] == 0.0, arr[:,1] < 0.0)[:,np.newaxis], -arr, arr)
    
    return arr




def _merge_keyarray(keys1, keys2, mask1=None, mask2=None):

    mask1 = mask1 if mask1 is not None else np.ones_like(keys1, dtype=np.bool)
    mask2 = mask2 if mask2 is not None else np.ones_like(keys2, dtype=np.bool)

    # Merge two groups of feed arrays
    cmask = np.logical_and(mask1, mask2)
    ckeys = _remap_keyarray(keys1 + 1.0J * keys2, mask=cmask)

    if mask1 is None and mask2 is None:
        return ckeys
    else:
        return ckeys, cmask


def _remap_keyarray(keyarray, mask=None):
    # Look through an array of keys and attach integer labels to each
    # equivalent classes of keys (also take into account masking).
    if mask == None:
        mask = np.ones(keyarray.shape, np.bool)

    ind = np.where(mask)

    un, inv = np.unique(keyarray[ind], return_inverse=True)

    fmap = -1*np.ones(keyarray.shape, dtype=np.int)

    fmap[ind] = np.arange(un.size)[inv]
    return fmap


def _get_indices(keyarray, mask=None):
    # Return a pair of indices for each group of equivalent feed pairs
    if mask == None:
        mask = np.ones(keyarray.shape, np.bool)

    wm = np.where(mask.ravel())[0]
    keysflat = keyarray.ravel()[wm]

    un, ind = np.unique(keysflat, return_index=True)
    # CHANGE: np (< 1.6) does not support multiple indices in np.unravel_index
    #upairs = np.array(np.unravel_index(wm[ind], keyarray.shape)).T
    upairs = np.array([np.unravel_index(i1, keyarray.shape) for i1 in wm[ind] ])

    return np.sort(upairs, axis=-1) # Sort to ensure we are in upper triangle



#_horizon_const = 0
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

    mmax = np.ceil(2 * np.pi * umax).astype(np.int64)# + _horizon_const
    lmax = np.ceil((mmax**2 + (2*np.pi*vmax)**2)**0.5).astype(np.int64)# + _horizon_const

    return lmax, mmax

def latlon_to_sphpol(latlon):

    zenith = np.array([np.pi / 2.0 - np.radians(latlon[0]),
                       np.remainder(np.radians(latlon[1]), 2*np.pi)])

    return zenith



class TransitTelescope(config.Reader):
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
    positive_m_only: boolean
        Whether to only deal with half the `m` range. In many cases we are
        much less sensitive to negative-m (depending on the hemisphere, and
        baseline alignment). This does not affect the beams calculated, only
        how they're used in further calculation. Default: False
    minlength, maxlength : scalar
        Minimum and maximum baseline lengths to include (in metres).

    """
    __metaclass__ = abc.ABCMeta  # Enforce Abstract class

    zenith = config.Property(proptype=latlon_to_sphpol, default=[45.0, 0.0])

    freq_lower = config.Property(proptype=float, default=400.0)
    freq_upper = config.Property(proptype=float, default=800.0)
    num_freq = config.Property(proptype=int, default=50)

    tsys_flat = config.Property(proptype=float, default=50.0, key='tsys')
    ndays = config.Property(proptype=int, default=733)

    accuracy_boost = config.Property(proptype=float, default=1.0)
    l_boost = config.Property(proptype=float, default=1.0)

    minlength = config.Property(proptype=float, default=0.0)
    maxlength = config.Property(proptype=float, default=1.0e7)

    auto_correlations = config.Property(proptype=bool, default=False)


    def __init__(self, latitude=45, longitude=0):
        """Initialise a telescope object.
        
        Parameters
        ----------
        latitude, longitude : scalar
            Position on the Earths surface of the telescope (in degrees).
        """

        self.zenith = [latitude, longitude]



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
            self.calculate_feedpairs()

        return self._baselines


    _redundancy = None

    @property
    def redundancy(self):
        """The redundancy of each baseline (corresponds to entries in
        cyl.baselines)."""
        if self._redundancy == None:
            self.calculate_feedpairs()

        return self._redundancy

    @property
    def nbase(self):
        """The number of unique baselines."""
        return self.npairs


    @property
    def npairs(self):
        """The number of unique feed pairs."""
        return self.uniquepairs.shape[0]

    
    _uniquepairs = None
    
    @property
    def uniquepairs(self):
        """An (npairs, 2) array of the feed pairs corresponding to each baseline."""
        if self._uniquepairs == None:
            self.calculate_feedpairs()
        return self._uniquepairs


    _feedmap = None

    @property
    def feedmap(self):
        """An (nfeed, nfeed) array giving the mapping between feedpairs and
        the calculated baselines. Each entry is an index into the arrays of unique pairs."""

        if self._feedmap == None:
            self.calculate_feedpairs()

        return self._feedmap


    _feedmask = None

    @property
    def feedmask(self):
        """An (nfeed, nfeed) array giving the entries that have been
        calculated. This allows to mask out pairs we want to ignore."""

        if self._feedmask == None:
            self.calculate_feedpairs()

        return self._feedmask

    _feedmap = None

    @property
    def feedconj(self):
        """An (nfeed, nfeed) array giving the feed pairs which must be complex
        conjugated."""

        if self._feedconj == None:
            self.calculate_feedpairs()

        return self._feedconj
    
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

        #self._frequencies = np.linspace(self.freq_lower, self.freq_upper, self.num_freq)
        self._frequencies = self.freq_lower + (np.arange(self.num_freq) + 0.5) * ((self.freq_upper - self.freq_lower) / self.num_freq)

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
        return int(np.ceil(lmax.max() * self.l_boost))

    @property
    def mmax(self):
        """The maximum m the telescope is sensitive to."""
        lmax, mmax = max_lm(self.baselines, self.wavelengths[-1], self.u_width, self.v_width)
        return int(np.ceil(mmax.max() * self.l_boost))

    #===================================================

    

    #== Methods for calculating the unique baselines ===
    
    def calculate_feedpairs(self):
        """Calculate all the unique feedpairs and their redundancies, and set
        the internal state of the object.
        """

        # Get unique pairs, and create mapping arrays
        self._feedmap, self._feedmask = self._get_unique()
        self._feedconj = np.tril(np.ones_like(self._feedmap), -1).astype(np.bool)
        self._uniquepairs = _get_indices(self._feedmap, mask=self._feedmask)
        self._redundancy = np.bincount(self._feedmap[np.where(self._feedmask * np.tri(self.nfeed))]) # Triangle mask to avoid double counting
        
        # Reorder and conjugate baselines such that the default feedpair
        # points W->E (to ensure we use positive-m)
        self._make_ew()
        self._baselines = self.feedpositions[self._uniquepairs[:, 0]] - self.feedpositions[self._uniquepairs[:, 1]]



    def _make_ew(self):
        # Reorder baselines pairs, such that the baseline vector always points E (or pure N)
        for i in range(self.npairs):
            sep = self.feedpositions[self._uniquepairs[i, 0]] - self.feedpositions[self._uniquepairs[i, 1]]

            if sep[0] < 0.0 or (sep[0] == 0.0 and sep[1] < 0.0):
                # Reorder feed pairs and conjugate mapping
                self._uniquepairs[i, 1], self._uniquepairs[i, 0] = self._uniquepairs[i, 0], self._uniquepairs[i, 1]
                self._feedconj = np.where(self._feedmap == i, np.logical_not(self._feedconj), self._feedconj)



    def _unique_baselines(self):
        """Map of equivalent baseline lengths, and mask of ones to exclude.
        """
        # Construct array of indices
        fshape = [self.nfeed, self.nfeed]
        f_ind = np.indices(fshape)

        # Construct array of baseline separations in complex representation
        bl1 = (self.feedpositions[f_ind[0]] - self.feedpositions[f_ind[1]])
        bl2 = map_half_plane(bl1.reshape(-1, 2)).reshape(bl1.shape)
        bl3 = np.around(bl2[..., 0] + 1.0J * bl2[..., 1], 7)

        # Construct array of baseline lengths
        blen = np.sum(bl1**2, -1)**0.5

        # Create mask of included baselines
        mask = np.logical_and(blen >= self.minlength, blen <= self.maxlength)

        return _remap_keyarray(bl3), mask



    def _unique_beams(self):
        """Map of unique beam pairs, and mask of ones to exclude.
        """
        # Construct array of indices
        fshape = [self.nfeed, self.nfeed]

        beam_map = np.zeros(fshape, dtype=np.int)

        if self.auto_correlations:
            beam_mask = np.ones(fshape, dtype=np.bool)
        else:
            beam_mask = np.logical_not(np.identity(self.nfeed, dtype=np.bool))

        return beam_map, beam_mask


    def _get_unique(self):
        """Calculate the unique baseline pairs.
        
        All feeds are assumed to be identical. Baselines are identified if
        they have the same length, and are selected such that they point East
        (to ensure that the sensitivity ends up in positive-m modes).

        It is also possible to select only baselines within a particular
        length range by setting the `minlength` and `maxlength` properties.

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

        # Fetch and merge map of unique feed pairs        
        base_map, base_mask = self._unique_baselines()
        beam_map, beam_mask = self._unique_beams()
        comb_map, comb_mask = _merge_keyarray(base_map, beam_map, mask1=base_mask, mask2=beam_mask)
        
        return comb_map, comb_mask
 

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
        if out_of_range(bl_indices, 0, self.npairs):
            raise Exception("Baseline indices aren't valid")

        if out_of_range(f_indices, 0, self.nfreq):
            raise Exception("Frequency indices aren't valid")

        # Fetch the set of lmax's for the baselines (in order to reduce time
        # regenerating Healpix maps)
        lmax, mmax = np.ceil(self.l_boost * np.array(max_lm(self.baselines[bl_indices], self.wavelengths[f_indices], self.u_width, self.v_width))).astype(np.int64)
        #lmax, mmax = lmax * self.l_boost, mmax * self.l_boost
        # Set the size of the (l,m) array to write into
        lside = self.lmax if global_lmax else lmax.max()

        # Generate the array for the Transfer functions

        tshape = bl_indices.shape + (self.num_pol_telescope, self.num_pol_sky, lside+1, 2*lside+1)
        print "Size: %i elements. Memory %f GB." % (np.prod(tshape), 2*np.prod(tshape) * 8.0 / 2**30)
        tarray = np.zeros(tshape, dtype=np.complex128)

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
        bi = np.arange(self.npairs)
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


    def noisepower_feedpairs(self, fi, fj, f_indices, m, ndays=None):
        ndays = self.ndays if not ndays else ndays

        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        delnu = units.t_sidereal * bw / (2*np.pi)
        noisepower = self.tsys(f_indices)**2 / (2 * np.pi * delnu * ndays)

        return np.ones_like(fi) * np.ones_like(fj) * np.ones_like(m) * noisepower / 2.0 # For unpolarised only at the moment.

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

    def _beam_map_single(self, bl_index, f_index):
        
        # Get beam maps for each feed.
        feedi, feedj = self.uniquepairs[bl_index]
        beami, beamj = self.beam(feedi, f_index), self.beam(feedj, f_index)

        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        # Beam solid angle (integrate over beam^2 - equal area pixels)
        omega_A = (np.abs(beami) * np.abs(beamj) * self._horizon).sum() * (4*np.pi / beami.size)

        # Calculate the complex visibility
        cvis = self._horizon * fringe * beami * beamj / omega_A

        return cvis


    def _transfer_single(self, bl_index, f_index, lmax, lside):
        
        if self._nside != hputil.nside_for_lmax(lmax, accuracy_boost=self.accuracy_boost):
            self._init_trans(hputil.nside_for_lmax(lmax, accuracy_boost=self.accuracy_boost))

        cvis = self._beam_map_single(bl_index, f_index)

        # Perform the harmonic transform to get the transfer matrix (conj is correct - see paper)
        btrans = hputil.sphtrans_complex(cvis.conj(), centered = False, lmax = lmax, lside=lside).conj()

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
    
    Abstract Methods
    ----------------
    beamx, beamy : methods
        Routines giving the field pattern for the x and y feeds.
    """
    __metaclass__ = abc.ABCMeta
    
    _npol_tel_ = 3
    _npol_sky_ = 3


    @abc.abstractmethod
    def beamx(self, feed, freq):
        """Beam for the x polarisation feed.
        
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

    @abc.abstractmethod
    def beamy(self, feed, freq):
        """Beam for the x polarisation feed.
        
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
    


    def _beam_map_single(self, bl_index, f_index):

        pIQU = [0.5 * np.array([[1.0, 0.0], [0.0, 1.0]]),
                0.5 * np.array([[1.0, 0.0], [0.0, -1.0]]),
                0.5 * np.array([[0.0, 1.0], [1.0, 0.0]]) ]

        # Get beam maps for each feed.
        feedi, feedj = self.uniquepairs[bl_index]
        beamix, beamiy = self.beamx(feedi, f_index), self.beamy(feedi, f_index)
        beamjx, beamjy = self.beamx(feedj, f_index), self.beamy(feedj, f_index)
        
        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        powIQU_xx = [ np.sum(beamix * np.dot(beamjx, polproj), axis=1) * self._horizon for polproj in pIQU]
        powIQU_xy = [ np.sum(beamix * np.dot(beamjy, polproj), axis=1) * self._horizon for polproj in pIQU]
        powIQU_yy = [ np.sum(beamiy * np.dot(beamjy, polproj), axis=1) * self._horizon for polproj in pIQU]
        
        pxarea = (4*np.pi / beamix.shape[0])

        om_ix = np.sum(np.abs(beamix)**2 * self._horizon[:, np.newaxis]) * pxarea
        om_iy = np.sum(np.abs(beamiy)**2 * self._horizon[:, np.newaxis]) * pxarea
        om_jx = np.sum(np.abs(beamjx)**2 * self._horizon[:, np.newaxis]) * pxarea
        om_jy = np.sum(np.abs(beamjy)**2 * self._horizon[:, np.newaxis]) * pxarea

        omega_A_xx = (om_ix * om_jx)**0.5
        omega_A_xy = (om_ix * om_jy)**0.5
        omega_A_yy = (om_iy * om_jy)**0.5
        
        cvIQUxx = [ p * (2 * fringe / omega_A_xx) for p in powIQU_xx ]
        cvIQUxy = [ p * (2 * fringe / omega_A_xy) for p in powIQU_xy ]
        cvIQUyy = [ p * (2 * fringe / omega_A_yy) for p in powIQU_yy ]

        return cvIQUxx, cvIQUxy, cvIQUyy


    #===== Implementations of abstract functions =======

    def _transfer_single(self, bl_index, f_index, lmax, lside):

        if self._nside != hputil.nside_for_lmax(lmax):
            self._init_trans(hputil.nside_for_lmax(lmax))

        bmaps = self._beam_map_single(bl_index, f_index)

        btrans = [ [ pb.conj() for pb in hputil.sphtrans_complex_pol([bm.conj() for bm in bmap], centered = False,
                                                                     lmax = int(lmax), lside=lside) ] for bmap in bmaps]

        return btrans


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


