import abc
import logging
from functools import cached_property
import numpy as np

from caput import cache
from caput import config
from caput import time as ctime

from cora.util import hputil, units

from . import visibility
from ..util._fast_tools import _construct_pol_real, _construct_pol_complex


# Create logger object
logger = logging.getLogger(__name__)


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
    arr = np.where((arr[:, 0] < 0.0)[:, np.newaxis], -arr, arr)
    arr = np.where(
        np.logical_and(arr[:, 0] == 0.0, arr[:, 1] < 0.0)[:, np.newaxis], -arr, arr
    )

    return arr


def _merge_keyarray(keys1, keys2, mask1=None, mask2=None):
    tmask1 = mask1 if mask1 is not None else np.ones_like(keys1, dtype=bool)
    tmask2 = mask2 if mask2 is not None else np.ones_like(keys2, dtype=bool)

    # Merge two groups of feed arrays
    cmask = np.logical_and(tmask1, tmask2)
    ckeys = _remap_keyarray(keys1 + 1.0j * keys2, mask=cmask)

    if mask1 is None and mask2 is None:
        return ckeys
    else:
        return ckeys, cmask


def _remap_keyarray(keyarray, mask=None):
    # Look through an array of keys and attach integer labels to each
    # equivalent classes of keys (also take into account masking).
    if mask is None:
        mask = np.ones(keyarray.shape, bool)

    ind = np.where(mask)

    un, inv = np.unique(keyarray[ind], return_inverse=True)

    fmap = -1 * np.ones(keyarray.shape, dtype=np.int64)

    fmap[ind] = np.arange(un.size)[inv]
    return fmap


def _get_indices(keyarray, mask=None):
    # Return a pair of indices for each group of equivalent feed pairs
    if mask is None:
        mask = np.ones(keyarray.shape, bool)

    wm = np.where(mask.ravel())[0]
    keysflat = keyarray.ravel()[wm]

    un, ind = np.unique(keysflat, return_index=True)
    # CHANGE: np (< 1.6) does not support multiple indices in np.unravel_index
    # upairs = np.array(np.unravel_index(wm[ind], keyarray.shape)).T
    upairs = np.array([np.unravel_index(i1, keyarray.shape) for i1 in wm[ind]])

    # return np.sort(upairs, axis=-1) # Sort to ensure we are in upper triangle
    return upairs


def max_lm(baselines, wavelengths, uwidth, vwidth=0.0):
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

    umax = (np.abs(baselines[:, 0]) + uwidth) / wavelengths
    vmax = (np.abs(baselines[:, 1]) + vwidth) / wavelengths

    mmax = np.ceil(2 * np.pi * umax).astype(np.int64)
    lmax = np.ceil((mmax**2 + (2 * np.pi * vmax) ** 2) ** 0.5).astype(np.int64)

    return lmax, mmax


class TransitTelescope(config.Reader, ctime.Observer, metaclass=abc.ABCMeta):
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

    Attributes
    ----------
    freq_lower, freq_higher : scalar
        The center of the lowest and highest frequency bands. Deprecated, use
        `freq_start`, `freq_end` instead.
    freq_start, freq_end : scalar
        The start and end frequencies in MHz. Defaults: 800, 400.
    num_freq : scalar
        The number of frequency bands (only use for setting up the frequency
        binning). Generally using `nfreq` is preferred. Default: 1024.
    freq_mode : {"centre", "edge"}
        Choose if `freq_start` and `freq_end` are the edges of the band
        ("edge"), or whether they are the central frequencies of the first
        and last channel, in this case the last (Nyquist) frequency can
        either be skipped ("centre", default) or included ("centre_nyquist").
        The behaviour of the "centre" mode matches the output of the CASPER
        PFB-FIR block.
    channel_bin : int, optional
        Number of channels to bin together. This must exactly devide the total number.
        Binning is performed prior to selection of any subset. Default: 1.
    channel_list : list, optional
        List of channel indices to select. If set, this takes priority over
        `channel_range`. Currently this is not implemented.
    channel_range : list, optional
        Select subset of frequencies using a range of frequency channel indices,
        either [start, stop, step], [start, stop], or [stop] is acceptable.
        Default selects all channels.
    tsys_flat : scalar
        The system temperature (in K). Override `tsys` for anything more
        sophisticated. Default: 50.
    ndays : int
        Number of days to assume when computing thermal noise. Default: 733.
    accuracy_boost : float
        When computing beam transfer function, increase nside of healpix maps by
        2**accuracy_boost compared to default determination of nside. Default: 1.0.
    l_boost: float
        Increase lmax and mmax for telescope, and lmax/mmax values computed for
        individual baselines, by a factor of l_boost compared to default computations.
        Default: 1.0.
    force_lmax, force_mmax : int
        Use specific values for the telescope's l_max and m_max, instead of computing
        these values based on the angular scales accessible to the longest baseline
        at the highest frequency. l_boost is ignored if these values are specified.
        This is useful if you intend to combine several sets of beam transfer matrices
        that are separately computed over different frequency ranges. Default: None.
    minlength, maxlength : scalar
        Minimum and maximum baseline lengths to include (in metres).
    auto_correlations : bool
        Include elements for feed auto-correlations in computed beam transfer matrices.
        Default: False.
    local_origin : bool
        If set the observers location is the terrestrial origin, and so the
        rotation angle corresponds to the right ascension that is overhead
        (Local Stellar Angle in `caput.time`). If not the origin is Greenwich,
        so the rotation angle is what is overhead at Greenwich (Earth Rotation
        Angle). Default: True.
    skip_freq : list
        Frequency indices (with the set of frequencies defined by the other parameters)
        to skip. Skipped frequencies are considered to be present, *but* their beam
        transfer matrices are implicitly zero and thus are skipped in the beam transfer
        matrix calculation. This is useful for RFI channels that we know will be masked.
    skip_baselines : list
        Baseline indices to skip. Like skipped frequencies, skipped baselines are
        considered to be present, *but* their beam transfer matrices are implicitly zero
        and thus are skipped in the beam transfer matrix calculation.
    beam_cache_size : float
        Size of the beam cache in MB. Setting this minimises the amount of recalculation
        of the primary beams while generating beam transfer matrices. Default is 200 MB.
    """

    freq_lower = config.Property(proptype=float, default=None)
    freq_upper = config.Property(proptype=float, default=None)

    freq_start = config.Property(proptype=float, default=800.0)
    freq_end = config.Property(proptype=float, default=400.0)
    num_freq = config.Property(proptype=int, default=1024)

    freq_mode = config.enum(["centre", "centre_nyquist", "edge"], default="centre")

    channel_bin = config.Property(proptype=int, default=1)
    channel_range = config.Property(proptype=list)
    channel_list = config.Property(proptype=list)

    tsys_flat = config.Property(proptype=float, default=50.0, key="tsys")
    ndays = config.Property(proptype=int, default=733)

    accuracy_boost = config.Property(proptype=float, default=1.0)
    l_boost = config.Property(proptype=float, default=1.0)
    force_lmax = config.Property(proptype=int, default=None)
    force_mmax = config.Property(proptype=int, default=None)

    minlength = config.Property(proptype=float, default=0.0)
    maxlength = config.Property(proptype=float, default=1.0e7)

    auto_correlations = config.Property(proptype=bool, default=False)

    local_origin = config.Property(proptype=bool, default=True)

    # Skipping frequency/baseline parameters
    skip_freq = config.list_type(type_=int, default=[])
    skip_baselines = config.list_type(type_=int, default=[])

    beam_cache_size = config.Property(proptype=int, default=200)

    def __init__(self, latitude=45, longitude=0, **kwargs):
        """Initialise a telescope object.

        Parameters
        ----------
        latitude, longitude : scalar
            Position on the Earths surface of the telescope (in degrees).
        """

        # Set the observers position on the Earth
        ctime.Observer.__init__(self, longitude, latitude, **kwargs)

    _pickle_keys = []

    def __getstate__(self):
        state = self.__dict__.copy()

        for key in self.__dict__:
            if (key not in self._pickle_keys) and (key[0] == "_"):
                del state[key]

        return state

    @property
    def zenith(self):
        """
        The zenith vector in spherical polars.

        The position of the zenith spherical polars (in radians). Read only.

        Returns
        -------
            zenith : [theta, phi]
        """

        # Set polar angle
        theta = np.pi / 2.0 - np.radians(self.latitude)

        # Set azimuthal angle
        phi = np.remainder(np.radians(self.longitude), 2 * np.pi)

        # If we want a local origin, the observers location is the terrestrial
        # origin, so the zenith should be at phi=0. Otherwise the origin is
        # Greenwich, so we need the longitude.
        phi = 0.0 if self.local_origin else phi

        return np.array([theta, phi])

    # ========= Properties related to baselines =========

    _baselines = None

    @property
    def baselines(self):
        """The unique baselines in the telescope."""
        if self._baselines is None:
            self.calculate_feedpairs()

        return self._baselines

    _redundancy = None

    @property
    def redundancy(self):
        """The redundancy of each baseline (corresponds to entries in
        cyl.baselines)."""
        if self._redundancy is None:
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
        if self._uniquepairs is None:
            self.calculate_feedpairs()
        return self._uniquepairs

    _feedmap = None

    @property
    def feedmap(self):
        """An (nfeed, nfeed) array giving the mapping between feedpairs and
        the calculated baselines. Each entry is an index into the arrays of unique pairs.
        """

        if self._feedmap is None:
            self.calculate_feedpairs()

        return self._feedmap

    _feedmask = None

    @property
    def feedmask(self):
        """An (nfeed, nfeed) array giving the entries that have been
        calculated. This allows to mask out pairs we want to ignore."""

        if self._feedmask is None:
            self.calculate_feedpairs()

        return self._feedmask

    _feedconj = None

    @property
    def feedconj(self):
        """An (nfeed, nfeed) array giving the feed pairs which must be complex
        conjugated."""

        if self._feedconj is None:
            self.calculate_feedpairs()

        return self._feedconj

    # ===================================================

    # ======== Properties related to frequencies ========

    _frequencies = None

    @property
    def frequencies(self):
        """The centre of each frequency band (in MHz)."""
        if self._frequencies is None:
            self.calculate_frequencies()

        return self._frequencies

    def calculate_frequencies(self):
        if self.freq_lower or self.freq_upper:
            import warnings

            warnings.warn(
                "`freq_lower` and `freq_upper` parameters are deprecated",
                DeprecationWarning,
            )
            self.freq_start = self.freq_lower
            self.freq_end = self.freq_upper

        if self.freq_mode == "centre":
            df = abs(self.freq_end - self.freq_start) / self.num_freq
            frequencies = np.linspace(
                self.freq_start, self.freq_end, self.num_freq, endpoint=False
            )
        elif self.freq_mode == "centre_nyquist":
            df = abs(self.freq_end - self.freq_start) / (self.num_freq - 1)
            frequencies = np.linspace(
                self.freq_start, self.freq_end, self.num_freq, endpoint=True
            )
        else:
            df = abs(self.freq_end - self.freq_start) / self.num_freq
            frequencies = self.freq_start + df * (np.arange(self.num_freq) + 0.5)

        # Rebin frequencies if needed
        if self.channel_bin > 1:
            if self.num_freq % self.channel_bin != 0:
                raise ValueError(
                    "Channel binning must exactly divide the total number of channels"
                )

            frequencies = frequencies.reshape(-1, self.channel_bin).mean(axis=1)
            df = df * self.channel_bin

        # Select a subset of channels if required
        if self.channel_list is not None:
            raise NotImplementedError(
                "`channel_list` is not yet supported, as sparse channel selections "
                "may break things downstream."
            )
        if self.channel_range is not None:
            frequencies = frequencies[self.channel_range[0] : self.channel_range[1]]

        # TODO: do something with the channel width `df` as well
        self._frequencies = frequencies

    @property
    def wavelengths(self):
        """The central wavelength of each frequency band (in metres)."""
        return units.c / (1e6 * self.frequencies)

    @property
    def nfreq(self):
        """The number of frequency bins."""
        return self.frequencies.shape[0]

    # ===================================================

    # ======== Properties related to the feeds ==========

    @property
    def input_index(self):
        """Override to add custom labelling of the inputs, e.g. serial numbers.

        This should give an identifier that uniquely labels a correlator input and so
        can be used to match inputs through subsetting and reordering.

        There are two conventional fields used in the output, either a `chan_id`
        field for an integer label, or a `correlator_input` for a string labelling
        (useful for serial number strings). If both are present, `correlator_input`
        is used.
        """
        return np.array(np.arange(self.nfeed), dtype=[("chan_id", "u2")])

    @property
    def nfeed(self):
        """The number of feeds."""
        return self.feedpositions.shape[0]

    # ===================================================

    # ======= Properties related to polarisation ========

    @property
    def num_pol_sky(self):
        """The number of polarisation combinations on the sky that we are
        considering. Should be either 1 (T=I only), 3 (T, Q, U) or 4 (T, Q, U and V).
        """
        return self._npol_sky_

    # ===================================================

    # ===== Properties related to harmonic spread =======

    @property
    def lmax(self):
        """The maximum l the telescope is sensitive to."""
        if self.force_lmax is not None:
            return self.force_lmax
        else:
            lmax, mmax = max_lm(
                self.baselines, self.wavelengths.min(), self.u_width, self.v_width
            )
            return int(np.ceil(lmax.max() * self.l_boost))

    @property
    def mmax(self):
        """The maximum m the telescope is sensitive to."""
        if self.force_mmax is not None:
            return self.force_mmax
        else:
            lmax, mmax = max_lm(
                self.baselines, self.wavelengths.min(), self.u_width, self.v_width
            )
            return int(np.ceil(mmax.max() * self.l_boost))

    # ===================================================

    # == Methods for calculating the unique baselines ===

    def calculate_feedpairs(self):
        """Calculate all the unique feedpairs and their redundancies, and set
        the internal state of the object.
        """

        # Get unique pairs, and create mapping arrays
        self._feedmap, self._feedmask, self._feedconj = self._get_unique()

        # Reorder and conjugate baselines such that the default feedpair
        # points W->E (to ensure we use positive-m)
        self._make_ew()

        # Sort baselines into order
        self._sort_pairs()

        # Create mask of included pairs, that are not conjugated
        tmask = np.logical_and(self._feedmask, np.logical_not(self._feedconj))

        self._uniquepairs = _get_indices(self._feedmap, mask=tmask)
        self._redundancy = np.bincount(
            self._feedmap[np.where(tmask)]
        )  # Triangle mask to avoid double counting
        self._baselines = (
            self.feedpositions[self._uniquepairs[:, 0]]
            - self.feedpositions[self._uniquepairs[:, 1]]
        )

    def _make_ew(self):
        # Reorder baselines pairs, such that the baseline vector always points E (or
        # pure N)

        tmask = np.logical_and(self._feedmask, np.logical_not(self._feedconj))
        uniq = _get_indices(self._feedmap, mask=tmask)

        conj_map = np.zeros(uniq.shape[0] + 1, dtype=bool)

        for i in range(uniq.shape[0]):
            sep = self.feedpositions[uniq[i, 0]] - self.feedpositions[uniq[i, 1]]

            if sep[0] < 0.0 or (sep[0] == 0.0 and sep[1] < 0.0):
                # Note down that we need to flip feedconj
                conj_map[i] = True

        # Flip the feedpairs
        self._feedconj = np.logical_xor(self._feedconj, conj_map[self._feedmap])

    # Tolerance used when comparing baselines. See np.around documentation for details.
    _bl_tol = 6

    def _unique_baselines(self):
        """Map of equivalent baseline lengths, and mask of ones to exclude."""
        # Construct array of indices
        fshape = [self.nfeed, self.nfeed]
        f_ind = np.indices(fshape)

        # Construct array of baseline separations in complex representation
        bl1 = self.feedpositions[f_ind[0]] - self.feedpositions[f_ind[1]]
        bl2 = np.around(bl1[..., 0] + 1.0j * bl1[..., 1], self._bl_tol)

        # Construct array of baseline lengths
        blen = np.sum(bl1**2, axis=-1) ** 0.5

        # Create mask of included baselines
        mask = np.logical_and(blen >= self.minlength, blen <= self.maxlength)

        # Remove the auto correlated baselines between all polarisations
        if not self.auto_correlations:
            mask = np.logical_and(blen > 0.0, mask)

        return _remap_keyarray(bl2, mask), mask

    def _unique_beams(self):
        """Map of unique beam pairs, and mask of ones to exclude."""
        # Construct array of indices
        fshape = [self.nfeed, self.nfeed]

        bci, bcj = np.broadcast_arrays(
            self.beamclass[:, np.newaxis], self.beamclass[np.newaxis, :]
        )

        beam_map = _merge_keyarray(bci, bcj)

        if self.auto_correlations:
            beam_mask = np.ones(fshape, dtype=bool)
        else:
            beam_mask = np.logical_not(np.identity(self.nfeed, dtype=bool))

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
        comb_map, comb_mask = _merge_keyarray(
            base_map, beam_map, mask1=base_mask, mask2=beam_mask
        )

        # Take into account conjugation by identifying the indices of conjugate pairs
        conj_map = comb_map > comb_map.T
        comb_map = np.dstack((comb_map, comb_map.T)).min(axis=-1)
        comb_map = _remap_keyarray(comb_map, comb_mask)

        return comb_map, comb_mask, conj_map

    def _sort_pairs(self):
        """Re-order keys into a desired sort order.

        By default the order is lexicographic in (baseline u, baselines v,
        beamclass i, beamclass j).
        """

        # Create mask of included pairs, that are not conjugated
        tmask = np.logical_and(self._feedmask, np.logical_not(self._feedconj))
        uniq = _get_indices(self._feedmap, mask=tmask)

        fi, fj = uniq[:, 0], uniq[:, 1]

        # Fetch keys by which to sort (lexicographically)
        bx = self.feedpositions[fi, 0] - self.feedpositions[fj, 0]
        by = self.feedpositions[fi, 1] - self.feedpositions[fj, 1]
        ci = self.beamclass[fi]
        cj = self.beamclass[fj]

        ## Sort by constructing a numpy array with the keys as fields, and use
        ## np.argsort to get the indices

        # Create array of keys to sort
        dt = np.dtype("f8,f8,i4,i4")
        sort_arr = np.zeros(fi.size, dtype=dt)
        sort_arr["f0"] = bx
        sort_arr["f1"] = by
        sort_arr["f2"] = cj
        sort_arr["f3"] = ci

        # Get map which sorts
        sort_ind = np.argsort(sort_arr)

        # Invert mapping
        tmp_sort_ind = sort_ind.copy()
        sort_ind[tmp_sort_ind] = np.arange(sort_ind.size)

        # Remap feedmap entries
        fm_copy = self._feedmap.copy()
        wmask = np.where(self._feedmask)
        fm_copy[wmask] = sort_ind[self._feedmap[wmask]]

        self._feedmap = fm_copy

    def _skip_freq(self, freq_ind):
        """Override to control omission of specific frequencies in the beam transfers.

        The skipped frequencies will have entries in the beam transfers, but they
        will be zeros.

        Parameters
        ----------
        freq_ind
            The frequency index to determine if it is being skipped.

        Returns
        -------
        skip
            True if the frequency should be omitted, False if not.
        """
        return freq_ind in self.skip_freq

    def _skip_baseline(self, bl_ind):
        """Override to control omission of specific baselines in the beam transfers.

        The skipped baselines will have entries in the beam transfers, but they
        will be zeros.

        Parameters
        ----------
        bl_ind
            The baseline index to check if it is being skipped.

        Returns
        -------
        skip
            True if the baseline should be omitted, False if not.
        """
        return bl_ind in self.skip_baselines

    @cached_property
    def included_freq(self) -> np.ndarray:
        """The frequency indices that *are* being calculated.

        Returns
        -------
        freq_ind
            Indices of included frequencies.
        """
        return np.array(
            [ind for ind in range(self.nfreq) if not self._skip_freq(ind)], dtype=int
        )

    @cached_property
    def included_baseline(self) -> np.ndarray:
        """The baseline indices that *are* being calculated.

        Returns
        -------
        bl_ind
            Indices of included baselines.
        """
        return np.array(
            [ind for ind in range(self.nbase) if not self._skip_baseline(ind)],
            dtype=int,
        )

    @cached_property
    def included_pol(self) -> np.ndarray:
        """The pol indices that *are* being calculated.

        Returns
        -------
        pol_ind
            Indices of included polarisations.
        """
        return np.arange(self.num_pol_sky)

    # ===================================================

    # ==== Methods for calculating Transfer matrices ====

    def transfer_matrices(self, bl_indices, f_indices, global_lmax=True):
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

        # Check indices are all in range
        if out_of_range(bl_indices, 0, self.npairs):
            raise ValueError("Baseline indices aren't valid")

        if out_of_range(f_indices, 0, self.nfreq):
            raise ValueError("Frequency indices aren't valid")

        # Fetch the set of lmax's for the baselines (in order to reduce time
        # regenerating Healpix maps)
        lmax, mmax = np.ceil(
            self.l_boost
            * np.array(
                max_lm(
                    self.baselines[bl_indices],
                    self.wavelengths[f_indices],
                    self.u_width,
                    self.v_width,
                )
            )
        ).astype(np.int64)
        # lmax, mmax = lmax * self.l_boost, mmax * self.l_boost
        # Set the size of the (l,m) array to write into
        lside = self.lmax if global_lmax else lmax.max()

        # Generate the array for the Transfer functions

        tshape = bl_indices.shape + (self.num_pol_sky, lside + 1, 2 * lside + 1)
        logger.info(
            "Size: %i elements. Memory %f GB."
            % (np.prod(tshape), 2 * np.prod(tshape) * 8.0 / 2**30)
        )
        tarray = np.zeros(tshape, dtype=np.complex128)

        # Sort the baselines by ascending lmax and iterate through in that
        # order, calculating the transfer matrices
        for iflat in np.argsort(lmax.flat):
            ind = np.unravel_index(iflat, lmax.shape)

            trans = self._transfer_single(
                bl_indices[ind], f_indices[ind], lmax[ind], lside
            )

            ## Iterate over pol combinations and copy into transfer array
            for pi in range(self.num_pol_sky):
                islice = ind + (pi,) + (slice(None), slice(None))
                tarray[islice] = trans[pi]

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

    # ===================================================

    # ======== Noise properties of the telescope ========

    def tsys(self, f_indices=None):
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
        if f_indices is None:
            freq = self.frequencies
        else:
            freq = self.frequencies[f_indices]
        return np.ones_like(freq) * self.tsys_flat

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

        ndays = self.ndays if not ndays else ndays  # Set to value if not set.

        # Broadcast arrays against each other
        bl_indices, f_indices = np.broadcast_arrays(bl_indices, f_indices)

        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        delnu = units.t_sidereal * bw / (2 * np.pi)
        noisepower = self.tsys(f_indices) ** 2 / (2 * np.pi * delnu * ndays)
        noisebase = noisepower / self.redundancy[bl_indices]

        return noisebase

    def noisepower_feedpairs(self, fi, fj, f_indices, m, ndays=None):
        ndays = self.ndays if not ndays else ndays

        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        delnu = units.t_sidereal * bw / (2 * np.pi)
        noisepower = self.tsys(f_indices) ** 2 / (2 * np.pi * delnu * ndays)

        return (
            np.ones_like(fi) * np.ones_like(fj) * np.ones_like(m) * noisepower / 2.0
        )  # For unpolarised only at the moment.

    # ===================================================

    _nside = None

    def _init_trans(self, nside):
        ## Internal function for generating some common Healpix maps (position,
        ## horizon). These should need to be generated only when nside changes.

        # Angular positions in healpix map of nside
        self._nside = nside
        self._angpos = hputil.ang_positions(nside)

        # The horizon function
        self._horizon = visibility.horizon(self._angpos, self.zenith)

    _beam_cache = None

    def _beam(self, feed_ind, freq_ind):
        # Cache the beam maps by (nside, freq, beamclass/pol) to minimise recomputation

        if self._beam_cache is None:
            self._beam_cache = cache.NumpyCache(self.beam_cache_size << 20)

        # Key by the beam class, and not the feed_index to allow for many beams being
        # identical
        beamclass = self.beamclass[feed_ind]

        beam_key = (self._nside, freq_ind, beamclass)

        if beam_key not in self._beam_cache:
            beam = self.beam(feed_ind, freq_ind)
            self._beam_cache[beam_key] = beam
        else:
            beam = self._beam_cache[beam_key]

        return beam

    # ===================================================

    # ====== Properties to help with draco pipeline =====

    @cached_property
    def prodstack(self):
        """Generate the results of a prodstack.

        This is similar to the output of `uniquepairs`, but has the same typing as used
        within draco.

        Returns
        -------
        prodstack : np.ndarray
            A structured array with (input_a, input_b) pairs.
        """
        upairs = self.uniquepairs

        # Construct the return type using the same dtype length as used in the telescope
        dtype = [("input_a", upairs.dtype), ("input_b", upairs.dtype)]

        return upairs.ravel().view(dtype)

    @cached_property
    def index_map_prod(self):
        """Generate a *full triangle* `index_map/prod` like object.

        Returns
        -------
        prodmap : np.ndarray
            A structured array of (input_a, input_b) pairs for the upper triangle.
        """
        tpairs = np.array(np.triu_indices(self.nfeed))
        dtype = [("input_a", tpairs.dtype), ("input_b", tpairs.dtype)]

        return tpairs.T.flatten().view(dtype)

    @cached_property
    def index_map_stack(self):
        """Generate an `index_map/stack` like object.

        Returns
        -------
        stack : np.ndarray
            A structured array with (prod_ind, conj) pairs the same length as
            `unique_pairs`.
        """

        # Taken from draco.util.tools.cmap, but we can't depend on it in driftscan
        # NOTE: garbage if i > j
        def ind2tri(i, j, n):
            return (n * (n + 1) // 2) - ((n - i) * (n - i + 1) // 2) + (j - i)

        upairs = self.uniquepairs

        stack_map = np.empty(len(upairs), dtype=[("prod", "<u4"), ("conjugate", "u1")])

        stack_map["conjugate"] = upairs[:, 0] > upairs[:, 1]
        input_a, input_b = np.where(stack_map["conjugate"], upairs[:, ::-1].T, upairs.T)

        stack_map["prod"] = ind2tri(input_a, input_b, self.nfeed)

        return stack_map

    @cached_property
    def reverse_map_stack(self):
        """Generate a `reverse_map/stack` like object.

        Returns
        -------
        stack : np.ndarray
            A structured array of (stack_ind, conj) pairs the same length as `prod`.
        """

        stack_revmap = np.empty(
            self.nfeed * (self.nfeed + 1) // 2,
            dtype=[("stack", "<i4"), ("conjugate", "u1")],
        )

        stack_revmap["stack"] = self.feedmap[np.triu_indices(self.nfeed)]
        stack_revmap["conjugate"] = self.feedconj[np.triu_indices(self.nfeed)]

        return stack_revmap

    # ===================================================

    # ===================================================
    # ================ ABSTRACT METHODS =================
    # ===================================================

    # Implement to specify feed positions in the telescope.
    @abc.abstractproperty
    def feedpositions(self):
        """An (nfeed,2) array of the feed positions relative to an arbitary point (in m)"""
        return

    # Implement to specify the beams of the telescope
    @abc.abstractproperty
    def beamclass(self):
        """An nfeed array of the class of each beam (identical labels are
        considered to have identical beams)."""
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

    # ===================================================
    # ============== END ABSTRACT METHODS ===============
    # ===================================================


class UnpolarisedTelescope(TransitTelescope, metaclass=abc.ABCMeta):
    """A base for an unpolarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the `beam` function.
    """

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

    # ===== Implementations of abstract functions =======

    def _beam_map_single(self, bl_index, f_index):
        # Get beam maps for each feed.
        feedi, feedj = self.uniquepairs[bl_index]
        beami, beamj = self._beam(feedi, f_index), self._beam(feedj, f_index)

        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        pxarea = 4 * np.pi / beami.shape[0]

        # Beam solid angle (integrate over beam^2 - equal area pixels)
        om_i = np.sum(np.abs(beami) ** 2 * self._horizon) * pxarea
        om_j = np.sum(np.abs(beamj) ** 2 * self._horizon) * pxarea

        omega_A = (om_i * om_j) ** 0.5

        # Calculate the complex visibility transfer function
        cvis = self._horizon * fringe * beami * beamj.conjugate() / omega_A

        return cvis

    def _transfer_single(self, bl_index, f_index, lmax, lside):
        if self._nside != hputil.nside_for_lmax(
            lmax, accuracy_boost=self.accuracy_boost
        ):
            self._init_trans(
                hputil.nside_for_lmax(lmax, accuracy_boost=self.accuracy_boost)
            )

        cvis = self._beam_map_single(bl_index, f_index)

        # Perform the harmonic transform to get the transfer matrix (conj is correct - see paper)
        btrans = hputil.sphtrans_complex(
            cvis.conj(), centered=False, lmax=lmax, lside=lside
        ).conj()

        return [btrans]

    # ===================================================

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

        return bnoise[..., np.newaxis] * 0.5  # Correction for unpolarisedness


class PolarisedTelescope(TransitTelescope, metaclass=abc.ABCMeta):
    """A base for a polarised telescope.

    Again, an abstract class, but the only things that require implementing
    are the `feedpositions`, `_get_unique` and the `beam` function, as well
    as the polarization property.

    Attributes
    ----------
    skip_V : bool, optional
        Omit calculation of Stokes V transfer function to a mild computational
        saving.  As there is almost no Stokes V emission on the sky this is a
        reasonable trade off.  The entries are left in the transfer matrices, but
        they will be filled with zeros.
    skip_pol
        Omit calculation of Stokes Q, U and V for a large computational saving.  This
        means that the effect of the (large) polarized signal from the sky will not
        be correctly calculated.  Only do this if you are *really* sure it's what you
        want.  The entries are left in the transfer matrices, but they will be filled
        with zeros.

    Methods
    -------
    beam : methods
        (abstract method) Routines giving the field pattern for the x and y feeds.
    """

    skip_V = config.Property(proptype=bool, default=False)
    skip_pol = config.Property(proptype=bool, default=False)

    _npol_sky_ = 4

    @property
    def polarisation(self):
        """
        Polarisation map.

        Returns
        -------
        pol : np.ndarray
            One-dimensional array of strings describing the polarisation.
        """
        raise NotImplementedError("`polarisation` must be implemented.")

    def _beam_map_single(self, bl_index, f_index):
        # Get beam maps for each feed.
        feedi, feedj = self.uniquepairs[bl_index]
        beami, beamj = self._beam(feedi, f_index), self._beam(feedj, f_index)

        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)
        horizon = self._horizon.astype(np.float64)

        if np.iscomplexobj(beami) or np.iscomplexobj(beamj):
            cv_stokes = _construct_pol_complex(beami, beamj, fringe, horizon)
        else:
            cv_stokes = _construct_pol_real(beami, beamj, fringe, horizon)

        return cv_stokes

    # ===== Implementations of abstract functions =======

    def _transfer_single(self, bl_index, f_index, lmax, lside):
        if self._nside != hputil.nside_for_lmax(lmax):
            self._init_trans(hputil.nside_for_lmax(lmax))

        # Fetch and conjugate the beam maps
        bmap = self._beam_map_single(bl_index, f_index).conj()

        btrans = np.zeros(
            (self._npol_sky_, lside + 1, 2 * lside + 1), dtype=np.complex128
        )

        if self.skip_pol:
            # Perform the SHTs of the beam maps, only process Stokes I
            btrans[0] = hputil.sphtrans_complex(
                bmap[0], lmax=lmax, lside=lside, centered=False
            ).conj()
        else:
            # Perform the SHTs of the beam maps, potentially skipping Stokes V
            npol = 3 if self.skip_V else 4

            # Copy over output, this works around the fact that older versions of cora
            # return a list of maps
            # TODO: switch to simple array assignment when cora has been fixed up
            t = hputil.sphtrans_complex_pol(
                bmap[:npol], centered=False, lmax=lmax, lside=lside
            )
            for pi in range(npol):
                btrans[pi] = t[pi].conj()

        return btrans

    @cached_property
    def included_pol(self) -> np.ndarray:
        """The included polarisation indices.

        Returns
        -------
        pol_ind
            Polarisation indices.
        """

        if self.skip_pol:
            npol = 1
        elif self.skip_V:
            npol = 3
        else:
            npol = 4

        return np.arange(npol)

    # ===================================================


class SimpleUnpolarisedTelescope(UnpolarisedTelescope, metaclass=abc.ABCMeta):
    """A base for a polarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.

    Methods
    -------
    beamx, beamy : methods
        (abstract methods) Routines giving the field pattern for the x and y feeds.
    """

    @property
    def beamclass(self):
        """Simple beam mode of dual polarisation feeds."""
        return np.zeros(self._single_feedpositions.shape[0], dtype=np.int64)

    @abc.abstractproperty
    def _single_feedpositions(self):
        """An (nfeed,2) array of the feed positions relative to an arbitary point (in m)"""
        return

    @property
    def feedpositions(self):
        return self._single_feedpositions


class SimplePolarisedTelescope(PolarisedTelescope, metaclass=abc.ABCMeta):
    """A base for a polarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.

    Methods
    -------
    beamx, beamy : methods
        (abstract methods) Routines giving the field pattern for the x and y feeds.
    """

    @property
    def polarisation(self):
        """
        Polarisation map.

        Returns
        -------
        pol : np.ndarray
            One-dimensional array with the polarization for each feed ('X' or 'Y').
        """
        return np.asarray(
            ["X" if feed % 2 == 0 else "Y" for feed in self.beamclass], dtype=str
        )

    @property
    def beamclass(self):
        """Simple beam mode of dual polarisation feeds."""
        nsfeed = self._single_feedpositions.shape[0]
        return np.concatenate((np.zeros(nsfeed), np.ones(nsfeed))).astype(np.int64)

    def beam(self, feed, freq):
        if self.polarisation[feed] == "X":
            return self.beamx(feed, freq)
        else:
            return self.beamy(feed, freq)

    @abc.abstractproperty
    def _single_feedpositions(self):
        """An (nfeed,2) array of the feed positions relative to an arbitary point (in m)"""
        return

    @property
    def feedpositions(self):
        return np.concatenate((self._single_feedpositions, self._single_feedpositions))

    @abc.abstractmethod
    def beamx(self, feed, freq):
        """Beam for the X polarisation feed.

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
        """Beam for the Y polarisation feed.

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
