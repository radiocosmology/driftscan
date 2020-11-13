"""Estimate powerspectra and forecast constraints from real data.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os
import abc
import time

import h5py
import numpy as np
import scipy.linalg as la

from caput import config, mpiutil

from cora.signal import corr21cm

from drift.core import skymodel
from drift.util import util
from draco.analysis.svdfilter import external_svdfile

from mpi4py import MPI
from future.utils import with_metaclass


def uniform_band(k, kstart, kend):
    return np.where(
        np.logical_and(k > kstart, k < kend), np.ones_like(k), np.zeros_like(k)
    )


def bandfunc_2d_polar(ks, ke, ts, te):
    def band(k, mu):

        # k = (kpar**2 + kperp**2)**0.5
        theta = np.arccos(mu)

        tb = (theta >= ts) * (theta <= te)
        kb = (k >= ks) * (k < ke)

        return (kb * tb).astype(np.float64)

    return band


def bandfunc_2d_cart(kpar_s, kpar_e, kperp_s, kperp_e):
    def band(k, mu):

        kpar = k * mu
        kperp = k * (1.0 - mu ** 2) ** 0.5

        parb = (kpar >= kpar_s) * (kpar <= kpar_e)
        perpb = (kperp >= kperp_s) * (kperp < kperp_e)

        return (parb * perpb).astype(np.float64)

    return band


def range_config(lst):

    lst2 = []

    endpoint = False
    count = 1
    for item in lst:
        if isinstance(item, dict):
            if count == len(lst):
                endpoint = True
            count += 1

            if item["spacing"] == "log":
                item = np.logspace(
                    np.log10(item["start"]),
                    np.log10(item["stop"]),
                    item["num"],
                    endpoint=endpoint,
                )
            elif item["spacing"] == "linear":
                item = np.linspace(
                    item["start"], item["stop"], item["num"], endpoint=endpoint
                )

            item = np.atleast_1d(item)

            lst2.append(item)
        else:
            raise Exception("Require a dict.")

    return np.concatenate(lst2)


def decorrelate_ps(ps, fisher):
    """Decorrelate the powerspectrum estimate.

    Parameters
    ----------
    ps : np.ndarray[nbands]
        Powerspectrum estimate.
    fisher : np.ndarrays[nbands, nbands]
        Fisher matrix.

    Returns
    -------
    psd : np.narray[nbands]
        Decorrelated powerspectrum estimate.
    errors : np.ndarray[nbands]
        Errors on decorrelated bands.
    window : np.ndarray[nbands, nbands]
        Window functions for each band row-wise.
    """
    # Factorise the Fisher matrix
    fh = la.cholesky(fisher, lower=True)
    fhi = la.inv(fh)

    # Create the mixing matrix, and window functions
    m = fhi / np.sum(fh.T, axis=1)[:, np.newaxis]
    w = np.dot(m, fisher)

    # Find the decorrelated powerspectrum and its errors
    evm = np.dot(m, np.dot(fisher, m.T)).diagonal() ** 0.5
    psd = np.dot(w, ps)

    return psd, evm, w


def decorrelate_ps_file(fname):
    """Load and decorrelate the powerspectrum in `fname`.

    Parameters
    ----------
    fname : string
        Name of file to load.

    Returns
    -------
    psd : np.narray[nbands]
        Decorrelated powerspectrum estimate.
    errors : np.ndarray[nbands]
        Errors on decorrelated bands.
    window : np.ndarray[nbands, nbands]
        Window functions for each band row-wise.
    """
    f1 = h5py.File(fname, "r")

    return decorrelate_ps(f1["powerspectrum"][:], f1["fisher"][:])


class PSEstimation(with_metaclass(abc.ABCMeta, config.Reader)):
    """Base class for quadratic powerspectrum estimation.

    See Tegmark 1997 for details.

    Attributes
    ----------
    bandtype : {'polar', 'cartesian'}
        Which types of bands to use (default: polar).


    k_bands : np.ndarray
        Array of band boundaries. e.g. np.array([0.0, 0.5, ]), polar only
    num_theta: integer
        Number of theta bands to use (polar only)

    kpar_bands : np.ndarray
        Array of band boundaries. e.g. np.array([0.0, 0.5, ]), cartesian only
    kperp_bands : np.ndarray
        Array of band boundaries. e.g. np.array([0.0, 0.5, ]), cartesian only

    threshold : scalar
        Threshold for including eigenmodes (default is 0.0, i.e. all modes)

    unit_bands : boolean
        If True, bands are sections of the exact powerspectrum (such that the
        fiducial bin amplitude is 1).

    zero_mean : boolean
        If True (default), then the fiducial parameters have zero mean.

    external_svd_basis_dir : string, optional
        Directory containing files defining external SVD basis (determined
        directly from measured visibilities, using
        draco.analysis.svdfilter.SVDFilter). If specified, C_a
        matrices are filtered according to thresholds below. Default: None.
    external_sv_threshold_global : float, optional
        Global external-SVD mode filtering threshold: removes external SVD
        modes with SV higher than this value times the largest mode at any m.
        Default: 1000 (i.e. no filtering).
    external_sv_threshold_local : float, optional
        As above, but removes modes with SV higher than this value times the
        largest mode at each m. Default: 1000 (i.e. no filtering).
    external_sv_mode_cut : int, optional
        If specified, supercede local and global thresholds, and just remove
        the first external_sv_mode_cut modes at each m. Default: None
    use_external_sv_freq_modes : bool, optional
        If True, use ext-SVD modes defined as combinations of frequencies. If
        False, use modes defined as combinations of baselines or tel-SVD
        modes. Default: False
    external_sv_from_m_modes : bool, optional
        If True, assume external-SVD modes are defined in m-mode space.
        If False, assume external-SVD modes are defined in telescope-SVD space
        (not currently implemented).
        Default: True
    """

    bandtype = config.Property(proptype=str, default="polar")

    # Properties to control polar bands
    k_bands = config.Property(
        proptype=range_config,
        default=[{"spacing": "linear", "start": 0.0, "stop": 0.4, "num": 20}],
    )
    num_theta = config.Property(proptype=int, default=1)

    # Properties for cartesian bands
    kpar_bands = config.Property(
        proptype=range_config,
        default=[{"spacing": "linear", "start": 0.0, "stop": 0.4, "num": 20}],
    )
    kperp_bands = config.Property(
        proptype=range_config,
        default=[{"spacing": "linear", "start": 0.0, "stop": 0.4, "num": 20}],
    )

    threshold = config.Property(proptype=float, default=0.0)

    unit_bands = config.Property(proptype=bool, default=True)

    zero_mean = config.Property(proptype=bool, default=True)

    external_svd_basis_dir = config.Property(proptype=str, default=None)

    external_sv_threshold_global = config.Property(proptype=float, default=1000.)
    external_sv_threshold_local = config.Property(proptype=float, default=1000.)
    external_sv_mode_cut = config.Property(proptype=int, default=None)
    use_external_sv_freq_modes = config.Property(proptype=bool, default=False)
    external_sv_from_m_modes = config.Property(proptype=bool, default=False)

    crosspower = False

    clarray = None

    fisher = None
    bias = None


    def __init__(self, kltrans, subdir="ps"):
        """Initialise a PS estimator class.

        Parameters
        ----------
        kltrans : KLTransform
            The KL Transform filter to use.
        subdir : string, optional
            Subdirectory of the KLTransform directory to store results in.
            Default is 'ps'.
        """

        self.kltrans = kltrans
        self.telescope = kltrans.telescope
        self.psdir = self.kltrans.evdir + "/" + subdir + "/"

        if mpiutil.rank0 and not os.path.exists(self.psdir):
            os.makedirs(self.psdir)

        # Define empty dicts for storing ext-SVD information, and set all entries
        # to None
        self.ext_svd_u = {}
        self.ext_svd_sig = {}
        self.ext_svd_vh = {}
        self.ext_svd_cut = {}
        self.beam_svd_with_ext_filtering = {}
        for mi in range(self.telescope.mmax + 1):
            self.ext_svd_u[mi] = None
            self.ext_svd_sig[mi] = None
            self.ext_svd_vh[mi] = None
            self.ext_svd_cut[mi] = None
            self.beam_svd_with_ext_filtering[mi] = None

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()


    @property
    def nbands(self):
        """Number of powerspectrum bands."""
        return self.k_center.size

    def num_evals(self, mi):
        """Number of eigenvalues for this `m` (and threshold).

        Parameters
        ----------
        mi : integer
            m-mode index.

        Returns
        -------
        num_evals : integer
        """

        evals = self.kltrans.modes_m(mi, threshold=self.threshold)[0]

        return evals.size if evals is not None else 0

    def _external_svdfile(self, mi):
        """File containing external SVD basis for a given m.
        """
        if self.external_svd_basis_dir is None:
            raise RuntimeError("Directory containing external SVD basis not specified!")
        else:
            return external_svdfile(self.external_svd_basis_dir, mi, self.telescope.mmax)

    def set_external_global_max_sv(self):
        if self.external_svd_basis_dir is not None:
            # Loop over all m's to find the maximum singular value
            max_sv = 0.

            for mi in mpiutil.mpirange(self.telescope.mmax + 1):
                # Skip this m if SVD basis file doesn't exist.
                if not os.path.exists(self._external_svdfile(mi)):
                    continue

                fe = h5py.File(self._external_svdfile(mi), 'r')
                sig = fe["sig"][:]
                max_sv = max(sig[0], max_sv)
                fe.close()

            self.external_global_max_sv = mpiutil.world.allreduce(max_sv, op=MPI.MAX)
            # print("Max external SV over all m: %g" % self.external_global_max_sv)

    # ========== Calculate powerspectrum bands ==========

    def genbands(self):
        """Precompute the powerspectrum bands, including the P(k, mu) bands
        and the angular powerspectrum.
        """

        print("Generating bands...")

        cr = corr21cm.Corr21cm()
        cr.ps_2d = False

        # Create different sets of bands depending on whether we're using polar bins or not.
        if self.bandtype == "polar":

            # Create the array of band bounds
            self.theta_bands = np.linspace(
                0.0, np.pi / 2.0, self.num_theta + 1, endpoint=True
            )

            # Broadcast the bounds against each other to make the 2D array of bands
            kb, tb = np.broadcast_arrays(
                self.k_bands[np.newaxis, :], self.theta_bands[:, np.newaxis]
            )

            # Pull out the start, end and centre of the bands in k, mu directions
            self.k_start = kb[1:, :-1].flatten()
            self.k_end = kb[1:, 1:].flatten()
            self.k_center = 0.5 * (self.k_end + self.k_start)

            self.theta_start = tb[:-1, 1:].flatten()
            self.theta_end = tb[1:, 1:].flatten()
            self.theta_center = 0.5 * (self.theta_end + self.theta_start)

            bounds = list(
                zip(self.k_start, self.k_end, self.theta_start, self.theta_end)
            )

            # Make a list of functions of the band window functions
            self.band_func = [bandfunc_2d_polar(*bound) for bound in bounds]

            # Create a list of functions of the band power functions
            if self.unit_bands:
                # Need slightly awkward double lambda because of loop closure scaling.
                self.band_pk = [
                    (lambda bandt: (lambda k, mu: cr.ps_vv(k) * bandt(k, mu)))(band)
                    for band in self.band_func
                ]
                self.band_power = np.ones_like(self.k_start)
            else:
                self.band_pk = self.band_func
                self.band_power = cr.ps_vv(self.k_center)

        elif self.bandtype == "cartesian":

            # Broadcast the bounds against each other to make the 2D array of bands
            kparb, kperpb = np.broadcast_arrays(
                self.kpar_bands[np.newaxis, :], self.kperp_bands[:, np.newaxis]
            )

            # Pull out the start, end and centre of the bands in k, mu directions
            self.kpar_start = kparb[1:, :-1].flatten()
            self.kpar_end = kparb[1:, 1:].flatten()
            self.kpar_center = 0.5 * (self.kpar_end + self.kpar_start)

            self.kperp_start = kperpb[:-1, 1:].flatten()
            self.kperp_end = kperpb[1:, 1:].flatten()
            self.kperp_center = 0.5 * (self.kperp_end + self.kperp_start)

            bounds = list(
                zip(self.kpar_start, self.kpar_end, self.kperp_start, self.kperp_end)
            )

            self.k_center = (self.kpar_center ** 2 + self.kperp_center ** 2) ** 0.5

            # Make a list of functions of the band window functions
            self.band_func = [bandfunc_2d_cart(*bound) for bound in bounds]

        else:
            raise Exception("Bandtype %s is not supported." % self.bandtype)

        # Create a list of functions of the band power functions
        if self.unit_bands:
            # Need slightly awkward double lambda because of loop closure scaling.
            self.band_pk = [
                (lambda bandt: (lambda k, mu: cr.ps_vv(k) * bandt(k, mu)))(band)
                for band in self.band_func
            ]
            self.band_power = np.ones_like(self.k_center)
        else:
            self.band_pk = self.band_func
            self.band_power = cr.ps_vv(self.k_center)

        # Use new parallel map to speed up computaiton of bands
        if self.clarray is None:

            self.make_clzz_array()

        print("Done.")

    def make_clzz(self, pk):
        """Make an angular powerspectrum from the input matter powerspectrum.

        Uses the lmax and frequencies from the telescope object.

        Parameters
        ----------
        pk : function, np.ndarray -> np.ndarray
            The input powerspectrum (must be vectorized).

        Returns
        -------
        aps : np.ndarray[lmax+1, nfreq, nfreq]
            The angular powerspectrum.
        """
        crt = corr21cm.Corr21cm(ps=pk, redshift=1.5)
        crt.ps_2d = True

        clzz = skymodel.im21cm_model(
            self.telescope.lmax,
            self.telescope.frequencies,
            self.telescope.num_pol_sky,
            cr=crt,
            temponly=True,
        )

        print("Rank: %i - Finished making band." % mpiutil.rank)
        return clzz

    def make_clzz_array(self):

        p_bands, s_bands, e_bands = mpiutil.split_all(self.nbands)
        p, s, e = mpiutil.split_local(self.nbands)

        self.clarray = np.zeros(
            (
                self.nbands,
                self.telescope.lmax + 1,
                self.telescope.nfreq,
                self.telescope.nfreq,
            ),
            dtype=np.float64,
        )

        for bi in range(s, e):
            self.clarray[bi] = self.make_clzz(self.band_pk[bi])

        bandsize = (
            (self.telescope.lmax + 1) * self.telescope.nfreq * self.telescope.nfreq
        )
        sizes = p_bands * bandsize
        displ = s_bands * bandsize

        MPI.COMM_WORLD.Allgatherv(
            MPI.IN_PLACE, [self.clarray, sizes, displ, MPI.DOUBLE]
        )

    def delbands(self):
        """Delete power spectrum bands to save memory."""

        self.clarray = None

    # ===================================================

    # ==== Calculate the per-m Fisher matrix/bias =======

    def fisher_bias_m(self, mi):
        """Generate the Fisher matrix and bias for a specific m.

        Parameters
        ----------
        mi : integer
            m-mode to calculate for.

        """

        if self.num_evals(mi) > 0:
            print("Making fisher (for m=%i)." % mi)

            # If doing ext-SVD filtering, read basis and determine number
            # of modes to cut for this m
            if self.external_svd_basis_dir is not None \
                and os.path.exists(self._external_svdfile(mi)):
                self._read_ext_svd_info(mi)

            fisher, bias = self._work_fisher_bias_m(mi)

            ###SJF temp
            if mi == 2: print('m=2 fisher:', fisher)

            # Delete ext-SVD info for this m, to save memory
            if self.external_svd_basis_dir is not None \
                and os.path.exists(self._external_svdfile(mi)):
                self._del_ext_svd_info(mi)

        else:
            print("No evals (for m=%i), skipping." % mi)

            fisher = np.zeros((self.nbands, self.nbands), dtype=np.complex128)
            bias = np.zeros((self.nbands,), dtype=np.complex128)

        return fisher, bias

    @abc.abstractmethod
    def _work_fisher_bias_m(self, mi):
        """Worker routine for calculating the Fisher and bias for a given m.

        This routine should be overriden for a new method of generating the Fisher matrix.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Returns
        -------
        fisher : np.ndarray[nbands, nbands]
            Fisher matrix.
        bias : np.ndarray[nbands]
            Bias vector.
        """
        pass

    def _read_ext_svd_info(self, mi):
        """Read external-SVD info for this m and compute modified beam-SVD matrix.

        This routine reads in the ext-SVD U, Sig, and Vh matrices from disk,
        computes the number of modes to cut, and stores this info in dicts,
        so that it can be accessed by the q_estimator routine. It then
        computes and stores a modified beam_svd matrix that can be used to project
        from the sky to telescope-SVD basis with ext-SVD mode filtering included.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Returns
        -------
        status : bool
            True if stuff was read from disk, False if it already existed
            in memory.
        """
        # If everything is in the dicts already, return False
        if self.ext_svd_u[mi] is not None and self.ext_svd_sig[mi] is not None \
            and self.ext_svd_vh[mi] is not None and self.ext_svd_cut[mi] is not None:
            return False

        # Open file for this m, and read U, singular values, and Vh
        fe = h5py.File(self._external_svdfile(mi), 'r')
        self.ext_svd_u[mi] = fe["u"][:]
        self.ext_svd_sig[mi] = fe["sig"][:]
        self.ext_svd_vh[mi] = fe["vh"][:]
        fe.close()

        if self.external_sv_mode_cut is not None:
            self.ext_svd_cut[mi] = self.external_sv_mode_cut
        else:
            # Determine how many modes to cut, based on global and local thresholds
            global_ext_sv_cut = (self.ext_svd_sig[mi] > self.external_sv_threshold_global
                                 * self.external_global_max_sv).sum()
            local_ext_sv_cut = (self.ext_svd_sig[mi] \
                > self.external_sv_threshold_local * self.ext_svd_sig[mi][0]).sum()
            self.ext_svd_cut[mi] = max(global_ext_sv_cut, local_ext_sv_cut)

        # Get matrix that does sky-to-tel transform.
        # Matrix comes packed as [nfreq,msign,nbase,npol,lmax+1],
        # but we reshape to [nfreq, msign*nbase (=ntel), npol*(lmax+1) (=nsky)]
        beam_m = self.kltrans.beamtransfer.beam_m(mi)
        beam_m = beam_m.reshape(
            self.telescope.nfreq,
            self.kltrans.beamtransfer.ntel,
            -1
        )

        # Get matrix that does tel-to-tel-SVD transform.
        # Matrix comes packed as [nfreq,nsvd,ntel].
        # Note that beam_svd is just the dot product of beam_m and beam_ut.
        beam_ut = self.kltrans.beamtransfer.beam_ut(mi)

        # Apply ext-SVD projection to beam_m, along freq axis
        if self.use_external_sv_freq_modes:
            # Construct matrix that filters out freq (V) modes of ext-SVD
            vh = self.ext_svd_vh[mi]
            Z = np.ones(self.telescope.nfreq)
            Z[:self.ext_svd_cut[mi]] = 0
            Z = np.diag(Z)
            proj = np.dot(vh.T.conj(), np.dot(Z, vh))

            # Apply filtering to beam_m
            beam_m_proj = np.dot(proj, beam_m.reshape(self.telescope.nfreq, -1)).reshape(
                self.telescope.nfreq,
                self.kltrans.beamtransfer.ntel,
                -1
            )

        else:
            raise NotImplementedError('u proj not implented!')

        # Construct beam_svd matrix
        beam_svd = np.zeros((self.telescope.nfreq,
            beam_ut.shape[1],
            self.kltrans.beamtransfer.nsky
        ), dtype=np.complex128)
        for fi in np.arange(self.telescope.nfreq):
            beam_svd[fi] = np.dot(beam_ut[fi], beam_m_proj[fi])

        # Store beam_svd matrix
        self.beam_svd_with_ext_filtering[mi] = beam_svd

        return True

    def _del_ext_svd_info(self, mi):
        """Delete external-SVD info from memory for this m.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.
        """
        self.ext_svd_u[mi] = None
        self.ext_svd_sig[mi] = None
        self.ext_svd_vh[mi] = None
        self.ext_svd_cut[mi] = None
        self.beam_svd_with_ext_filtering[mi] = None

    # ===================================================

    # ==== Calculate the total Fisher matrix/bias =======

    def generate(self, regen=False):
        """Calculate the total Fisher matrix and bias and save to a file.

        Parameters
        ----------
        regen : boolean, optional
            Force regeneration if products already exist (default `False`).
        """

        if mpiutil.rank0:
            st = time.time()
            print("======== Starting PS calculation ========")

        ffile = self.psdir + "/fisher.hdf5"

        if os.path.exists(ffile) and not regen:
            print("Fisher matrix file: %s exists. Skipping..." % ffile)
            return

        if self.external_svd_basis_dir is not None and not self.external_sv_from_m_modes:
            raise NotImplementedError("Ext-SVD projection using tel-SVD modes "
                                        + "not implemented in PSEstimation!")

        # If external SVD filtering is desired, compute global max SV
        self.set_external_global_max_sv()

        mpiutil.barrier()

        # Pre-compute all the angular power spectra for the bands
        self.genbands()

        # Calculate Fisher and bias for each m
        # Pair up each list item with its position.
        zlist = list(enumerate(range(self.telescope.mmax + 1)))
        # Partition list based on MPI rank
        llist = mpiutil.partition_list_mpi(zlist)
        # Operate on sublist
        fisher_bias_list = [self.fisher_bias_m(item) for ind, item in llist]

        # Unpack into separate lists of the Fisher matrix and bias
        fisher_loc, bias_loc = zip(*fisher_bias_list)

        # Sum over all local m-modes to get the over all Fisher and bias pe process
        fisher_loc = np.sum(
            np.array(fisher_loc), axis=0
        ).real  # Be careful of the .real here
        bias_loc = np.sum(
            np.array(bias_loc), axis=0
        ).real  # Be careful of the .real here

        self.fisher = mpiutil.allreduce(fisher_loc, op=MPI.SUM)
        self.bias = mpiutil.allreduce(bias_loc, op=MPI.SUM)

        # Write out all the PS estimation products
        if mpiutil.rank0:
            et = time.time()
            print("======== Ending PS calculation (time=%f) ========" % (et - st))

            # Check to see ensure that Fisher matrix isn't all zeros.
            if not (self.fisher == 0).all():
                # Generate derived quantities (covariance, errors..)

                ###SJF temp
                print(self.fisher)

                cv = la.pinv(self.fisher, rcond=1e-8)
                err = cv.diagonal() ** 0.5
                cr = cv / np.outer(err, err)
            else:
                cv = np.zeros_like(self.fisher)
                err = cv.diagonal()
                cr = np.zeros_like(self.fisher)

            f = h5py.File(self.psdir + "/fisher.hdf5", "w")
            f.attrs["bandtype"] = np.string_(self.bandtype)  # HDF5 string issues

            f.create_dataset("fisher/", data=self.fisher)
            f.create_dataset("bias/", data=self.bias)
            f.create_dataset("covariance/", data=cv)
            f.create_dataset("errors/", data=err)
            f.create_dataset("correlation/", data=cr)

            f.create_dataset("band_power/", data=self.band_power)

            if self.bandtype == "polar":
                f.create_dataset("k_start/", data=self.k_start)
                f.create_dataset("k_end/", data=self.k_end)
                f.create_dataset("k_center/", data=self.k_center)

                f.create_dataset("theta_start/", data=self.theta_start)
                f.create_dataset("theta_end/", data=self.theta_end)
                f.create_dataset("theta_center/", data=self.theta_center)

                f.create_dataset("k_bands", data=self.k_bands)
                f.create_dataset("theta_bands", data=self.theta_bands)

            elif self.bandtype == "cartesian":

                f.create_dataset("kpar_start/", data=self.kpar_start)
                f.create_dataset("kpar_end/", data=self.kpar_end)
                f.create_dataset("kpar_center/", data=self.kpar_center)

                f.create_dataset("kperp_start/", data=self.kperp_start)
                f.create_dataset("kperp_end/", data=self.kperp_end)
                f.create_dataset("kperp_center/", data=self.kperp_center)

                f.create_dataset("kpar_bands", data=self.kpar_bands)
                f.create_dataset("kperp_bands", data=self.kperp_bands)

            f.close()

    # ===================================================

    def fisher_file(self):
        """Fetch the h5py file handle for the Fisher matrix.

        Returns
        -------
        file : h5py.File
            File pointing at the hdf5 file with the Fisher matrix.
        """
        return h5py.File(self.psdir + "fisher.hdf5", "r")

    def fisher_bias(self):

        with h5py.File(self.psdir + "/fisher.hdf5", "r") as f:

            return f["fisher"][:], f["bias"][:]

    # ===================================================

    # ====== Estimate the q-parameters from data ========

    def q_estimator(self, mi, vec1, vec2=None, noise=False):
        """Estimate the q-parameters from given data (see paper).

        Parameters
        ----------
        mi : integer
            The m-mode we are calculating for.
        vec : np.ndarray[num_kl, num_realisatons]
            The vector(s) of data we are estimating from. These are KL-mode
            coefficients.
        noise : boolean, optional
            Whether we should project against the noise matrix. Used for
            estimating the bias by Monte-Carlo. Default is False.

        Returns
        -------
        qa : np.ndarray[numbands]
            Array of q-parameters. If noise=True then the array is one longer,
            and the last parameter is the projection against the noise.
        """

        evals, evecs = self.kltrans.modes_m(mi)

        if evals is None:
            return np.zeros((self.nbands + 1 if noise else self.nbands,))

        # Weight by C**-1 (transposes are to ensure broadcast works for 1 and 2d vecs)
        x0 = (vec1.T / (evals + 1.0)).T

        # Project back into SVD basis
        x1 = np.dot(evecs.T.conj(), x0)

        if self.external_svd_basis_dir is not None \
            and os.path.exists(self._external_svdfile(mi)):
            # Read ext-SVD basis from disk and compute number of modes to cut
            ext_svd_from_file = self._read_ext_svd_info(mi)
            x2 = self._project_vector_svd_to_sky_with_ext_svd(mi, x1)

        else:
            # Project back into sky basis
            x2 = self.kltrans.beamtransfer.project_vector_svd_to_sky(mi, x1, conj=True)

        if vec2 is not None:
            y0 = (vec2.T / (evals + 1.0)).T
            y1 = np.dot(evecs.T.conj(), x0)
            if self.external_svd_basis_dir is not None \
                and os.path.exists(self._external_svdfile(mi)):
                y2 = self._project_vector_svd_to_sky_with_ext_svd(mi, y1)
            else:
                y2 = self.kltrans.beamtransfer.project_vector_svd_to_sky(mi, y1, conj=True)
        else:
            y0 = x0
            y2 = x2

        # Create empty q vector (length depends on if we're calculating the noise term too)
        qa = np.zeros((self.nbands + 1 if noise else self.nbands,) + vec1.shape[1:])

        lside = self.telescope.lmax + 1

        # Calculate q_a for each band
        for bi in range(self.nbands):

            for li in range(lside):

                lxvec = x2[:, 0, li]
                lyvec = y2[:, 0, li]

                qa[bi] += np.sum(
                    lyvec.conj()
                    * np.dot(self.clarray[bi][li].astype(np.complex128), lxvec),
                    axis=0,
                ).astype(
                    np.float64
                )  # TT only.

        # Calculate q_a for noise power (x0^H N x0 = |x0|^2)
        if noise:

            # If calculating crosspower don't include instrumental noise
            noisemodes = 0.0 if self.crosspower else 1.0
            noisemodes = noisemodes + (evals if self.zero_mean else 0.0)

            qa[-1] = np.sum((x0 * y0.conj()).T.real * noisemodes, axis=-1)

        if self.external_svd_basis_dir is not None and ext_svd_from_file:
            self._del_ext_svd_info(mi)

        return qa.real


    def _project_vector_svd_to_sky_with_ext_svd(self, mi, vec):
        """Project a vector from the telescope-SVD basis to the sky basis,
        incorporating ext-SVD filtering.

        This is the same as beamtransfer.project_vector_svd_to_sky(), but
        it uses the Hermitian conjugate of the sky-to-tel-SVD transform to
        do the reverse transform, and it includes an extra matrix that
        filters out a specified number of modes defined in the ext-SVD basis.

        Parameters
        ----------
        mi : integer
            The m-mode we are calculating for.
        vec : np.ndarray[num_kl] or np.ndarray[num_kl, num_realisatons]
            Input vector(s) in tel-SVD basis.

        Returns
        -------
        vec_sky : np.ndarray[nfreq, npol, lmax+1]
            Output vector(s) in sky basis.
        """
        # Fetch matrix that transforms from sky to tel-SVD basis
        beam_svd = self.beam_svd_with_ext_filtering[mi]

        # Fetch Number of significant tel-SVD modes at each frequency,
        # and the array bounds
        svnum, svbounds = self.kltrans.beamtransfer._svd_num(mi)

        # Make an empty array to store the results
        vec_sky = np.zeros(
            (self.telescope.nfreq, self.kltrans.beamtransfer.nsky) + vec.shape[1:],
            dtype=np.complex128
        )

        # For each frequency, select the relevant entries from the input vector(s),
        # and apply the Hermitian transpose of the beam_svd matrix
        for fi in self.kltrans.beamtransfer._svd_freq_iter(mi):
            lvec = vec[
                svbounds[fi] : svbounds[fi + 1]
            ]
            vec_sky[fi] = np.dot(beam_svd[fi, :svnum[fi]].T.conj(), lvec)

        # Reshape the result
        vec_sky = vec_sky.reshape(
            (self.telescope.nfreq,
            self.telescope.num_pol_sky,
            self.telescope.lmax + 1) + vec.shape[1:]
        )

        #### Debugging stuff
        # vec_sky_good = self.kltrans.beamtransfer.project_vector_svd_to_sky(mi, vec, conj=conj)
        # print(mi, np.allclose(vec_sky_good, vec_sky))
        # print(mi, vec_sky_good[0,0,10:20], '\n', vec_sky[0,0,10:20])

        return vec_sky


    # ===================================================


class PSExact(PSEstimation):
    """PS Estimation class with exact calculation of the Fisher matrix."""

    @property
    def _cfile(self):
        # Pattern to form the `m` ordered cache file.
        return (
            self.psdir
            + "/ps_c_m_"
            + util.intpattern(self.telescope.mmax)
            + "_b_"
            + util.natpattern(self.nbands - 1)
            + ".hdf5"
        )

    def makeproj(self, mi, bi):
        """Project angular powerspectrum band into KL-basis.

        Parameters
        ----------
        mi : integer
            m-mode.
        bi : integer
            band index.

        Returns
        -------
        klcov : np.ndarray[nevals, nevals]
            Covariance in KL-basis.
        """
        # print "Projecting to eigenbasis."
        # nevals = self.kltrans.modes_m(mi, threshold=self.threshold)[0].size

        # if nevals < 1000:
        #     return self.kltrans.project_sky_matrix_forward_old(mi, self.clarray[bi], self.threshold)
        # else:
        # return self.kltrans.project_sky_matrix_forward(mi, self.clarray[bi], self.threshold)

        clarray = self.clarray[bi].reshape((1, 1) + self.clarray[bi].shape)
        ###SJF temp
        print('clarray.shape:', clarray.shape)
        svdmat = self.kltrans.beamtransfer.project_matrix_sky_to_svd(
            mi, clarray, temponly=True
        )
        return self.kltrans.project_matrix_svd_to_kl(mi, svdmat, self.threshold)

    def cacheproj(self, mi):
        """Cache projected covariances on disk.

        Parameters
        ----------
        mi : integer
            m-mode.
        """

        ## Don't generate cache for small enough matrices
        if self.num_evals(mi) < 500:
            self._bp_cache = []

        for i in range(len(self.clarray)):
            print("Generating cache for m=%i band=%i" % (mi, i))
            projm = self.makeproj(mi, i)

            ## Don't generate cache for small enough matrices
            if self.num_evals(mi) < 500:
                self._bp_cache.append(projm)

            else:
                print("Creating cache file:" + self._cfile % (mi, i))
                f = h5py.File(self._cfile % (mi, i), "w")
                f.create_dataset("proj", data=projm)
                f.close()

    def delproj(self, mi):
        """Deleted cached covariances from disk.

        Parameters
        ----------
        mi : integer
            m-mode.
        """
        ## As we don't cache for small matrices, just return
        if self.num_evals(mi) < 500:
            self._bp_cache = []

        for i in range(len(self.clarray)):

            fn = self._cfile % (mi, i)
            if os.path.exists(fn):
                print("Deleting cache file:" + fn)
                os.remove(self._cfile % (mi, i))

    def getproj(self, mi, bi):
        """Fetch cached KL-covariance (either from disk or just calculate if small enough).

        Parameters
        ----------
        mi : integer
            m-mode.
        bi : integer
            band index.

        Returns
        -------
        klcov : np.ndarray[nevals, nevals]
            Covariance in KL-basis.
        """
        fn = self._cfile % (mi, bi)

        ## For small matrices or uncached files don't fetch cache, just generate
        ## immediately
        if self.num_evals(mi) < 500:  # or not os.path.exists:
            proj = self._bp_cache[bi]
            # proj = self.makeproj(mi, bi)
        else:
            f = h5py.File(fn, "r")
            proj = f["proj"][:]
            f.close()

        return proj

    def _work_fisher_bias_m(self, mi):
        """Worker routine for calculating the Fisher and bias for a given m.

        This method exactly calculates the quantities by forward projecting
        the correlations.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Returns
        -------
        fisher : np.ndarray[nbands, nbands]
            Fisher matrix.
        bias : np.ndarray[nbands]
            Bias vector.
        """

        evals = self.kltrans.evals_m(mi, self.threshold)

        fisher = np.zeros((self.nbands, self.nbands), dtype=np.complex128)
        bias = np.zeros(self.nbands, dtype=np.complex128)

        self.cacheproj(mi)

        ci = 1.0 / (evals + 1.0) ** 0.5
        ci = np.outer(ci, ci)

        for ia in range(self.nbands):
            c_a = self.getproj(mi, ia)
            fisher[ia, ia] = np.sum(c_a * c_a.T * ci ** 2)

            for ib in range(ia):
                c_b = self.getproj(mi, ib)
                fisher[ia, ib] = np.sum(c_a * c_b.T * ci ** 2)
                fisher[ib, ia] = np.conj(fisher[ia, ib])

        self.delproj(mi)

        return fisher, bias
