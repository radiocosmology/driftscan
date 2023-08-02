import logging
import time
import os
import re

import numpy as np
import scipy.linalg as la
import h5py

from caput import config, mpiutil

from cora.util import hputil

from drift.util import util
from drift.core import skymodel

# Get the module logging object
logger = logging.getLogger(__name__)


def collect_m_arrays(mlist, func, shapes, dtype):
    data = [(mi, func(mi)) for mi in mpiutil.partition_list_mpi(mlist)]

    mpiutil.barrier()

    if mpiutil.rank0 and mpiutil.size == 1:
        p_all = [data]
    else:
        p_all = mpiutil.world.gather(data, root=0)

    mpiutil.barrier()  # Not sure if this barrier really does anything,
    # but hoping to stop collect breaking

    marrays = None
    if mpiutil.rank0:
        marrays = [np.zeros((len(mlist),) + shape, dtype=dtype) for shape in shapes]

        for p_process in p_all:
            for mi, result in p_process:
                for si in range(len(shapes)):
                    if result[si] is not None:
                        marrays[si][mi] = result[si]

    mpiutil.barrier()

    return marrays


def collect_m_array(mlist, func, shape, dtype):
    res = collect_m_arrays(mlist, lambda mi: [func(mi)], [shape], dtype)

    return res[0] if mpiutil.rank0 else None


def eigh_gen(A, B, message=""):
    """Solve the generalised eigenvalue problem. :math:`\mathbf{A} \mathbf{v} =
    \lambda \mathbf{B} \mathbf{v}`

    This routine will attempt to correct for when `B` is not positive definite
    (usually due to numerical precision), by adding a constant diagonal to make
    all of its eigenvalues positive.

    Parameters
    ----------
    A, B : np.ndarray
        Matrices to operate on.
    message : string, optional
        Optional string to print if an exception is thrown. Default: "".

    Returns
    -------
    evals : np.ndarray
        Eigenvalues of the problem.
    evecs : np.ndarray
        2D array of eigenvectors (packed column by column).
    add_const : scalar
        The constant added on the diagonal to regularise.
    """
    add_const = 0.0

    if (A == 0).all():
        evals, evecs = (
            np.zeros(A.shape[0], dtype=A.real.dtype),
            np.identity(A.shape[0], dtype=A.dtype),
        )

    else:
        try:
            evals, evecs = la.eigh(A, B, overwrite_a=True, overwrite_b=True)
        except la.LinAlgError as e:
            logger.info(f"Error occurred in eigenvalue solve: {message}")
            # Get error number
            mo = re.search("order (\\d+)", e.args[0])

            # If exception unrecognised then re-raise.
            if mo is None:
                raise e

            errno = mo.group(1)

            if int(errno) < (A.shape[0] + 1):
                logger.info(
                    "Matrix probably not positive definite due to numerical issues. "
                    + "Trying to add a constant diagonal...."
                )

                evb = la.eigvalsh(B)
                add_const = 1e-15 * evb[-1] - 2.0 * evb[0] + 1e-60

                B[np.diag_indices(B.shape[0])] += add_const
                evals, evecs = la.eigh(A, B, overwrite_a=True, overwrite_b=True)

            else:
                logger.info(
                    "Strange convergence issue. Trying non divide and conquer routine."
                )
                evals, evecs = la.eigh(
                    A, B, overwrite_a=True, overwrite_b=True, turbo=False
                )

    return evals, evecs, add_const


def inv_gen(A):
    """Find the inverse of A.

    If a standard matrix inverse has issues try using the pseudo-inverse.

    Parameters
    ----------
    A : np.ndarray
        Matrix to invert.

    Returns
    -------
    inv : np.ndarray
    """
    try:
        inv = la.inv(A)
    except la.LinAlgError:
        inv = la.pinv(A)

    return inv


class KLTransform(config.Reader):
    """Perform KL transform.

    Attributes
    ----------
    subset : boolean
        If True, throw away modes below a S/N `threshold`.
    threshold : scalar
        S/N threshold to cut modes at.
    inverse : boolean
        If True construct and cache inverse transformation.
    use_thermal, use_foregrounds : boolean
        Whether to use instrumental noise/foregrounds (default: both True)
    _foreground_regulariser : scalar
        The regularisation constant for the foregrounds. Adds in a diagonal of
        size reg * cf.max(). Default is 2e-15
    """

    subset = config.Property(proptype=bool, default=True, key="subset")
    inverse = config.Property(proptype=bool, default=False, key="inverse")

    threshold = config.Property(proptype=float, default=0.1, key="threshold")

    _foreground_regulariser = config.Property(
        proptype=float, default=1e-14, key="regulariser"
    )

    use_thermal = config.Property(proptype=bool, default=True)
    use_foregrounds = config.Property(proptype=bool, default=True)
    use_polarised = config.Property(proptype=bool, default=True)

    pol_length = config.Property(proptype=float, default=None)

    evdir = ""

    _cvfg = None
    _cvsg = None

    @property
    def _evfile(self):
        # Pattern to form the `m` ordered file.
        return self.evdir + "/ev_m_" + util.natpattern(self.telescope.mmax) + ".hdf5"

    def __init__(self, bt, subdir=None):
        self.beamtransfer = bt
        self.telescope = self.beamtransfer.telescope

        subdir = "ev" if subdir is None else subdir

        # Create directory if required
        self.evdir = self.beamtransfer.directory + "/" + subdir
        if mpiutil.rank0 and not os.path.exists(self.evdir):
            os.makedirs(self.evdir)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

    def foreground(self):
        """Compute the foreground covariance matrix (on the sky).

        Returns
        -------
        cv_fg : np.ndarray[pol2, pol1, l, freq1, freq2]
        """

        if self._cvfg is None:
            npol = self.telescope.num_pol_sky

            if npol != 1 and npol != 3 and npol != 4:
                raise Exception(
                    "Can only handle unpolarised only (num_pol_sky \
                                 = 1), or I, Q and U (num_pol_sky = 3)."
                )

            # If not polarised then zero out the polarised components of the array
            if self.use_polarised:
                self._cvfg = skymodel.foreground_model(
                    self.telescope.lmax,
                    self.telescope.frequencies,
                    npol,
                    pol_length=self.pol_length,
                )
            else:
                self._cvfg = skymodel.foreground_model(
                    self.telescope.lmax, self.telescope.frequencies, npol, pol_frac=0.0
                )

        return self._cvfg

    def signal(self):
        """Compute the signal covariance matrix (on the sky).

        Returns
        -------
        cv_fg : np.ndarray[pol2, pol1, l, freq1, freq2]
        """

        if self._cvsg is None:
            npol = self.telescope.num_pol_sky

            if npol != 1 and npol != 3 and npol != 4:
                raise Exception(
                    "Can only handle unpolarised only (num_pol_sky \
                                = 1), or I, Q and U (num_pol_sky = 3)."
                )

            self._cvsg = skymodel.im21cm_model(
                self.telescope.lmax, self.telescope.frequencies, npol
            )

        return self._cvsg

    def sn_covariance(self, mi):
        """Compute the signal and noise covariances (on the telescope).

        The signal is formed from the 21cm signal, whereas the noise includes
        both foregrounds and instrumental noise. This is for a single m-mode.

        Parameters
        ----------
        mi : integer
            The m-mode to calculate at.

        Returns
        -------
        s, n : np.ndarray[nfreq, ntel, nfreq, ntel]
            Signal and noice covariance matrices.
        """

        if not (self.use_foregrounds or self.use_thermal):
            raise Exception(
                "Either `use_thermal` or `use_foregrounds`, or both must be True."
            )

        # Project the signal and foregrounds from the sky onto the telescope.
        cvb_s = self.beamtransfer.project_matrix_sky_to_svd(mi, self.signal())

        if self.use_foregrounds:
            cvb_n = self.beamtransfer.project_matrix_sky_to_svd(mi, self.foreground())
        else:
            cvb_n = np.zeros_like(cvb_s)

        # Add in a small diagonal to regularise the noise matrix.
        cnr = cvb_n.reshape((self.beamtransfer.ndof(mi), -1))
        cnr[np.diag_indices_from(cnr)] += self._foreground_regulariser * cnr.max()

        # Even if noise=False, we still want a very small amount of
        # noise, so we multiply by a constant to turn Tsys -> 1 mK.
        nc = 1.0
        if not self.use_thermal:
            nc = (1e-3 / self.telescope.tsys_flat) ** 2

        # Construct diagonal noise power in telescope basis
        bl = np.arange(self.telescope.npairs)
        bl = np.concatenate((bl, bl))
        npower = nc * self.telescope.noisepower(
            bl[np.newaxis, :], np.arange(self.telescope.nfreq)[:, np.newaxis]
        ).reshape(self.telescope.nfreq, self.beamtransfer.ntel)

        # Project into SVD basis and add into noise matrix
        cvb_n += self.beamtransfer.project_matrix_diagonal_telescope_to_svd(mi, npower)

        return cvb_s, cvb_n

    def _transform_m(self, mi):
        """Perform the KL-transform for a single m.

        Parameters
        ----------
        mi : integer
            The m-mode to calculate for.

        Returns
        -------
        evals, evecs : np.ndarray
            The KL-modes. The evals correspond to the diagonal of the
            covariances in the new basis, and the evecs define the basis.
        """

        logger.info("Solving for Eigenvalues....")

        # Fetch the covariance matrices to diagonalise
        st = time.time()
        nside = self.beamtransfer.ndof(mi)

        # Ensure that number of SVD degrees of freedom is non-zero before proceeding
        if nside == 0:
            return np.array([]), np.array([[]]), np.array([[]]), {"ac": 0.0}

        cvb_sr, cvb_nr = [cv.reshape(nside, nside) for cv in self.sn_covariance(mi)]
        et = time.time()
        logger.info(f"Time = {et - st}")

        # Perform the generalised eigenvalue problem to get the KL-modes.
        st = time.time()
        evals, evecs, ac = eigh_gen(cvb_sr, cvb_nr, message=f"m = {mi}")
        et = time.time()
        logger.info(f"Time = {et - st}")

        evecs = evecs.T.conj()

        # Generate inverse if required
        inv = None
        if self.inverse:
            inv = inv_gen(evecs).T

        # Construct dictionary of extra parameters to return
        evextra = {"ac": ac}

        return evals, evecs, inv, evextra

    def transform_save(self, mi):
        """Save the KL-modes for a given m.

        Perform the transform and cache the results for later use.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Results
        -------
        evals, evecs : np.ndarray
            See `transfom_m` for details.
        """

        # Perform the KL-transform
        logger.info(f"Constructing signal and noise covariances for m = {mi} ...")
        evals, evecs, inv, evextra = self._transform_m(mi)

        ## Write out Eigenvals and Vectors

        # Create file and set some metadata
        logger.info(f"Creating file {self._evfile % mi} ....")
        f = h5py.File(self._evfile % mi, "w")
        f.attrs["m"] = mi
        f.attrs["SUBSET"] = self.subset

        ## If modes have been already truncated (e.g. DoubleKL) then pad out
        ## with zeros at the lower end.
        nside = self.beamtransfer.ndof(mi)
        evalsf = np.zeros(nside, dtype=np.float64)
        if evals.size != 0:
            evalsf[(-evals.size) :] = evals
        f.create_dataset("evals_full", data=evalsf)

        # Discard eigenmodes with S/N below threshold if requested.
        if self.subset:
            i_ev = np.searchsorted(evals, self.threshold)

            evals = evals[i_ev:]
            evecs = evecs[i_ev:]
            logger.info(
                "Modes with S/N > %f: %i of %i"
                % (self.threshold, evals.size, evalsf.size)
            )

        # Write out potentially reduced eigen spectrum.
        f.create_dataset("evals", data=evals)
        f.create_dataset("evecs", data=evecs)
        f.attrs["num_modes"] = evals.size

        if self.inverse:
            if self.subset:
                inv = inv[i_ev:]

            f.create_dataset("evinv", data=inv)

        # Call hook which allows derived classes to save special information
        # into the EV file.
        self._ev_save_hook(f, evextra)

        f.close()

        return evals, evecs

    def _ev_save_hook(self, f, evextra):
        ac = evextra["ac"]

        # If we had to regularise because the noise spectrum is numerically ill
        # conditioned, write out the constant we added to the diagonal (see
        # eigh_gen).
        if ac != 0.0:
            f.attrs["add_const"] = ac
            f.attrs["FLAGS"] = "NotPositiveDefinite"
        else:
            f.attrs["FLAGS"] = "Normal"

    def evals_all(self):
        """Collects the full eigenvalue spectrum for all m-modes.

        Reads in from files on disk.

        Returns
        -------
        evarray : np.ndarray
            The full set of eigenvalues across all m-modes.
        """

        f = h5py.File(self.evdir + "/evals.hdf5", "r")
        ev = f["evals"][:]
        f.close()

        return ev

    def _collect(self):
        def evfunc(mi):
            evf = np.zeros(self.beamtransfer.ndofmax)

            f = h5py.File(self._evfile % mi, "r")
            if f["evals_full"].shape[0] > 0:
                ev = f["evals_full"][:]
                evf[-ev.size :] = ev
            f.close()

            return evf

        if mpiutil.rank0:
            logger.info("Creating eigenvalues file (process 0 only).")

        mlist = list(range(self.telescope.mmax + 1))
        shape = (self.beamtransfer.ndofmax,)
        evarray = collect_m_array(mlist, evfunc, shape, np.float64)

        if mpiutil.rank0:
            if os.path.exists(self.evdir + "/evals.hdf5"):
                logger.info(f"File: {self.evdir + '/evals.hdf5'} exists. Skipping...")
                return

            f = h5py.File(self.evdir + "/evals.hdf5", "w")
            f.create_dataset("evals", data=evarray)
            f.close()

    def generate(self, regen=False):
        """Perform the KL-transform for all m-modes and save the result.

        Uses MPI to distribute the work (if available).

        Parameters
        ----------
        mlist : array_like, optional
            Set of m's to calculate KL-modes for By default do all m-modes.
        """

        if mpiutil.rank0:
            st = time.time()
            logger.info("======== Starting KL calculation ========")

        # Iterate list over MPI processes.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):
            if os.path.exists(self._evfile % mi) and not regen:
                logger.info(
                    f"m index {mi}. File: {self._evfile % mi} exists. Skipping..."
                )
                continue

            self.transform_save(mi)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        if mpiutil.rank0:
            et = time.time()
            logger.info(f"======== Ending KL calculation (time={et - st:f}) ========")

        # Collect together the eigenvalues
        self._collect()

    olddatafile = False

    @util.cache_last
    def modes_m(self, mi, threshold=None):
        """Fetch the KL-modes for a particular m.

        This attempts to read in the results from disk, if available and if not
        will create them.

        Also, it will cache the previous m-mode in memory, so as to avoid disk
        access in many cases. However *this* is not sensitive to changes in the
        threshold, be careful.

        Parameters
        ----------
        mi : integer
            m to fetch KL-modes for.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        evals, evecs : np.ndarray
            KL-modes with S/N greater than some threshold. Both evals and evecs
            are potentially `None`, if there are no modes either in the file, or
            satisfying S/N > threshold.
        """

        # If modes not already saved to disk, create file.
        if not os.path.exists(self._evfile % mi):
            modes = self.transform_save(mi)
        else:
            f = h5py.File(self._evfile % mi, "r")

            # If no modes are in the file, return None, None
            if f["evals"].shape[0] == 0:
                modes = None, None
            else:
                # Find modes satisfying threshold (if required).
                evals = f["evals"][:]
                startind = (
                    np.searchsorted(evals, threshold) if threshold is not None else 0
                )

                if startind == evals.size:
                    modes = None, None
                else:
                    modes = (evals[startind:], f["evecs"][startind:])

                    # If old data file perform complex conjugate
                    modes = (
                        modes if not self.olddatafile else (modes[0], modes[1].conj())
                    )
            f.close()

        return modes

    @util.cache_last
    def evals_m(self, mi, threshold=None):
        """Fetch the KL-modes for a particular m.

        This attempts to read in the results from disk, if available and if not
        will create them.

        Also, it will cache the previous m-mode in memory, so as to avoid disk
        access in many cases. However *this* is not sensitive to changes in the
        threshold, be careful.

        Parameters
        ----------
        mi : integer
            m to fetch KL-modes for.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        evals : np.ndarray
            KL-modes with S/N greater than some threshold. Both evals and evecs
            are potentially `None`, if there are no modes either in the file, or
            satisfying S/N > threshold.
        """

        # If modes not already saved to disk, create file.
        if not os.path.exists(self._evfile % mi):
            modes = self.transform_save(mi)
        else:
            f = h5py.File(self._evfile % mi, "r")

            # If no modes are in the file, return None, None
            if f["evals"].shape[0] == 0:
                modes = None
            else:
                # Find modes satisfying threshold (if required).
                evals = f["evals"][:]
                startind = (
                    np.searchsorted(evals, threshold) if threshold is not None else 0
                )

                if startind == evals.size:
                    modes = None
                else:
                    modes = evals[startind:]

            f.close()

        return modes

    @util.cache_last
    def invmodes_m(self, mi, threshold=None):
        """Get the inverse modes.

        If the true inverse has been cached, return the modes for the current
        `threshold`. Otherwise generate the Moore-Penrose pseudo-inverse.

        Parameters
        ----------
        mi : integer
            m-mode to generate for.
        threshold : scalar
            S/N threshold to use.

        Returns
        -------
        invmodes : np.ndarray
        """

        evals = self.evals_m(mi, threshold)

        with h5py.File(self._evfile % mi, "r") as f:
            if "evinv" in f:
                inv = f["evinv"][:]

                if threshold != None:
                    nevals = evals.size
                    inv = inv[(-nevals):]

                return inv.T

            else:
                logger.info("Inverse not cached, generating pseudo-inverse.")
                return la.pinv(self.modes_m(mi, threshold)[1])

    @util.cache_last
    def skymodes_m(self, mi, threshold=None):
        """
        Find the representation of the KL-modes on the sky.

        Use the beamtransfers to rotate the SN-modes onto the sky. This routine
        is based on `modes_m`, as such the same caching and caveats apply.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        skymodes : np.ndarray
            The modes as found in :math:`a_{lm}(\\nu)` space. Note this routine does not
            return the evals.

        See Also
        --------
        :py:func:`modes_m`
        """

        # Fetch the modes in the telescope basis.
        evals, evecs = self.modes_m(mi, threshold=threshold)

        if evals is None:
            raise Exception("Don't seem to be any evals to use.")

        bt = self.beamtransfer

        ## Rotate onto the sky basis. Slightly complex as need to do
        ## frequency-by-frequency
        beam = self.beamtransfer.beam_m(mi).reshape((bt.nfreq, bt.ntel, bt.nsky))
        evecs = evecs.reshape((-1, bt.nfreq, bt.ntel))

        evsky = np.zeros((evecs.shape[0], bt.nfreq, bt.nsky), dtype=np.complex128)

        for fi in range(bt.nfreq):
            evsky[:, fi, :] = np.dot(evecs[:, fi, :], beam[fi])

        return evsky

    def project_vector_svd_to_kl(self, mi, vec, threshold=None):
        """Project a telescope data vector into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Telescope data vector.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        evals, evecs = self.modes_m(mi, threshold)

        if evals is None:
            return np.zeros((0,), dtype=np.complex128)

        if vec.shape[0] != evecs.shape[1]:
            raise Exception("Vectors are incompatible.")

        return np.dot(evecs, vec)

    def project_vector_kl_to_svd(self, mi, vec, threshold=None):
        """Project a vector in the Eigenbasis back into the telescope space.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Eigenbasis data vector.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        evals, evecs = self.modes_m(mi, threshold)

        if evals is None:
            return np.zeros(self.beamtransfer.ndofmax, dtype=np.complex128)

        if vec.shape[0] != evecs.shape[0]:
            raise Exception("Vectors are incompatible.")

        # Construct the pseudo inverse
        invmodes = self.invmodes_m(mi, threshold)

        return np.dot(invmodes, vec)

    def project_vector_sky_to_kl(self, mi, vec, threshold=None):
        """Project an m-vector from the sky into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [freq, pol, l]
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        tvec = self.beamtransfer.project_vector_sky_to_svd(mi, vec)

        return self.project_vector_svd_to_kl(mi, tvec, threshold)

    def project_matrix_svd_to_kl(self, mi, mat, threshold=None):
        """Project a matrix from the telescope basis into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Telescope matrix to project.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projmatrix : np.ndarray
            The matrix projected into the eigenbasis.
        """
        evals, evecs = self.modes_m(mi, threshold)

        if (mat.shape[0] != evecs.shape[1]) or (mat.shape[0] != mat.shape[1]):
            raise Exception("Matrix size incompatible.")

        return np.dot(np.dot(evecs, mat), evecs.T.conj())

    def project_matrix_sky_to_kl(self, mi, mat, threshold=None):
        """Project a covariance matrix from the sky into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix to project.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projmatrix : np.ndarray
            The matrix projected into the eigenbasis.
        """

        mproj = self.beamtransfer.project_matrix_sky_to_svd(mi, mat)

        return self.project_matrix_svd_to_kl(mi, mproj, threshold)

    def project_sky_matrix_forward_old(self, mi, mat, threshold=None):
        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1
        nfreq = self.telescope.nfreq

        st = time.time()

        evsky = self.skymodes_m(mi, threshold).reshape((-1, nfreq, npol, lside))
        et = time.time()

        # print "Evsky: %f" % (et-st)

        st = time.time()
        ev1n = np.transpose(evsky, (2, 3, 0, 1)).copy()
        ev1h = np.transpose(evsky, (2, 3, 1, 0)).conj()
        matf = np.zeros((evsky.shape[0], evsky.shape[0]), dtype=np.complex128)

        for pi in range(npol):
            for pj in range(npol):
                for li in range(lside):
                    matf += np.dot(np.dot(ev1n[pi, li], mat[pi, pj, li]), ev1h[pj, li])

        et = time.time()

        # print "Rest: %f" % (et-st)

        return matf

    def project_sky(self, sky, mlist=None, threshold=None, harmonic=False):
        # Set default list of m-modes (i.e. all of them), and partition
        if mlist is None:
            mlist = list(range(self.telescope.mmax + 1))
        mpart = mpiutil.partition_list_mpi(mlist)

        # Total number of sky modes.
        nmodes = self.beamtransfer.nfreq * self.beamtransfer.ntel

        # If sky is alm fine, if not perform spherical harmonic transform.
        alm = sky if harmonic else hputil.sphtrans_sky(sky, lmax=self.telescope.lmax)

        ## Routine to project sky onto eigenmodes
        def _proj(mi):
            p1 = self.project_sky_vector_forward(mi, alm[:, :, mi], threshold)
            p2 = np.zeros(nmodes, dtype=np.complex128)
            p2[-p1.size :] = p1
            return p2

        # Map over list of m's and project sky onto eigenbasis
        proj_sec = [(mi, _proj(mi)) for mi in mpart]

        # Gather projections onto the rank=0 node.
        proj_all = mpiutil.world.gather(proj_sec, root=0)

        proj_arr = None

        if mpiutil.rank0:
            # Create array to put projections into
            proj_arr = np.zeros(
                (2 * self.telescope.mmax + 1, nmodes), dtype=np.complex128
            )

            # Iterate over all gathered projections and insert into the array
            for proc_rank in proj_all:
                for pm in proc_rank:
                    proj_arr[pm[0]] = pm[1]

        # Return the projections (rank=0) or None elsewhere.
        return proj_arr
