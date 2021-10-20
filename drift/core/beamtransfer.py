"""Calculation and management of Beam Transfer matrices"""

import logging
import pickle
import os
import time
from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import h5py

from caput import config
from caput import misc
from caput import mpiutil
from caput import profile
from caput.truncate import bit_truncate_max_complex

from drift.util import util, blockla
from drift.core import kltransform


# Get the logger object
logger = logging.getLogger(__name__)

try:
    import bitshuffle.h5

    BITSHUFFLE_IMPORTED = True
except ImportError:
    logger.warn("Error importing bitshuffle")
    BITSHUFFLE_IMPORTED = False


def svd_gen(A, errmsg=None, *args, **kwargs):
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
        res = la.svd(A, *args, **kwargs)
    except la.LinAlgError:
        sv = la.svdvals(A)[0]
        At = A + sv * 1e-10 * np.eye(A.shape[0], A.shape[1])
        try:
            res = la.svd(At, *args, **kwargs)
        except la.LinAlgError as e:
            logger.error("Failed completely.", exc_info=e)
            raise e

        if errmsg is None:
            logger.info("Matrix SVD did not converge. Regularised.")
        else:
            logger.warn(f"Matrix SVD did not converge ({errmsg}).")

    return res


def matrix_image(A, rtol=1e-8, atol=None, errmsg=""):
    if A.shape[0] == 0:
        return np.array([], dtype=A.dtype).reshape(0, 0), np.array([], dtype=np.float64)

    try:
        # First try SVD to find matrix image
        u, s, v = la.svd(A, full_matrices=False)

        image, spectrum = u, s

    except la.LinAlgError as e:
        # Try QR with pivoting
        logger.info(f"SVD1 not converged. {errmsg}")

        q, r, p = la.qr(A, pivoting=True, mode="economic")

        try:
            # Try applying QR first, then SVD (this seems to help occasionally)
            u, s, v = la.svd(np.dot(q.T.conj(), A), full_matrices=False)

            image = np.dot(q, u)
            spectrum = s

        except la.LinAlgError as e:
            logger.warn("SVD2 not converged." % errmsg, exc_info=e)

            image = q
            spectrum = np.abs(r.diagonal())

    if atol is None:
        cut = (spectrum > spectrum[0] * rtol).sum()
    else:
        cut = (spectrum > atol).sum()

    image = image[:, :cut].copy()

    return image, spectrum


def matrix_nullspace(A, rtol=1e-8, atol=None, errmsg=""):
    if A.shape[0] == 0:
        return np.array([], dtype=A.dtype).reshape(0, 0), np.array([], dtype=np.float64)

    try:
        # First try SVD to find matrix nullspace
        u, s, v = la.svd(A, full_matrices=True)

        nullspace, spectrum = u, s

    except la.LinAlgError as e:
        # Try QR with pivoting
        logger.info(f"SVD1 not converged. {errmsg}")

        q, r, p = la.qr(A, pivoting=True, mode="full")

        try:
            # Try applying QR first, then SVD (this seems to help occasionally)
            u, s, v = la.svd(np.dot(q.T.conj(), A))

            nullspace = np.dot(q, u)
            spectrum = s

        except la.LinAlgError as e:
            logger.warn(f"SVD2 not converged. {errmsg}", exc_info=e)

            nullspace = q
            spectrum = np.abs(r.diagonal())

    if atol is None:
        cut = (spectrum >= spectrum[0] * rtol).sum()
    else:
        cut = (spectrum >= atol).sum()

    nullspace = nullspace[:, cut:].copy()

    return nullspace, spectrum


class BeamTransfer(config.Reader):
    """A class for reading and writing Beam Transfer matrices from disk.

    In addition this provides methods for projecting vectors and matrices
    between the sky and the telescope basis.

    Parameters
    ----------
    directory : string
        Path of directory to read and write Beam Transfers from.
    telescope : drift.core.telescope.TransitTelescope, optional
        Telescope object to use for calculation. If `None` (default), try to
        load a cached version from the given directory.

    Attributes
    ----------
    mem_chunk : float
        The amount of memory to use per process in this calculation in GB. This is a
        target and not a strict upper limit.  This will change the number of chunks the
        calculation is split into. Default is 3 GB.
    svcut : float
        The relative precision below the maximum singular value to exclude low
        sensitivity SVD modes. This can be dynamically changed as it is evaluated
        when performing projections.
    polsvcut : float
        The relative precision below the maximum value to assume the polarisation
        sensitivity is zero. This is used to find the polarisation null space. This
        is used to generate the cached SVD modes and so cannot be changed after they
        are generated.
    truncate : bool
        Whether precision truncation of the beam transfer matrices should be done.
    truncate_rel : float
        The relative per element precision to use for the truncation.
    truncate_maxl : float
        The truncation precision to use relative the maximum value for all l's of
        that mode.
    chunk_cache_size : int
        The size of the per m-file HDF5 chunk cache. Default is 128 MB.
    """

    mem_chunk = config.Property(proptype=float, default=3.0)

    svcut = config.Property(proptype=float, default=1e-6)
    polsvcut = config.Property(proptype=float, default=1e-4)

    # ====== Beam transfer file options ======
    truncate = config.Property(proptype=bool, default=BITSHUFFLE_IMPORTED)
    truncate_rel = config.Property(proptype=float, default=1e-7)
    truncate_maxl = config.Property(proptype=float, default=1e-8)
    chunk_cache_size = config.Property(proptype=int, default=128)

    # ====== Properties giving internal filenames =======

    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.directory + "/telescopeobject.pickle"

    def _mdir(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self.directory + "/beam_m/" + util.natpattern(self.telescope.mmax)
        return pat % abs(mi)

    def _mfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + "/beam.hdf5"

    def _svdfile(self, mi):
        # Pattern to form the `m` ordered file.

        # Pattern to form the `m` ordered file.
        pat = (
            self.directory
            + "/beam_m/"
            + util.natpattern(self.telescope.mmax)
            + "/svd.hdf5"
        )

        return pat % mi

    # ===================================================

    @property
    def _telescope_pickle(self):
        # The pickled telescope object
        return pickle.dumps(self.telescope)

    def __init__(self, directory, telescope=None):
        self.directory = directory
        self.telescope = telescope

        # Create directory if required
        if mpiutil.rank0 and not os.path.exists(directory):
            os.makedirs(directory)

        mpiutil.barrier()

        if self.telescope is None:
            logger.info("Attempting to read telescope from disk...")

            try:
                with open(self._picklefile, "rb") as f:
                    self.telescope = pickle.load(f)
            except (IOError, pickle.UnpicklingError) as e:
                raise RuntimeError("Could not load Telescope object from disk.") from e

    # ===================================================

    # ====== Loading m-order beams ======================

    @util.cache_last
    def beam_m(self, mi: int, fi: Optional[int] = None) -> np.ndarray:
        """Fetch the beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, 2, npairs, npol_sky, lmax+1)
        """

        nfreq = self.telescope.nfreq
        nbase = self.telescope.nbase
        npol_sky = self.telescope.num_pol_sky
        lmax = self.telescope.lmax

        # Set the properties as if we were selecting a frequency
        ind_list = [
            np.arange(2),
            self.telescope.included_baseline,
            self.telescope.included_pol,
            np.arange(mi, lmax + 1),
        ]
        shape = (2, nbase, npol_sky, lmax + 1)

        # ... and add in the extra axis if we are returning all frequencies
        if fi is None:
            ind_list = [self.telescope.included_freq] + ind_list
            shape = (nfreq,) + shape

        # Allocate the output array
        bf = np.zeros(shape, dtype=np.complex128)

        # Check if we are selecting a single frequency...
        if fi is not None:
            # ... if we are, look up the index in the file
            fi = _find_index_sorted(self.telescope.included_freq, fi)

            # ... and if it's not in the file, just return zeros
            if fi is None:
                return bf

        # Create the broadcasting indices, and then look up the BTM in the file and
        # assign into the correct location in the output array
        ind = np.ix_(*ind_list)
        bf[ind] = _load_beam_f(self._mfile(mi), "beam_m", fi)

        return bf

    # ===================================================

    # ====== Pseudo-inverse beams =======================

    noise_weight = True

    @util.cache_last
    def invbeam_m(self, mi):
        """Pseudo-inverse of the beam (for a given m).

        Uses the Moore-Penrose Pseudo-inverse as the optimal inverse for
        reconstructing the data. No `single` option as this only makes sense
        when combined.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Returns
        -------
        invbeam : np.ndarray (nfreq, npol_sky, lmax+1, 2, npairs)
        """

        beam = self.beam_m(mi)

        if self.noise_weight:
            noisew = self.telescope.noisepower(
                np.arange(self.telescope.npairs), 0
            ).flatten() ** (-0.5)
            beam = beam * noisew[:, np.newaxis, np.newaxis]

        beam = beam.reshape((self.nfreq, self.ntel, self.nsky))

        ibeam = blockla.pinv_dm(beam, rcond=1e-6)

        if self.noise_weight:
            # Reshape to make it easy to multiply baselines by noise level
            ibeam = ibeam.reshape((-1, self.telescope.npairs))
            ibeam = ibeam * noisew

        shape = (
            self.nfreq,
            self.telescope.num_pol_sky,
            self.telescope.lmax + 1,
            self.ntel,
        )

        return ibeam.reshape(shape)

    # ===================================================

    # ====== SVD Beam loading ===========================

    @util.cache_last
    def beam_svd(self, mi: int, fi: Optional[int] = None) -> np.ndarray:
        """Fetch the SVD beam transfer matrix (S V^H) for a given m. This SVD beam
        transfer projects from the sky into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len, npol_sky, lmax+1)
        """

        return _load_beam_f(self._svdfile(mi), "beam_svd", fi)

    @util.cache_last
    def invbeam_svd(self, mi: int, fi: Optional[int] = None) -> np.ndarray:
        """Fetch the SVD beam transfer matrix (S V^H) for a given m. This SVD beam
        transfer projects from the sky into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len, npol_sky, lmax+1)
        """
        return _load_beam_f(self._svdfile(mi), "invbeam_svd", fi)

    @util.cache_last
    def beam_ut(self, mi: int, fi: Optional[int] = None) -> np.ndarray:
        """Fetch the SVD beam transfer matrix (U^H) for a given m. This SVD beam
        transfer projects from the telescope space into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len, ntel)
        """
        return _load_beam_f(self._svdfile(mi), "beam_ut", fi)

    @util.cache_last
    def beam_singularvalues(self, mi: int) -> np.ndarray:
        """Fetch the vector of beam singular values for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len)
        """
        return _load_beam_f(self._svdfile(mi), "singularvalues")

    # ===================================================

    # ====== Generation of all the cache files ==========

    def generate(self, regen=False, skip_svd=False, skip_svd_inv=False):
        """Save out all beam transfer matrices to disk.

        Parameters
        ----------
        regen : boolean, optional
            Force regeneration even if cache files exist (default: False).
        skip_svd : boolen, optional
            Skip SVD beam generation. Saves time and space if you are only map making.
        """

        st = time.time()

        self._generate_dirs()

        # Save pickled telescope object
        if mpiutil.rank0:
            with open(self._picklefile, "wb") as f:
                logger.info("Saving Telescope object.")
                pickle.dump(self.telescope, f)

        with profile.IOUsage(logger=logger):
            self._generate_mfiles(regen)

        if not skip_svd:
            self._generate_svdfiles(regen, skip_svd_inv)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:
            logger.info(f"Beam generation time: {et - st:f}")

    generate_cache = generate  # For compatibility with old code

    def _generate_dirs(self):
        ## Create all the directories required to store the beam transfers.

        if mpiutil.rank0:
            # Create main directory for beamtransfer
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            # Create directories for m beams and svd files.
            for mi in range(self.telescope.mmax + 1):
                dirname = self._mdir(mi)

                # Create directory if required
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

        mpiutil.barrier()

    def _generate_mfiles(self, regen=False):
        if os.path.exists(self.directory + "/beam_m/COMPLETED") and not regen:
            if mpiutil.rank0:
                logger.info("m-files already generated")
            return

        st = time.time()

        # Get the frequencies and baselines that we are going to calculate and create a
        # lookup table for the order in which we are going to calculate them
        freq_to_include = self.telescope.included_freq
        baselines_to_include = self.telescope.included_baseline

        # Calculate the number of included entries
        nf_inc = len(self.telescope.included_freq)
        nb_inc = len(self.telescope.included_baseline)
        np_inc = len(self.telescope.included_pol)
        nl = self.telescope.lmax + 1
        nm = self.telescope.mmax + 1

        nfb = nf_inc * nb_inc

        fbmap = np.array(
            np.meshgrid(freq_to_include, baselines_to_include, indexing="ij")
        ).reshape(2, nfb)

        fbcompact = np.array(
            np.meshgrid(np.arange(nf_inc), np.arange(nb_inc), indexing="ij")
        ).reshape(2, nfb)

        # Calculate the number of baselines to deal with at any one time. Aim
        # to have a maximum of "mem_chunk" GB in memory at any one time
        fbsize = self.telescope.num_pol_sky * nl * 2 * nm * 16.0
        nodemem = self.mem_chunk * 2**30.0

        num_fb_per_node = int(nodemem / fbsize)
        num_fb_per_chunk = num_fb_per_node * mpiutil.size
        num_chunks = int(
            np.ceil(1.0 * nfb / num_fb_per_chunk)
        )  # Number of chunks to break the calculation into

        if mpiutil.rank0:
            logger.info(f"Splitting into {int(num_chunks)} chunks....")

        # The local m sections
        lm, sm, em = mpiutil.split_local(self.telescope.mmax + 1)

        if self.truncate:
            compression_kwargs = {
                "compression": bitshuffle.h5.H5FILTER,
                "compression_opts": (0, bitshuffle.h5.H5_COMPRESS_LZ4),
            }
        else:
            compression_kwargs = {"compression": "lzf"}

        # Iterate over all m's and create the hdf5 files we will write into.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):
            if os.path.exists(self._mfile(mi)) and not regen:
                logger.info(
                    f"m index {mi}. File: {self._mfile(mi)} exists. Skipping..."
                )
                continue

            f = h5py.File(self._mfile(mi), "w")

            dsize = (nf_inc, 2, nb_inc, np_inc, nl - mi)
            csize = (1, 2, min(10, nb_inc), np_inc, nl - mi)

            f.create_dataset(
                "beam_m", dsize, chunks=csize, dtype=np.complex128, **compression_kwargs
            )

            # Write a few useful attributes.
            # f.attrs['baselines'] = self.telescope.baselines
            f.attrs["m"] = mi
            f.attrs["frequencies"] = self.telescope.frequencies

            f.close()

        mpiutil.barrier()

        # Iterate over chunks
        for ci, fbrange in enumerate(mpiutil.split_m(nfb, num_chunks).T):
            if mpiutil.rank0:
                logger.info(f"Starting chunk {int(ci + 1)} of {int(num_chunks)}")

            # Unpack freq-baselines range into num, start and end
            fbnum, fbstart, fbend = fbrange

            # Split the fb list into the ones local to this node
            loc_num, loc_start, loc_end = mpiutil.split_local(fbnum)

            # Get the fb indices for everything in this chunk
            fb_ind_chunk = np.arange(fbstart, fbend)

            # Rotate indices to get a better distribution of work between ranks
            fb_ind_chunk = np.concatenate(
                [fb_ind_chunk[i :: mpiutil.size] for i in range(mpiutil.size)]
            )

            # fb_ind = list(range(fbstart + loc_start, fbstart + loc_end))
            fb_ind = fb_ind_chunk[loc_start:loc_end]

            # Extract the local frequency and baselines indices
            f_ind = fbmap[0, fb_ind]
            bl_ind = fbmap[1, fb_ind]

            # Create array to hold local matrix section
            fb_array = np.zeros((loc_num, 2, np_inc, nl, nm), dtype=np.complex128)

            if loc_num > 0:
                # Calculate the local Beam Matrices
                tarray = self.telescope.transfer_matrices(bl_ind, f_ind)

                # Cut out only the polarisations we are interested in
                tarray = tarray[:, :np_inc]

                # Expensive memory copy into array section
                for mi in range(1, nm):
                    fb_array[:, 0, ..., mi] = tarray[..., mi]
                    fb_array[:, 1, ..., mi] = (-1) ** mi * tarray[..., -mi].conj()

                fb_array[:, 0, ..., 0] = tarray[..., 0]

                del tarray

            if mpiutil.rank0:
                logger.info("Transposing and writing chunk.")

            # Perform an in memory MPI transpose to get the m-ordered array
            m_array = mpiutil.transpose_blocks(fb_array, (fbnum, 2, np_inc, nl, nm))

            del fb_array

            # Transpose to get l as the last axis, this is needed for the (optional)
            # precision truncation
            m_array = m_array.transpose((4, 0, 1, 2, 3)).copy()

            # Truncate the precision of the beam transfers
            if self.truncate:
                bit_truncate_max_complex(
                    m_array.reshape(-1, m_array.shape[-1]),
                    self.truncate_rel,
                    self.truncate_maxl,
                )

            # Write out the current set of chunks into the m-files.
            for lmi, mi in enumerate(range(sm, em)):
                # Open up correct m-file
                with h5py.File(
                    self._mfile(mi), "r+", rdcc_nbytes=(self.chunk_cache_size << 20)
                ) as mfile:
                    # Lookup where to write Beam Transfers and write into file.
                    # Do this in sorted order to try and improve the performance writing
                    # into chunked HDF5 files
                    for fbs in np.argsort(fb_ind_chunk):
                        fbi = fb_ind_chunk[fbs]

                        # Get the indices to write into the compact layout file
                        bci = fbcompact[1, fbi]
                        fci = fbcompact[0, fbi]
                        mfile["beam_m"][fci, :, bci] = m_array[lmi, fbs, ..., mi:]

            del m_array

        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:
            # Make file marker that the m's have been correctly generated:
            open(self.directory + "/beam_m/COMPLETED", "a").close()

            # Print out timing
            logger.info(f"=== MPI transpose took {et - st:f} s ===")

    def _generate_svdfiles(self, regen=False, skip_svd_inv=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        m_list = np.arange(self.telescope.mmax + 1)
        if mpiutil.rank0:
            # For each m, check whether the file exists, if so, whether we
            # can open it. If these tests all pass, we can skip the file.
            # Otherwise, we need to generate a new SVD file for that m.
            for mi in m_list:
                if os.path.exists(self._svdfile(mi)) and not regen:
                    # File may exist but be un-openable, so we catch such an
                    # exception. This shouldn't happen if we use caput.misc.lock_file(),
                    # but we catch it just in case.
                    try:
                        fs = h5py.File(self._svdfile(mi), "r")
                        fs.close()

                        logger.info(
                            f"m index {mi}. Complete file: {self._svdfile(mi)} exists."
                            "Skipping..."
                        )
                        m_list[mi] = -1
                    except Exception:
                        logger.info(
                            f"m index {mi}. ***INCOMPLETE file: {self._svdfile(mi)} "
                            "exists. Will regenerate..."
                        )

            # Reduce m_list to the m's that we need to compute
            m_list = m_list[m_list != -1]

        # Broadcast reduced list to all tasks
        m_list = mpiutil.bcast(m_list)

        # Print m list
        if mpiutil.rank0:
            logger.info(f"m's remaining in beam SVD computation: {m_list}")
        mpiutil.barrier()

        # Distribute m list over tasks, and do computations
        for mi in mpiutil.partition_list_mpi(m_list):
            logger.info(f"m index {mi}. Creating SVD file: {self._svdfile(mi)}")
            self._generate_svdfile_m(mi, skip_svd_inv=skip_svd_inv)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()

    def _generate_svdfile_m(self, mi, skip_svd_inv=False):
        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file.

        # Open file to write SVD results into, using caput.misc.lock_file()
        # to guard against crashes while the file is open. With preserve=True,
        # the temp file will be saved with a period in front of its name
        # if a crash occurs.
        with misc.lock_file(self._svdfile(mi), preserve=True) as fs_lock:
            with h5py.File(fs_lock, "w") as fs:
                # Create a chunked dataset for writing the SVD beam matrix into.
                dsize_bsvd = (
                    self.telescope.nfreq,
                    self.svd_len,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    1,
                    min(10, self.svd_len),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                if not skip_svd_inv:
                    # Create a chunked dataset for writing the inverse SVD beam matrix into.
                    dsize_ibsvd = (
                        self.telescope.nfreq,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        self.svd_len,
                    )
                    csize_ibsvd = (
                        1,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        min(10, self.svd_len),
                    )
                    dset_ibsvd = fs.create_dataset(
                        "invbeam_svd",
                        dsize_ibsvd,
                        chunks=csize_ibsvd,
                        compression="lzf",
                        dtype=np.complex128,
                    )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (self.telescope.nfreq, self.svd_len, self.ntel)
                csize_ut = (1, min(10, self.svd_len), self.ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (self.telescope.nfreq, self.svd_len)
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ## For each frequency in the m-files read in the block, SVD it,
                ## and construct the new beam matrix, and save.
                for fi in np.arange(self.telescope.nfreq):
                    # Read the positive and negative m beams, and combine into one.
                    bf = self.beam_m(mi, fi).reshape(
                        self.ntel,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )

                    noisew = self.telescope.noisepower(
                        np.arange(self.telescope.npairs), fi
                    ).flatten() ** (-0.5)
                    noisew = np.concatenate([noisew, noisew])
                    bf = bf * noisew[:, np.newaxis, np.newaxis]

                    # Reshape total beam to a 2D matrix
                    bfr = bf.reshape(self.ntel, -1)

                    # If unpolarised skip straight to the final SVD, otherwise
                    # project onto the polarised null space.
                    if self.telescope.num_pol_sky == 1:
                        bf2 = bfr
                        ut2 = np.identity(self.ntel, dtype=np.complex128)
                    else:
                        ## SVD 1 - coarse projection onto sky-modes
                        u1, s1 = matrix_image(
                            bfr, rtol=1e-10, errmsg=("SVD1 m=%i f=%i" % (mi, fi))
                        )

                        ut1 = u1.T.conj()
                        bf1 = np.dot(ut1, bfr)

                        ## SVD 2 - project onto polarisation null space
                        bfp = bf1.reshape(
                            bf1.shape[0],
                            self.telescope.num_pol_sky,
                            self.telescope.lmax + 1,
                        )[:, 1:]
                        bfp = bfp.reshape(
                            bf1.shape[0],
                            (self.telescope.num_pol_sky - 1)
                            * (self.telescope.lmax + 1),
                        )
                        u2, s2 = matrix_nullspace(
                            bfp,
                            rtol=self.polsvcut,
                            errmsg=("SVD2 m=%i f=%i" % (mi, fi)),
                        )

                        ut2 = np.dot(u2.T.conj(), ut1)
                        bf2 = np.dot(ut2, bfr)

                    # Check to ensure polcut hasn't thrown away all modes. If it
                    # has, just leave datasets blank.
                    if bf2.shape[0] > 0 and (
                        self.telescope.num_pol_sky == 1 or (s1 > 0.0).any()
                    ):
                        ## SVD 3 - decompose polarisation null space
                        bft = bf2.reshape(
                            -1, self.telescope.num_pol_sky, self.telescope.lmax + 1
                        )[:, 0]

                        u3, s3 = matrix_image(
                            bft, rtol=0.0, errmsg=("SVD3 m=%i f=%i" % (mi, fi))
                        )
                        ut3 = np.dot(u3.T.conj(), ut2)

                        nmodes = ut3.shape[0]

                        # Skip if nmodes is zero for some reason.
                        if nmodes == 0:
                            continue

                        # Final products
                        ut = ut3
                        sig = s3[:nmodes]
                        beam = np.dot(ut3, bfr)

                        # Save out the evecs (for transforming from the telescope frame into the SVD basis)
                        dset_ut[fi, :nmodes] = ut * noisew[np.newaxis, :]

                        # Save out the modified beam matrix (for mapping from the sky into the SVD basis)
                        dset_bsvd[fi, :nmodes] = beam.reshape(
                            nmodes, self.telescope.num_pol_sky, self.telescope.lmax + 1
                        )

                        if not skip_svd_inv:
                            # Find the pseudo-inverse of the beam matrix and save to disk.
                            # First try la.pinv, which uses a least-squares solver.
                            try:
                                ibeam = la.pinv(beam)
                            except la.LinAlgError as e:
                                # If la.pinv fails, try la.pinv2, which is SVD-based and
                                # more likely to succeed. If successful, add file
                                # attribute
                                # indicating pinv2 was used for this frequency.
                                logger.info(
                                    "***Beam-SVD pesudoinverse (scipy.linalg.pinv) "
                                    f"failure: m = {mi}, fi = {fi}. Trying pinv2..."
                                )
                                try:
                                    ibeam = la.pinv2(beam)
                                    if "inv_bsvd_from_pinv2" not in fs.attrs.keys():
                                        fs.attrs["inv_bsvd_from_pinv2"] = [fi]
                                    else:
                                        bad_freqs = fs.attrs["inv_bsvd_from_pinv2"]
                                        fs.attrs["inv_bsvd_from_pinv2"] = (
                                            bad_freqs.append(fi)
                                        )
                                except:
                                    # If pinv2 fails, print error message
                                    raise Exception(
                                        "Beam-SVD pseudoinverse (scipy.linalg.pinv2) failure: m = %d, fi = %d"
                                        % (mi, fi)
                                    )

                            dset_ibsvd[fi, :, :, :nmodes] = ibeam.reshape(
                                self.telescope.num_pol_sky,
                                self.telescope.lmax + 1,
                                nmodes,
                            )

                        # Save out the singular values for each block
                        dset_sig[fi, :nmodes] = sig

                # Write a few useful attributes.
                fs.attrs["baselines"] = self.telescope.baselines
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies

    def _collect_svd_spectrum(self):
        """Gather the SVD spectrum into a single file."""

        svd_func = lambda mi: self.beam_singularvalues(mi)

        svdspectrum = kltransform.collect_m_array(
            list(range(self.telescope.mmax + 1)),
            svd_func,
            (self.nfreq, self.svd_len),
            np.float64,
        )

        if mpiutil.rank0:
            with h5py.File(self.directory + "/svdspectrum.hdf5", "w") as f:
                f.create_dataset("singularvalues", data=svdspectrum)

        mpiutil.barrier()

    def svd_all(self):
        """Collects the full SVD spectrum for all m-modes.

        Reads in from file on disk.

        Returns
        -------
        svarray : np.ndarray[mmax+1, nfreq, svd_len]
            The full set of singular values across all m-modes.
        """

        f = h5py.File(self.directory + "/svdspectrum.hdf5", "r")
        svd = f["singularvalues"][:]
        f.close()

        return svd

    # ===================================================

    # ====== Projection between spaces ==================

    def project_vector_sky_to_telescope(self, mi, vec):
        """Project a vector from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, npol, lmax+1]

        Returns
        -------
        tvec : np.ndarray
            Telescope vector to return, packed as [nfreq, ntel]
        """

        vecf = np.zeros((self.nfreq, 2, self.telescope.nbase), dtype=np.complex128)

        vec = vec
        # Trim the input vector down to the known non-zero BTM entries
        ind = np.ix_(
            self.telescope.included_freq,
            self.telescope.included_pol,
            np.arange(mi, self.telescope.lmax + 1),
        )

        nfreq_trim = len(self.telescope.included_freq)
        nsky_trim = len(self.telescope.included_pol) * (self.telescope.lmax + 1 - mi)
        vec = vec[ind].reshape((nfreq_trim, nsky_trim))

        if np.all(vec == 0):
            return vecf.reshape(self.nfreq, self.ntel)

        with h5py.File(self._mfile(mi), "r") as mfile:
            for file_fi, fi in enumerate(self.telescope.included_freq):
                beamf = mfile["beam_m"][file_fi][:].reshape(-1, nsky_trim)

                t = np.dot(beamf, vec[file_fi]).reshape(2, -1)
                vecf[fi][:, self.telescope.included_baseline] = t

        return vecf.reshape(self.nfreq, self.ntel)

    project_vector_forward = project_vector_sky_to_telescope

    def project_vector_telescope_to_sky(self, mi, vec):
        """Invert a vector from the telescope space onto the sky. This is the
        map-making process.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Telescope data vector, packed as [nfreq, ntel].

        Returns
        -------
        tvec : np.ndarray
            Sky vector to return, packed as [nfreq, npol, lmax+1].
        """

        vecb = np.zeros((self.nfreq, self.nsky), dtype=np.complex128)
        vec = vec.reshape((self.nfreq, self.ntel))

        if np.all(vec == 0):
            return vecb.reshape(
                (self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
            )

        ibeam = self.invbeam_m(mi).reshape((self.nfreq, self.nsky, self.ntel))

        for fi in range(self.nfreq):
            vecb[fi] = np.dot(ibeam[fi], vec[fi, :].reshape(self.ntel))

        return vecb.reshape(
            (self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
        )

    project_vector_backward = project_vector_telescope_to_sky

    def project_vector_backward_dirty(self, mi, vec):
        vecb = np.zeros((self.nfreq, self.nsky), dtype=np.complex128)
        vec = vec.reshape((self.nfreq, self.ntel))

        if np.all(vec == 0):
            return vecb.reshape(
                (self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
            )

        dbeam = self.beam_m(mi).reshape((self.nfreq, self.ntel, self.nsky))
        dbeam = dbeam.transpose((0, 2, 1)).conj()

        for fi in range(self.nfreq):
            norm = np.dot(dbeam[fi].T.conj(), dbeam[fi]).diagonal()
            norm = np.where(norm < 1e-6, 0.0, 1.0 / norm)
            # norm = np.dot(dbeam[fi], dbeam[fi].T.conj()).diagonal()
            # norm = np.where(np.logical_or(np.abs(norm) < 1e-4,
            # np.abs(norm) < np.abs(norm.max()*1e-2)), 0.0, 1.0 / norm)
            vecb[fi] = np.dot(dbeam[fi], vec[fi, :].reshape(self.ntel) * norm)

        return vecb.reshape(
            (self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
        )

    def project_matrix_sky_to_telescope(self, mi, mat, temponly=False):
        """Project a covariance matrix from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [pol, pol, l, freq, freq]
        temponly: boolean
            Force projection of temperature (TT) part only (default: False)

        Returns
        -------
        tmat : np.ndarray
            Covariance in telescope basis, packed as [nfreq, ntel, nfreq, ntel].
        """
        npol = 1 if temponly else self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1

        beam = self.beam_m(mi).reshape(
            (self.nfreq, self.ntel, self.telescope.num_pol_sky, lside)
        )

        matf = np.zeros(
            (self.nfreq, self.ntel, self.nfreq, self.ntel), dtype=np.complex128
        )

        # Should it be a +=?
        for pi in range(npol):
            for pj in range(npol):
                for fi in range(self.nfreq):
                    for fj in range(self.nfreq):
                        matf[fi, :, fj, :] += np.dot(
                            (beam[fi, :, pi, :] * mat[pi, pj, :, fi, fj]),
                            beam[fj, :, pj, :].T.conj(),
                        )

        return matf

    project_matrix_forward = project_matrix_sky_to_telescope

    def _svd_num(self, mi):
        ## Calculate the number of SVD modes meeting the cut for each
        ## frequency, return the number and the array bounds

        # Get the array of singular values for each mode
        sv = self.beam_singularvalues(mi)

        # Number of significant sv modes at each frequency
        svnum = (sv > sv.max() * self.svcut).sum(axis=1)

        # Calculate the block bounds within the full matrix
        svbounds = np.cumsum(np.insert(svnum, 0, 0))

        return svnum, svbounds

    def _svd_freq_iter(self, mi):
        num = self._svd_num(mi)[0]
        return [fi for fi in range(self.nfreq) if (num[fi] > 0)]

    def project_matrix_sky_to_svd(self, mi, mat, temponly=False):
        """Project a covariance matrix from the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [pol, pol, l, freq, freq]. Must have pol
            indices even if `temponly=True`.
        temponly: boolean
            Force projection of temperature (TT) part only (default: False)

        Returns
        -------
        tmat : np.ndarray [nsvd, nsvd]
            Covariance in SVD basis.
        """

        npol = 1 if temponly else self.telescope.num_pol_sky

        lside = self.telescope.lmax + 1

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Create the output matrix
        matf = np.zeros((svbounds[-1], svbounds[-1]), dtype=np.complex128)

        # Should it be a +=?
        for pi in range(npol):
            for pj in range(npol):
                for fi in self._svd_freq_iter(mi):
                    fibeam = beam[
                        fi, : svnum[fi], pi, :
                    ]  # Beam for this pol, freq, and svcut (i)

                    for fj in self._svd_freq_iter(mi):
                        fjbeam = beam[
                            fj, : svnum[fj], pj, :
                        ]  # Beam for this pol, freq, and svcut (j)
                        lmat = mat[
                            pi, pj, :, fi, fj
                        ]  # Local section of the sky matrix (i.e C_l part)

                        matf[
                            svbounds[fi] : svbounds[fi + 1],
                            svbounds[fj] : svbounds[fj + 1],
                        ] += np.dot(fibeam * lmat, fjbeam.T.conj())

        return matf

    def project_matrix_diagonal_telescope_to_svd(self, mi, dmat):
        """Project a diagonal matrix from the telescope basis into the SVD basis.

        This slightly specialised routine is for projecting the noise
        covariance into the SVD space.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        dmat : np.ndarray
            Sky matrix packed as [nfreq, ntel]

        Returns
        -------
        tmat : np.ndarray [nsvd, nsvd]
            Covariance in SVD basis.
        """

        svdfile = h5py.File(self._svdfile(mi), "r")

        # Get the SVD beam matrix
        beam = svdfile["beam_ut"]

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Create the output matrix
        matf = np.zeros((svbounds[-1], svbounds[-1]), dtype=np.complex128)

        # Should it be a +=?
        for fi in self._svd_freq_iter(mi):
            fbeam = beam[fi, : svnum[fi], :]  # Beam matrix for this frequency and cut
            lmat = dmat[fi, :]  # Matrix section for this frequency

            matf[svbounds[fi] : svbounds[fi + 1], svbounds[fi] : svbounds[fi + 1]] = (
                np.dot((fbeam * lmat), fbeam.T.conj())
            )

        svdfile.close()

        return matf

    def project_vector_telescope_to_svd(self, mi, vec):
        """Map a vector from the telescope space into the SVD basis.

        This projection may be lose information about the sky, depending on
        the polarisation filtering.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Telescope data vector packed as [freq, baseline, polarisation]

        Returns
        -------
        svec : np.ndarray[svdnum]
            SVD vector to return.
        """

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Create the output matrix (shape is calculated from input shape)
        vecf = np.zeros((svbounds[-1],) + vec.shape[2:], dtype=np.complex128)

        if np.all(vec == 0):
            return vecf

        # Get the SVD beam matrix
        beam = self.beam_ut(mi)

        # Should it be a +=?
        for fi in self._svd_freq_iter(mi):
            fbeam = beam[fi, : svnum[fi], :]  # Beam matrix for this frequency and cut
            lvec = vec[fi, :]  # Matrix section for this frequency

            vecf[svbounds[fi] : svbounds[fi + 1]] = np.dot(fbeam, lvec)

        return vecf

    def project_vector_svd_to_telescope(self, mi, svec):
        """Map a vector from the SVD basis into the original data basis.

        This projection may be lose information about the sky, depending on the
        polarisation filtering. This essentially uses the pseudo-inverse which
        is simply related to the original projection matrix.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        svec : np.ndarray
            SVD data vector.

        Returns
        -------
        vec : np.ndarray[freq, sign, baseline]
            Data vector to return.
        """

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Create the output matrix (shape is calculated from input shape)
        vecf = np.zeros((self.nfreq, self.ntel), dtype=np.complex128)

        if np.all(svec == 0):
            return vecf.reshape(self.nfreq, 2, self.telescope.npairs)

        # Get the SVD beam matrix
        beam = self.beam_ut(mi)

        # Should it be a +=?
        for fi in self._svd_freq_iter(mi):
            noise = self.telescope.noisepower(
                np.arange(self.telescope.npairs), fi
            ).flatten()
            noise = np.concatenate([noise, noise])

            fbeam = beam[fi, : svnum[fi], :]  # Beam matrix for this frequency and cut
            lvec = svec[
                svbounds[fi] : svbounds[fi + 1]
            ]  # Matrix section for this frequency

            # As the form of the forward projection is simply a scaling and then
            # projection onto an orthonormal basis, the pseudo-inverse is simply
            # related.
            vecf[fi, :] = noise * np.dot(fbeam.T.conj(), lvec)

        return vecf.reshape(self.nfreq, 2, self.telescope.npairs)

    def project_vector_sky_to_svd(self, mi, vec, temponly=False):
        """Project a vector from the the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, lmax+1]
        temponly: boolean
            Force projection of temperature part only (default: False)

        Returns
        -------
        svec : np.ndarray
            SVD vector to return.
        """
        npol = 1 if temponly else self.telescope.num_pol_sky

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Create the output matrix
        vecf = np.zeros((svbounds[-1],) + vec.shape[3:], dtype=np.complex128)

        if np.all(vec == 0):
            return vecf

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)

        for pi in range(npol):
            for fi in self._svd_freq_iter(mi):
                fbeam = beam[
                    fi, : svnum[fi], pi, :
                ]  # Beam matrix for this frequency and cut
                lvec = vec[fi, pi]  # Matrix section for this frequency

                vecf[svbounds[fi] : svbounds[fi + 1]] += np.dot(fbeam, lvec)

        return vecf

    def project_vector_svd_to_sky(self, mi, vec, temponly=False, conj=False):
        """Project a vector from the the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, lmax+1]
        temponly: boolean
            Force projection of temperature part only (default: False)
        conj: boolean
            Reverse projection by applying conjugation (as opposed to pseudo-
            inverse). Default is False.

        Returns
        -------
        svec : np.ndarray
            SVD vector to return.
        """
        npol = 1 if temponly else self.telescope.num_pol_sky

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Create the output matrix
        vecf = np.zeros(
            (self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
            + vec.shape[1:],
            dtype=np.complex128,
        )

        if np.all(vec == 0):
            return vecf

        # Get the SVD beam matrix
        beam = self.beam_svd(mi) if conj else self.invbeam_svd(mi)

        for pi in range(npol):
            for fi in self._svd_freq_iter(mi):
                if conj:
                    fbeam = beam[
                        fi, : svnum[fi], pi, :
                    ].T.conj()  # Beam matrix for this frequency and cut
                else:
                    fbeam = beam[
                        fi, pi, :, : svnum[fi]
                    ]  # Beam matrix for this frequency and cut

                lvec = vec[
                    svbounds[fi] : svbounds[fi + 1]
                ]  # Matrix section for this frequency

                vecf[fi, pi] += np.dot(fbeam, lvec)

        return vecf

    # ===================================================

    # ====== Dimensionality of the various spaces =======

    @property
    def ntel(self):
        """Degrees of freedom measured by the telescope (per frequency)"""
        return 2 * self.telescope.npairs

    @property
    def nsky(self):
        """Degrees of freedom on the sky at each frequency."""
        return (self.telescope.lmax + 1) * self.telescope.num_pol_sky

    @property
    def nfreq(self):
        """Number of frequencies measured."""
        return self.telescope.nfreq

    @property
    def svd_len(self):
        """The size of the SVD output matrices."""
        return min(self.telescope.lmax + 1, self.ntel)

    @property
    def ndofmax(self):
        return self.svd_len * self.nfreq

    def ndof(self, mi):
        """The number of degrees of freedom at a given m."""
        return self._svd_num(mi)[1][-1]

    # ===================================================


class BeamTransferTempSVD(BeamTransfer):
    """BeamTransfer class that performs the old temperature only SVD."""

    def _generate_svdfiles(self, regen=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):
            if os.path.exists(self._svdfile(mi)) and not regen:
                logger.info(
                    f"m index {mi}. File: {self._svdfile(mi)} exists. Skipping..."
                )
                continue
            else:
                logger.info(f"m index {mi}. Creating SVD file: {self._svdfile(mi)}")

            # Open file to write SVD results into.
            with h5py.File(self._svdfile(mi), "w") as fs:
                # Create a chunked dataset for writing the SVD beam matrix into.
                dsize_bsvd = (
                    self.telescope.nfreq,
                    self.svd_len,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    1,
                    min(10, self.svd_len),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a chunked dataset for writing the inverse SVD beam matrix into.
                dsize_ibsvd = (
                    self.telescope.nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    self.svd_len,
                )
                csize_ibsvd = (
                    1,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    min(10, self.svd_len),
                )
                dset_ibsvd = fs.create_dataset(
                    "invbeam_svd",
                    dsize_ibsvd,
                    chunks=csize_ibsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (self.telescope.nfreq, self.svd_len, self.ntel)
                csize_ut = (1, min(10, self.svd_len), self.ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (self.telescope.nfreq, self.svd_len)
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ## For each frequency in the m-files read in the block, SVD it,
                ## and construct the new beam matrix, and save.
                for fi in np.arange(self.telescope.nfreq):
                    # Read the positive and negative m beams, and combine into one.
                    bf = self.beam_m(mi, fi).reshape(
                        self.ntel,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )

                    noisew = self.telescope.noisepower(
                        np.arange(self.telescope.npairs), fi
                    ).flatten() ** (-0.5)
                    noisew = np.concatenate([noisew, noisew])
                    bf = bf * noisew[:, np.newaxis, np.newaxis]

                    # Get the T-mode only beam matrix
                    bft = bf[:, 0, :]

                    # Perform the SVD to find the left evecs
                    u, sig, v = svd_gen(bft, full_matrices=False)
                    u = u.T.conj()  # We only need u^H so just keep that.

                    # Save out the evecs (for transforming from the telescope frame into the SVD basis)
                    dset_ut[fi] = u * noisew[np.newaxis, :]

                    # Save out the modified beam matrix (for mapping from the sky into the SVD basis)
                    bsvd = np.dot(u, bf.reshape(self.ntel, -1))
                    dset_bsvd[fi] = bsvd.reshape(
                        self.svd_len,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )

                    # Find the pseudo-inverse of the beam matrix and save to disk.
                    dset_ibsvd[fi] = la.pinv(bsvd).reshape(
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        self.svd_len,
                    )

                    # Save out the singular values for each block
                    dset_sig[fi] = sig

                # Write a few useful attributes.
                fs.attrs["baselines"] = self.telescope.baselines
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies
                fs.attrs["cylobj"] = self._telescope_pickle

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()


class BeamTransferFullSVD(BeamTransfer):
    """BeamTransfer class that performs the old temperature only SVD."""

    def _generate_svdfiles(self, regen=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):
            if os.path.exists(self._svdfile(mi)) and not regen:
                logger.info(
                    f"m index {mi}. File: {self._svdfile(mi)} exists. Skipping..."
                )
                continue
            else:
                logger.info(f"m index {mi}. Creating SVD file: {self._svdfile(mi)}")

            # Open file to write SVD results into
            with h5py.File(self._svdfile(mi), "w") as fs:
                # Create a chunked dataset for writing the SVD beam matrix into.
                dsize_bsvd = (
                    self.telescope.nfreq,
                    self.svd_len,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    1,
                    min(10, self.svd_len),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a chunked dataset for writing the inverse SVD beam matrix into.
                dsize_ibsvd = (
                    self.telescope.nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    self.svd_len,
                )
                csize_ibsvd = (
                    1,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    min(10, self.svd_len),
                )
                dset_ibsvd = fs.create_dataset(
                    "invbeam_svd",
                    dsize_ibsvd,
                    chunks=csize_ibsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (self.telescope.nfreq, self.svd_len, self.ntel)
                csize_ut = (1, min(10, self.svd_len), self.ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (self.telescope.nfreq, self.svd_len)
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ## For each frequency in the m-files read in the block, SVD it,
                ## and construct the new beam matrix, and save.
                for fi in np.arange(self.telescope.nfreq):
                    # Read the positive and negative m beams, and combine into one.
                    bf = self.beam_m(mi, fi).reshape(
                        self.ntel,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )

                    noisew = self.telescope.noisepower(
                        np.arange(self.telescope.npairs), fi
                    ).flatten() ** (-0.5)
                    noisew = np.concatenate([noisew, noisew])
                    bf = bf * noisew[:, np.newaxis, np.newaxis]

                    bf = bf.reshape(self.ntel, -1)

                    # Perform the SVD to find the left evecs
                    u, sig, v = svd_gen(bf, full_matrices=False)
                    u = u.T.conj()  # We only need u^H so just keep that.

                    # Save out the evecs (for transforming from the telescope frame into the SVD basis)
                    dset_ut[fi] = u * noisew[np.newaxis, :]

                    # Save out the modified beam matrix (for mapping from the sky into the SVD basis)
                    bsvd = np.dot(u, bf)
                    dset_bsvd[fi] = bsvd.reshape(
                        self.svd_len,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )

                    # Find the pseudo-inverse of the beam matrix and save to disk.
                    dset_ibsvd[fi] = la.pinv(bsvd).reshape(
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        self.svd_len,
                    )

                    # Save out the singular values for each block
                    dset_sig[fi] = sig

                # Write a few useful attributes.
                fs.attrs["baselines"] = self.telescope.baselines
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies
                fs.attrs["cylobj"] = self._telescope_pickle

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()

    @property
    def svd_len(self):
        """The size of the SVD output matrices."""
        return min((self.telescope.lmax + 1) * self.telescope.num_pol_sky, self.ntel)


class BeamTransferNoSVD(BeamTransfer):
    """Subclass of BeamTransfer that skips SVD decomposition.

    To use in a driftscan config file, the `nosvd` flag needs
    to be set. (In this case, the value of `skip_svd` is ignored,
    because the SVD step is automatically skipped.)
    """

    # No SV cut
    svcut = 0.0

    noise_weight = False

    def _svd_num(self, mi):
        """Compute number of SVD modes meeting the cut.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.

        Returns
        -------
        svnum : np.ndarray
            Number of remaining SV modes at each frequency.
        svbounds : np.ndarray
            Indices bounding SV modes after concatenation over frequency.
            Has size nfreq+1.
        """
        # Number of significant SV modes at each frequency,
        # which is *all* SV modes in no-SVD case, since SV modes
        # are just the m-modes.
        svnum = (np.ones(self.nfreq) * self.ntel).astype(int)

        # Calculate the block bounds within the full matrix
        svbounds = np.cumsum(np.insert(svnum, 0, 0))

        return svnum, svbounds

    def _generate_svdfiles(self, regen=False, skip_svd_inv=False):
        print("======== Skipping telescope SVD step ========")

    def project_matrix_sky_to_svd(self, mi, mat, temponly=False):
        """Project a covariance matrix from the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [pol, pol, l, freq, freq].
        temponly: boolean
            Force projection of temperature (TT) part only (default: False)

        Returns
        -------
        tmat : np.ndarray
            SVD-basis matrix, packed as [ndof, ndof]. Recall that
            ndof = ntel * nfreq = 2 * nbase * nfreq.
        """
        return self.project_matrix_sky_to_telescope(mi, mat, temponly=temponly).reshape(
            self.ndof(mi), self.ndof(mi)
        )

    def project_vector_sky_to_svd(self, mi, vec, *args, **kwargs):
        """Project a vector from the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, npol, lmax+1]

        Returns
        -------
        tvec : np.ndarray
            Telescope vector to return, packed as [ndof].
        """
        return self.project_vector_sky_to_telescope(mi, vec).flatten()

    def project_matrix_telescope_to_svd(self, mi, mat):
        """Map a matrix from the telescope space into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Telescope-basis matrix, packed as [freq, baseline, freq, baseline].

        Returns
        -------
        out_mat : np.ndarray
            SVD-basis matrix, packed as [ndof, ndof].
        """
        return mat.reshape(self.ndof(mi), self.ndof(mi))

    def project_matrix_diagonal_telescope_to_svd(self, mi, dmat, *args, **kwargs):
        """Project a diagonal matrix from the telescope basis to the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        dmat : np.ndarray
            Diagonal telescope-basis matrix packed as [nfreq, ntel].

        Returns
        -------
        tvec : np.ndarray
            SVD-basis matrix to return, packed as [ndof, ndof].
        """
        return np.diag(dmat.flatten())

    def project_vector_telescope_to_svd(self, mi, vec, *args, **kwargs):
        """Project a vector from the telescope basis into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Telescope data vector packed as [nfreq, ntel].

        Returns
        -------
        tvec : np.ndarray
            SVD vector to return, packed as [ndof].
        """
        return vec.flatten()

    def project_vector_svd_to_sky(self, mi, vec, temponly=False, conj=False):
        """Project a vector from the the SVD basis into the sky basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            SVD vector, packed as [ndof, ...]

        Returns
        -------
        svec : np.ndarray
            Sky vector to return, packed as [nfreq, npol, lmax+1, ...]
        """

        if temponly:
            raise NotImplementedError(
                "temponly not implemented for no-SVD project_vector_svd_to_sky!"
            )

        # Create the output matrix
        svec = np.zeros(
            (self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
            + vec.shape[1:],
            dtype=np.complex128,
        )

        # Get inverse or Hermitian conjugate of beam matrix
        if conj:
            beam = self.beam_m(mi).reshape((self.nfreq, self.ntel, self.nsky))

            # Loop through frequencies, doing tel-to-sky projection at each freq
            for fi in range(self.nfreq):
                svec[fi] = np.dot(
                    beam[fi].T.conj(), vec.reshape(self.nfreq, self.ntel, -1)[fi]
                ).reshape(
                    (self.telescope.num_pol_sky, self.telescope.lmax + 1)
                    + vec.shape[1:]
                )

        else:
            ibeam = self.invbeam_m(mi).reshape((self.nfreq, self.nsky, self.ntel))

            # Loop through frequencies, doing tel-to-sky projection at each freq
            for fi in range(self.nfreq):
                svec[fi] = np.dot(
                    ibeam[fi], vec.reshape(self.nfreq, self.ntel, -1)[fi]
                ).reshape(
                    (self.telescope.num_pol_sky, self.telescope.lmax + 1)
                    + vec.shape[1:]
                )

        return svec

    def beam_svd(self, mi, *args, **kwargs):
        """Fetch beam SVD matrix.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.

        Returns
        -------
        mat : np.ndarray
            Beam-SVD matrix, which in this class is just the beam transfer
            matrix, packed as [nfreq, 2, nbase, npol, lmax+1].
        """
        return self.beam_m(mi)

    def ndof(self, mi, *args, **kwargs):
        """Compute number of degrees of freedom in telescope basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.

        Returns
        -------
        n : int
            N_dof = ntel * nfreq = nbase * 2 * nfreq.
        """
        return self.ntel * self.nfreq

    @property
    def ndofmax(self):
        """Compute number of degrees of freedom in telescope basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.

        Returns
        -------
        n : int
            N_dof = ntel * nfreq = nbase * 2 * nfreq.
        """
        return self.ntel * self.nfreq


Index1D = Union[int, slice]
IndexND = Union[Index1D, Tuple[Index1D, ...]]


def _load_beam_f(
    path: os.PathLike, dset_name: str, ind: Optional[IndexND] = None
) -> np.ndarray:
    # Load a beam from a file with the appropriate type checking

    # Use a full slice if ind is None
    ind = ind if ind is not None else slice(None)

    with h5py.File(path, "r") as fh:
        dset = fh[dset_name]

        if not isinstance(dset, h5py.Dataset):
            raise RuntimeError(f"Malformed beam file: {path}")

        # If ind is None, return the full entry, to frequency blocks. Otherwise just the one requested.
        beam = dset[ind]

    # Check that we have got out a valid beam array
    assert isinstance(beam, np.ndarray)

    return beam


def _find_index_sorted(a: np.ndarray, v: int) -> Optional[int]:
    """Find the index of the first entry in `a` equal to `v`."""
    ind = np.searchsorted(a, v)

    return ind if v == a[ind] else None
