"""
========================================================
Beam Transfer Matrices (:mod:`~drift.core.beamtransfer`)
========================================================

A class for calculating and managing Beam Transfer matrices

Classes
=======

.. autosummary::
    :toctree: generated/

    BeamTransfer

"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pickle
import os
import time

import numpy as np
import scipy.linalg as la
import h5py
from mpi4py import MPI

from caput import mpiutil

from drift.util import util, blockla
from drift.core import kltransform
from draco.analysis.svdfilter import external_svdfile


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
            print("Failed completely. %s" % errmsg)
            raise e

        if errmsg is None:
            print("Matrix SVD did not converge. Regularised.")
        else:
            print("Matrix SVD did not converge (%s)." % errmsg)

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
        print("SVD1 not converged. %s" % errmsg)

        q, r, p = la.qr(A, pivoting=True, mode="economic")

        try:
            # Try applying QR first, then SVD (this seems to help occasionally)
            u, s, v = la.svd(np.dot(q.T.conj(), A), full_matrices=False)

            image = np.dot(q, u)
            spectrum = s

        except la.LinAlgError as e:
            print("SVD2 not converged. %s" % errmsg)

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
        print("SVD1 not converged. %s" % errmsg)

        q, r, p = la.qr(A, pivoting=True, mode="full")

        try:
            # Try applying QR first, then SVD (this seems to help occasionally)
            u, s, v = la.svd(np.dot(q.T.conj(), A))

            nullspace = np.dot(q, u)
            spectrum = s

        except la.LinAlgError as e:
            print("SVD2 not converged. %s" % errmsg)

            nullspace = q
            spectrum = np.abs(r.diagonal())

    if atol is None:
        cut = (spectrum >= spectrum[0] * rtol).sum()
    else:
        cut = (spectrum >= atol).sum()

    nullspace = nullspace[:, cut:].copy()

    return nullspace, spectrum


class BeamTransfer(object):
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
    svcut
    polsvcut
    ntel
    nsky
    nfreq
    svd_len
    ndofmax
    external_svd_basis_dir
    external_svthreshold_global
    external_svthreshold_local
    external_sv_mode_cut
    external_global_max_sv
    prewhiten_with_ext_svd_projection

    Methods
    -------
    ndof
    beam_m
    invbeam_m
    beam_svd
    beam_ut
    invbeam_svd
    beam_singularvalues
    generate
    project_vector_sky_to_telescope
    project_vector_telescope_to_sky
    project_vector_sky_to_svd
    project_vector_svd_to_sky
    project_vector_telescope_to_svd
    project_matrix_sky_to_telescope
    project_matrix_sky_to_svd
    """

    _mem_switch = 2.0  # Rough chunks (in GB) to divide calculation into.

    svcut = 1e-6
    polsvcut = 1e-4

    # Directory containing files defining external SVD basis (determined
    # directly from measured visibilities, using draco.analysis.svdfilter.SVDFilter)
    external_svd_basis_dir = None

    # Thresholds for filtering modes defined in external SVD basis:
    #  global ->    Remove modes with singular value higher than external_svthreshold_global
    #               times the largest mode on any m
    #  local  ->    Remove modes with singular value higher than external_svthreshold_local
    #               times the largest mode on each m
    # Default values are such that no modes are filtered out - user must specify something
    # for filtering to take place!
    external_svthreshold_global = 1000.
    external_svthreshold_local = 1000.
    external_sv_mode_cut = None

    # If using an external SVD basis, this controls whether the beam transfer
    # matrices should be prewhitened using a noise matrix that also has the
    # ext-SVD projection applied. False by default, because doing this is actually
    # nontrivial, due to the possibility that the noise matrix because
    # non-positive-definite after ext-SVD projection
    prewhiten_with_ext_svd_projection = False

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

    def _fdir(self, fi):
        # Pattern to form the `freq` ordered file.
        pat = self.directory + "/beam_f/" + util.natpattern(self.telescope.nfreq)
        return pat % fi

    def _ffile(self, fi):
        # Pattern to form the `freq` ordered file.
        return self._fdir(fi) + "/beam.hdf5"

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

    def _external_svdfile(self, mi):
        """File containing external SVD basis for a given m.
        """
        if self.external_svd_basis_dir is None:
            raise RuntimeError("Directory containing external SVD basis not specified!")
        else:
            return external_svdfile(self.external_svd_basis_dir, mi, self.telescope.mmax)

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
            print("Attempting to read telescope from disk...")

            try:
                with open(self._picklefile, "rb") as f:
                    self.telescope = pickle.load(f)
            except (IOError, pickle.UnpicklingError):
                raise Exception("Could not load Telescope object from disk.")

    # ===================================================

    # ====== Loading m-order beams ======================

    def _load_beam_m(self, mi, fi=None):
        ## Read in beam from disk
        mfile = h5py.File(self._mfile(mi), "r")

        # If fi is None, return all frequency blocks. Otherwise just the one requested.
        if fi is None:
            beam = mfile["beam_m"][:]
        else:
            beam = mfile["beam_m"][fi][:]

        mfile.close()

        return beam

    @util.cache_last
    def beam_m(self, mi, fi=None):
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

        return self._load_beam_m(mi, fi=fi)

    # ===================================================

    # ====== Loading freq-ordered beams =================

    @util.cache_last
    def _load_beam_freq(self, fi, fullm=False):

        tel = self.telescope
        mside = 2 * tel.lmax + 1 if fullm else 2 * tel.mmax + 1

        ffile = h5py.File(self._ffile(fi), "r")
        beamf = ffile["beam_freq"][:]
        ffile.close()

        if fullm:
            beamt = np.zeros(
                beamf.shape[:-1] + (2 * tel.lmax + 1,), dtype=np.complex128
            )

            for mi in range(-tel.mmax, tel.mmax + 1):
                beamt[..., mi] = beamf[..., mi]

            beamf = beamt

        return beamf

    @util.cache_last
    def beam_freq(self, fi, fullm=False, single=False):
        """Fetch the beam transfer matrix for a given frequency.

        Parameters
        ----------
        fi : integer
            Frequency to fetch.
        fullm : boolean, optional
            Pad out m-modes such that we have :math:`mmax = 2*lmax-1`. Useful
            for projecting around a_lm's. Default is False.
        single : boolean, optional
            When set, fetch only the uncombined beam transfers (that is only
            positive or negative m). Default is False.

        Returns
        -------
        beam : np.ndarray
        """
        bf = self._load_beam_freq(fi, fullm)

        if single:
            return bf

        mside = (bf.shape[-1] + 1) // 2

        bfc = np.zeros((mside, 2) + bf.shape[:-1], dtype=bf.dtype)

        bfc[0, 0] = bf[..., 0]

        for mi in range(1, mside):
            bfc[mi, 0] = bf[..., mi]
            bfc[mi, 1] = (-1) ** mi * bf[..., -mi].conj()

        return bfc

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
    def beam_svd(self, mi, fi=None):
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

        svdfile = h5py.File(self._svdfile(mi), "r")

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            bs = svdfile["beam_svd"][:]
        else:
            bs = svdfile["beam_svd"][fi][:]

        svdfile.close()

        return bs

    @util.cache_last
    def invbeam_svd(self, mi, fi=None):
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

        svdfile = h5py.File(self._svdfile(mi), "r")

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            ibs = svdfile["invbeam_svd"][:]
        else:
            ibs = svdfile["invbeam_svd"][fi][:]

        svdfile.close()

        return ibs

    @util.cache_last
    def beam_ut(self, mi, fi=None):
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

        svdfile = h5py.File(self._svdfile(mi), "r")

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            bs = svdfile["beam_ut"][:]
        else:
            bs = svdfile["beam_ut"][fi][:]

        svdfile.close()

        return bs

    @util.cache_last
    def beam_singularvalues(self, mi):
        """Fetch the vector of beam singular values for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len)
        """

        svdfile = h5py.File(self._svdfile(mi), "r")
        sv = svdfile["singularvalues"][:]
        svdfile.close()

        return sv

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
                print("=== Saving Telescope object. ===")
                pickle.dump(self.telescope, f)

        self._generate_mfiles(regen)

        if not skip_svd:
            self._generate_svdfiles(regen, skip_svd_inv)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:
            print("***** Beam generation time: %f" % (et - st))

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

    def _generate_ffiles(self, regen=False):
        ## Generate the beam transfers ordered by frequency.
        ## Divide frequencies between MPI processes and calculate the beams
        ## for the baselines, then write out into separate files.

        for fi in mpiutil.mpirange(self.nfreq):

            if os.path.exists(self._ffile(fi)) and not regen:
                print(
                    "f index %i. File: %s exists. Skipping..." % (fi, (self._ffile(fi)))
                )
                continue
            else:
                print("f index %i. Creating file: %s" % (fi, (self._ffile(fi))))

            f = h5py.File(self._ffile(fi), "w")

            # Set a few useful attributes.
            # f.attrs['baselines'] = self.telescope.baselines
            # f.attrs['baseline_indices'] = np.arange(self.telescope.npairs)
            f.attrs["frequency_index"] = fi
            f.attrs["frequency"] = self.telescope.frequencies[fi]
            f.attrs["cylobj"] = self._telescope_pickle

            dsize = (
                self.telescope.nbase,
                self.telescope.num_pol_sky,
                self.telescope.lmax + 1,
                2 * self.telescope.mmax + 1,
            )

            csize = (
                min(10, self.telescope.nbase),
                self.telescope.num_pol_sky,
                self.telescope.lmax + 1,
                1,
            )

            dset = f.create_dataset(
                "beam_freq", dsize, chunks=csize, compression="lzf", dtype=np.complex128
            )

            # Divide into roughly 5 GB chunks
            nsections = int(
                np.ceil(np.prod(dsize) * 16.0 / 2 ** 30.0 / self._mem_switch)
            )

            print(
                "Dividing calculation of %f GB array into %i sections."
                % (np.prod(dsize) * 16.0 / 2 ** 30.0, nsections)
            )

            b_sec = np.array_split(
                np.arange(self.telescope.npairs, dtype=np.int), nsections
            )
            f_sec = np.array_split(
                fi * np.ones(self.telescope.npairs, dtype=np.int), nsections
            )

            # Iterate over each section, generating transfers and save them.
            for si in range(nsections):
                print("Calculating section %i of %i...." % (si, nsections))
                b_ind, f_ind = b_sec[si], f_sec[si]
                tarray = self.telescope.transfer_matrices(b_ind, f_ind)
                dset[
                    (b_ind[0]) : (b_ind[-1] + 1), ..., : (self.telescope.mmax + 1)
                ] = tarray[..., : (self.telescope.mmax + 1)]
                dset[
                    (b_ind[0]) : (b_ind[-1] + 1), ..., (-self.telescope.mmax) :
                ] = tarray[..., (-self.telescope.mmax) :]
                del tarray

            f.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

    def _generate_mfiles(self, regen=False):

        if os.path.exists(self.directory + "/beam_m/COMPLETED") and not regen:
            if mpiutil.rank0:
                print("******* m-files already generated ********")
            return

        st = time.time()

        nfb = self.telescope.nfreq * self.telescope.nbase
        fbmap = np.mgrid[: self.telescope.nfreq, : self.telescope.nbase].reshape(2, nfb)

        # Calculate the number of baselines to deal with at any one time. Aim
        # to have a maximum of 4 GB in memory at any one time
        fbsize = (
            self.telescope.num_pol_sky
            * (self.telescope.lmax + 1)
            * (2 * self.telescope.mmax + 1)
            * 16.0
        )

        nodemem = 3.0 * 2 ** 30.0

        num_fb_per_node = int(nodemem / fbsize)
        num_fb_per_chunk = num_fb_per_node * mpiutil.size
        num_chunks = int(
            np.ceil(1.0 * nfb / num_fb_per_chunk)
        )  # Number of chunks to break the calculation into

        if mpiutil.rank0:
            print("Splitting into %i chunks...." % num_chunks)

        # The local m sections
        lm, sm, em = mpiutil.split_local(self.telescope.mmax + 1)

        # Iterate over all m's and create the hdf5 files we will write into.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._mfile(mi)) and not regen:
                print(
                    "m index %i. File: %s exists. Skipping..." % (mi, (self._mfile(mi)))
                )
                continue

            f = h5py.File(self._mfile(mi), "w")

            dsize = (
                self.telescope.nfreq,
                2,
                self.telescope.nbase,
                self.telescope.num_pol_sky,
                self.telescope.lmax + 1,
            )
            csize = (
                1,
                2,
                min(10, self.telescope.nbase),
                self.telescope.num_pol_sky,
                self.telescope.lmax + 1,
            )
            f.create_dataset(
                "beam_m", dsize, chunks=csize, compression="lzf", dtype=np.complex128
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
                print("Starting chunk %i of %i" % (ci + 1, num_chunks))

            # Unpack freq-baselines range into num, start and end
            fbnum, fbstart, fbend = fbrange

            # Split the fb list into the ones local to this node
            loc_num, loc_start, loc_end = mpiutil.split_local(fbnum)
            fb_ind = list(range(fbstart + loc_start, fbstart + loc_end))

            # Extract the local frequency and baselines indices
            f_ind = fbmap[0, fb_ind]
            bl_ind = fbmap[1, fb_ind]

            # Create array to hold local matrix section
            fb_array = np.zeros(
                (
                    loc_num,
                    2,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    self.telescope.mmax + 1,
                ),
                dtype=np.complex128,
            )

            if loc_num > 0:

                # Calculate the local Beam Matrices
                tarray = self.telescope.transfer_matrices(bl_ind, f_ind)

                # Expensive memory copy into array section
                for mi in range(1, self.telescope.mmax + 1):
                    fb_array[:, 0, ..., mi] = tarray[..., mi]
                    fb_array[:, 1, ..., mi] = (-1) ** mi * tarray[..., -mi].conj()

                fb_array[:, 0, ..., 0] = tarray[..., 0]

                del tarray

            if mpiutil.rank0:
                print("Transposing and writing chunk.")

            # Perform an in memory MPI transpose to get the m-ordered array
            m_array = mpiutil.transpose_blocks(
                fb_array,
                (
                    fbnum,
                    2,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    self.telescope.mmax + 1,
                ),
            )

            del fb_array

            # Write out the current set of chunks into the m-files.
            for lmi, mi in enumerate(range(sm, em)):

                # Open up correct m-file
                with h5py.File(self._mfile(mi), "r+") as mfile:

                    # Lookup where to write Beam Transfers and write into file.
                    for fbl, fbi in enumerate(range(fbstart, fbend)):
                        fi = fbmap[0, fbi]
                        bi = fbmap[1, fbi]
                        mfile["beam_m"][fi, :, bi] = m_array[fbl, ..., lmi]

            del m_array

        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(self.directory + "/beam_m/COMPLETED", "a").close()

            # Print out timing
            print("=== MPI transpose took %f s ===" % (et - st))

    def _generate_svdfiles(self, regen=False, skip_svd_inv=False):

        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # If external SVD basis is specified, loop over all m's to find the
        # maximum singular value
        if self.external_svd_basis_dir is not None:
            max_sv = 0.

            for mi in mpiutil.mpirange(self.telescope.mmax + 1):
                fe = h5py.File(self._external_svdfile(mi))
                ext_sig = fe["sig"][:]
                max_sv = max(ext_sig[0], max_sv)
                fe.close()

            self.external_global_max_sv = mpiutil.world.allreduce(max_sv, op=MPI.MAX)
            # print("Rank %d: max_sv=%g" % (mpiutil.rank,self.external_global_max_sv ))

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print(
                    "m index %i. File: %s exists. Skipping..."
                    % (mi, (self._svdfile(mi)))
                )
                continue
            else:
                print("m index %i. Creating SVD file: %s" % (mi, self._svdfile(mi)))

            # Open m beams for reading.
            fm = h5py.File(self._mfile(mi), "r")

            # If external SVD basis is specified...
            if self.external_svd_basis_dir is not None:
                # Open file for this m, and read U and singular values
                fe = h5py.File(self._external_svdfile(mi))
                ext_u = fe["u"][:]
                ext_sig = fe["sig"][:]
                fe.close()
                # Determine how many modes to cut, based on global and local thresholds
                global_ext_sv_cut = (ext_sig > self.external_svthreshold_global * self.external_global_max_sv).sum()
                local_ext_sv_cut = (ext_sig > self.external_svthreshold_local * ext_sig[0]).sum()
                cut = max(global_ext_sv_cut, local_ext_sv_cut)
                if self.external_sv_mode_cut is not None:
                    cut = self.external_sv_mode_cut
                # Define vector of ones with same length as ext_sig, and put zeros
                # for modes we want to cut
                Z_ext_vec = np.ones(ext_u.shape[0])
                Z_ext_vec[:cut] = 0.0

            # Open file to write SVD results into.
            fs = h5py.File(self._svdfile(mi), "w")

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
            dset_sig = fs.create_dataset("singularvalues", dsize_sig, dtype=np.float64)

            ## For each frequency in the m-files read in the block, SVD it,
            ## and construct the new beam matrix, and save.
            for fi in np.arange(self.telescope.nfreq):

                # Read the positive and negative m beams, and combine into one.
                bf = fm["beam_m"][fi][:].reshape(
                    self.ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1
                )

                # Apply external SVD projection to B matrix by acting
                # from the left
                if self.external_svd_basis_dir is not None:
                    bf = self._project_beam_with_ext_svd(bf, ext_u, Z_ext_vec)

                # If desired, apply external SVD projection to noise matrix
                # before pre-whitening B matrix. This involves sandwiching
                # N between the ext-SVD projection matrix, inverting the
                # projected N, doing a Cholesky decomposition, and applying the
                # result to B
                if self.external_svd_basis_dir is not None \
                        and self.prewhiten_with_ext_svd_projection:
                    noise_matrix = self.telescope.noisepower(
                        np.arange(self.telescope.npairs), fi
                    ).flatten()
                    noise_matrix = np.concatenate([noise_matrix, noise_matrix])
                    noise_matrix = np.diag(noise_matrix)
                    p = self._projection_matrix_from_ext_svd(ext_u, Z_ext_vec)
                    noise_matrix = np.dot(p, np.dot(noise_matrix, p.T.conj()))

                    noise_matrix_inv = la.inv(noise_matrix)

                    # This is a bad hack that has not been tested much: if
                    # Cholesky of projected N^-1 fails, regularize the diagonal
                    # and try again.
                    try:
                        noise_matrix_inv_chol = la.cholesky(noise_matrix_inv).T.conj()
                    except np.linalg.LinAlgError:
                        print('Cholesky of inverse noise matrix failed! Regularizing...')
                        noise_matrix[np.diag_indices_from(noise_matrix)] \
                            += 1e-5 * noise_matrix.max()
                        noise_matrix_inv = la.inv(noise_matrix)
                        noise_matrix_inv_chol = la.cholesky(noise_matrix_inv).T.conj()

                    # noise_matrix_inv_chol = la.cholesky(noise_matrix)
                    # noise_matrix_inv_chol = la.pinv(noise_matrix_inv_chol)

                    bfr = bf.reshape(self.ntel, -1)
                    for i in range(bfr.shape[-1]):
                        bfr[:,i] = np.dot(noise_matrix_inv_chol, bfr[:,i])

                # If not doing ext-SVD on noise matrix before pre-whitening B,
                # things are much easier
                else:
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
                        (self.telescope.num_pol_sky - 1) * (self.telescope.lmax + 1),
                    )
                    u2, s2 = matrix_nullspace(
                        bfp, rtol=self.polsvcut, errmsg=("SVD2 m=%i f=%i" % (mi, fi))
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

                    # Save out the evecs (for transforming from the telescope frame
                    # into the SVD basis). If ext-SVD is applied to N before
                    # prewhitening, need to do the same thing here.
                    if self.external_svd_basis_dir is not None \
                            and self.prewhiten_with_ext_svd_projection:
                        dset_ut[fi, :nmodes] = np.dot(ut, noise_matrix_inv_chol)
                    else:
                        dset_ut[fi, :nmodes] = ut * noisew[np.newaxis, :]
                    # If doing ext-SVD projection, include that as first operation
                    # in tel-SVD projection (prior to noise pre-whitening)
                    if self.external_svd_basis_dir is not None:
                        dset_ut[fi, :nmodes] = np.dot(
                            dset_ut[fi, :nmodes],
                            self._projection_matrix_from_ext_svd(ext_u, Z_ext_vec)
                        )

                    # Save out the modified beam matrix (for mapping from the sky into the SVD basis)
                    dset_bsvd[fi, :nmodes] = beam.reshape(
                        nmodes, self.telescope.num_pol_sky, self.telescope.lmax + 1
                    )

                    if not skip_svd_inv:
                        ibeam = la.pinv(beam)
                        # Find the pseudo-inverse of the beam matrix and save to disk.
                        dset_ibsvd[fi, :, :, :nmodes] = ibeam.reshape(
                            self.telescope.num_pol_sky, self.telescope.lmax + 1, nmodes
                        )

                    # Save out the singular values for each block
                    dset_sig[fi, :nmodes] = sig

            # Write a few useful attributes.
            fs.attrs["baselines"] = self.telescope.baselines
            fs.attrs["m"] = mi
            fs.attrs["frequencies"] = self.telescope.frequencies

            fs.close()
            fm.close()

            if self.external_svd_basis_dir is not None:
                fe.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()

        # If external SVD basis was used, make marker file to indicate that
        if mpiutil.rank0 and self.external_svd_basis_dir is not None:
            open(self.directory + "/beam_m/EXT_SVD_USED", "a").close()

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
            Telescope vector to return.
        """

        vecf = np.zeros((self.nfreq, self.ntel), dtype=np.complex128)

        with h5py.File(self._mfile(mi), "r") as mfile:

            for fi in range(self.nfreq):
                beamf = mfile["beam_m"][fi][:].reshape((self.ntel, self.nsky))
                vecf[fi] = np.dot(beamf, vec[fi].reshape(self.nsky))

        return vecf

    project_vector_forward = project_vector_sky_to_telescope

    def project_vector_telescope_to_sky(self, mi, vec):
        """Invert a vector from the telescope space onto the sky. This is the
        map-making process.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [freq, baseline, polarisation]

        Returns
        -------
        tvec : np.ndarray
            Sky vector to return.
        """

        ibeam = self.invbeam_m(mi).reshape((self.nfreq, self.nsky, self.ntel))

        vecb = np.zeros((self.nfreq, self.nsky), dtype=np.complex128)
        vec = vec.reshape((self.nfreq, self.ntel))

        for fi in range(self.nfreq):
            vecb[fi] = np.dot(ibeam[fi], vec[fi, :].reshape(self.ntel))

        return vecb.reshape(
            (self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
        )

    project_vector_backward = project_vector_telescope_to_sky

    def project_vector_backward_dirty(self, mi, vec):

        dbeam = self.beam_m(mi).reshape((self.nfreq, self.ntel, self.nsky))
        dbeam = dbeam.transpose((0, 2, 1)).conj()

        vecb = np.zeros((self.nfreq, self.nsky), dtype=np.complex128)
        vec = vec.reshape((self.nfreq, self.ntel))

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

    def project_matrix_sky_to_telescope(self, mi, mat):
        """Project a covariance matrix from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [pol, pol, l, freq, freq]

        Returns
        -------
        tmat : np.ndarray
            Covariance in telescope basis.
        """
        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1

        beam = self.beam_m(mi).reshape((self.nfreq, self.ntel, npol, lside))

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
        mat : np.ndarray
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

            matf[
                svbounds[fi] : svbounds[fi + 1], svbounds[fi] : svbounds[fi + 1]
            ] = np.dot((fbeam * lmat), fbeam.T.conj())

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

        # Get the SVD beam matrix
        beam = self.beam_ut(mi)

        # Create the output matrix (shape is calculated from input shape)
        vecf = np.zeros((svbounds[-1],) + vec.shape[2:], dtype=np.complex128)

        # Should it be a +=?
        for fi in self._svd_freq_iter(mi):

            fbeam = beam[fi, : svnum[fi], :]  # Beam matrix for this frequency and cut
            lvec = vec[fi, :]  # Matrix section for this frequency

            vecf[svbounds[fi] : svbounds[fi + 1]] = np.dot(fbeam, lvec)

        return vecf

    def project_matrix_telescope_to_svd(self, mi, mat):
        """Map a matrix from the telescope space into the SVD basis.

        This projection may be lose information about the sky, depending on
        the polarisation filtering.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Telescope-basis matrix, packed as [freq, baseline, freq, baseline].

        Returns
        -------
        out_mat : np.ndarray
            SVD-basis matrix, packed as [svdnum, svdnum].
        """

        # Number of significant sv modes at each frequency, and corresponding
        # array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get SVD beam matrix, packed as (nfreq, svd_len, ntel)
        beam = self.beam_ut(mi)

        # Make empty array for output matrix
        out_mat = np.zeros((svbounds[-1],svbounds[-1]), dtype=np.complex128)

        # Loop over frequencies, projecting each frequency block of the input
        # matrix using the SVD beam matrix at that frequency
        for fi in self._svd_freq_iter(mi):
            fi_beam = beam[fi, : svnum[fi], :]

            for fj in self._svd_freq_iter(mi):
                fj_beam = beam[fj, : svnum[fj], :]

                out_mat[svbounds[fi] : svbounds[fi + 1], svbounds[fj] : svbounds[fj + 1]] \
                    = np.dot(fi_beam, np.dot(mat[fi, :, fj, :], fj_beam.T.conj()))

        return out_mat


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

        # Get the SVD beam matrix
        beam = self.beam_ut(mi)

        # Create the output matrix (shape is calculated from input shape)
        vecf = np.zeros((self.nfreq, self.ntel), dtype=np.complex128)

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

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)

        # Create the output matrix
        vecf = np.zeros((svbounds[-1],) + vec.shape[3:], dtype=np.complex128)

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

        # if not conj:
        #     raise Exception("Not implemented non conj yet.")

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_svd(mi) if conj else self.invbeam_svd(mi)

        # Create the output matrix
        vecf = np.zeros(
            (self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
            + vec.shape[1:],
            dtype=np.complex128,
        )

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

    def _project_beam_with_ext_svd(self, bf, u, Z):
        """Project modes from external SVD basis out from beam transfer matrix

        Parameters
        ----------
        bfr : np.array
            Beam transfer matrix at a given m and frequency, packed as
            [ntel, npol, lmax+1]
        u : np.array
            U matrix from external SVD, packed as [ntel,nmodes]
        Z : np.array
            Vector of zeros and ones, packed as [nmodes], with zeros for modes
            we want to cut and ones for modes we

        Returns
        -------
        bfp : np.array
            Beam transfer matrix with some SVD modes projected out,
            in original packing
        """

        bfp = bf.reshape(self.ntel, -1)
        p = self._projection_matrix_from_ext_svd(u, Z)
        bfp = np.dot(p, bfp)
        # bfp = np.dot( u, np.dot(np.diag(Z), np.dot(u.T.conj(), bfp) ) )
        bfp = bfp.reshape(
            self.ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1
        )

        return bfp

    def _projection_matrix_from_ext_svd(self, u, Z):
        """Construct projection matrix that gets applied to beam transfer matrix

        Parameters
        ----------
        u : np.array
            U matrix from external SVD, packed as [ntel,nmodes]
        Z : np.array
            Vector of zeros and ones, packed as [nmodes], with zeros for modes
            we want to cut and ones for modes we want to keep

        Returns
        -------
        p : np.array
            Projection matrix that zeros external-SVD modes as specified by Z,
            packed as [ntel, ntel]
        """
        return np.dot(u, np.dot(np.diag(Z), u.T.conj()))


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
    """BeamTransfer class that performs the old temperature only SVD.
    """

    def _generate_svdfiles(self, regen=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print(
                    "m index %i. File: %s exists. Skipping..."
                    % (mi, (self._svdfile(mi)))
                )
                continue
            else:
                print("m index %i. Creating SVD file: %s" % (mi, self._svdfile(mi)))

            # Open m beams for reading.
            fm = h5py.File(self._mfile(mi), "r")

            # Open file to write SVD results into.
            fs = h5py.File(self._svdfile(mi), "w")

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
            dset_sig = fs.create_dataset("singularvalues", dsize_sig, dtype=np.float64)

            ## For each frequency in the m-files read in the block, SVD it,
            ## and construct the new beam matrix, and save.
            for fi in np.arange(self.telescope.nfreq):

                # Read the positive and negative m beams, and combine into one.
                bf = fm["beam_m"][fi][:].reshape(
                    self.ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1
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
                    self.svd_len, self.telescope.num_pol_sky, self.telescope.lmax + 1
                )

                # Find the pseudo-inverse of the beam matrix and save to disk.
                dset_ibsvd[fi] = la.pinv(bsvd).reshape(
                    self.telescope.num_pol_sky, self.telescope.lmax + 1, self.svd_len
                )

                # Save out the singular values for each block
                dset_sig[fi] = sig

            # Write a few useful attributes.
            fs.attrs["baselines"] = self.telescope.baselines
            fs.attrs["m"] = mi
            fs.attrs["frequencies"] = self.telescope.frequencies
            fs.attrs["cylobj"] = self._telescope_pickle

            fs.close()
            fm.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()


class BeamTransferFullSVD(BeamTransfer):
    """BeamTransfer class that performs the old temperature only SVD.
    """

    def _generate_svdfiles(self, regen=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print(
                    "m index %i. File: %s exists. Skipping..."
                    % (mi, (self._svdfile(mi)))
                )
                continue
            else:
                print("m index %i. Creating SVD file: %s" % (mi, self._svdfile(mi)))

            # Open m beams for reading.
            fm = h5py.File(self._mfile(mi), "r")

            # Open file to write SVD results into.
            fs = h5py.File(self._svdfile(mi), "w")

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
            dset_sig = fs.create_dataset("singularvalues", dsize_sig, dtype=np.float64)

            ## For each frequency in the m-files read in the block, SVD it,
            ## and construct the new beam matrix, and save.
            for fi in np.arange(self.telescope.nfreq):

                # Read the positive and negative m beams, and combine into one.
                bf = fm["beam_m"][fi][:].reshape(
                    self.ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1
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
                    self.svd_len, self.telescope.num_pol_sky, self.telescope.lmax + 1
                )

                # Find the pseudo-inverse of the beam matrix and save to disk.
                dset_ibsvd[fi] = la.pinv(bsvd).reshape(
                    self.telescope.num_pol_sky, self.telescope.lmax + 1, self.svd_len
                )

                # Save out the singular values for each block
                dset_sig[fi] = sig

            # Write a few useful attributes.
            fs.attrs["baselines"] = self.telescope.baselines
            fs.attrs["m"] = mi
            fs.attrs["frequencies"] = self.telescope.frequencies
            fs.attrs["cylobj"] = self._telescope_pickle

            fs.close()
            fm.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()

    @property
    def svd_len(self):
        """The size of the SVD output matrices."""
        return min((self.telescope.lmax + 1) * self.telescope.num_pol_sky, self.ntel)


class BeamTransferNoSVD(BeamTransfer):

    svcut = 0.0

    def project_matrix_sky_to_svd(self, mi, mat, *args, **kwargs):
        return self.project_matrix_sky_to_telescope(mi, mat).reshape(
            self.ndof(mi), self.ndof(mi)
        )

    def project_vector_sky_to_svd(self, mi, vec, *args, **kwargs):
        return self.project_vector_sky_to_telescope(mi, vec)

    def project_matrix_diagonal_telescope_to_svd(self, mi, dmat, *args, **kwargs):
        return np.diag(dmat.flatten())

    def project_vector_telescope_to_svd(self, mi, vec, *args, **kwargs):
        return vec

    def beam_svd(self, mi, *args, **kwargs):
        return self.beam_m(mi)

    def ndof(self, mi, *args, **kwargs):

        return self.ntel * self.nfreq

    @property
    def ndofmax(self):
        return self.ntel * self.nfreq



class BeamTransferFullFreq(BeamTransfer):
    """BeamTransfer class that allows for off-diagonal frequency elements in SVD.

    Beam transfer matrices are still assumed to be frequency-diagonal,
    but off-diagonality is allowed in any step related to the SVD
    decompositions.

    Parameters
    ----------
    directory : string
        Path of directory to read and write Beam Transfers from.
    telescope : drift.core.telescope.TransitTelescope, optional
        Telescope object to use for calculation. If `None` (default), try to
        load a cached version from the given directory.

    Attributes
    ----------
    svcut
    polsvcut
    ntel
    nsky
    nfreq
    svd_len
    ndofmax

    Methods
    -------
    ndof
    beam_m
    invbeam_m
    beam_svd
    beam_ut
    invbeam_svd
    beam_singularvalues
    generate
    project_vector_sky_to_telescope
    project_vector_telescope_to_sky
    project_vector_sky_to_svd
    project_vector_svd_to_sky
    project_vector_telescope_to_svd
    project_matrix_sky_to_telescope
    project_matrix_sky_to_svd
    """

    # ====== SVD Beam loading ===========================

    @util.cache_last
    def beam_svd(self, mi, fi=None):
        """Fetch the SVD beam transfer matrix (S V^H) for a given m. This SVD beam
        transfer projects from the sky into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer, optional
            frequency block to fetch - raises exception if used. (Default: None)

        Returns
        -------
        beam : np.ndarray [nfreq*svd_len, nfreq, npol_sky, lmax+1]
        """

        svdfile = h5py.File(self._svdfile(mi), "r")

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            bs = svdfile["beam_svd"][:]
        else:
            raise Exception(
                "Cannot request individual beam_svd frequency in BeamTransferFullFreq!"
            )

        svdfile.close()

        return bs

    @util.cache_last
    def invbeam_svd(self, mi, fi=None):
        """Fetch the SVD beam transfer matrix (S V^H) for a given m. This SVD beam
        transfer projects from the sky into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer, optional
            frequency block to fetch - raises exception if used. (Default: None)

        Returns
        -------
        beam : np.ndarray [nfreq, npol_sky, lmax+1, nfreq*svd_len]
        """

        svdfile = h5py.File(self._svdfile(mi), "r")

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            ibs = svdfile["invbeam_svd"][:]
        else:
            raise Exception(
                "Cannot request individual invbeam_svd frequency in BeamTransferFullFreq!"
            )

        svdfile.close()

        return ibs

    @util.cache_last
    def beam_ut(self, mi, fi=None):
        """Fetch the SVD beam transfer matrix (U^H) for a given m. This SVD beam
        transfer projects from the telescope space into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch - raises exception if used. (Default: None)

        Returns
        -------
        beam : np.ndarray (nfreq*svd_len, nfreq*ntel)
        """

        svdfile = h5py.File(self._svdfile(mi), "r")

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            bs = svdfile["beam_ut"][:]
        else:
            raise Exception(
                "Cannot request individual beam_ut frequency in BeamTransferFullFreq!"
            )

        svdfile.close()

        return bs

    @util.cache_last
    def beam_singularvalues(self, mi):
        """Fetch the vector of beam singular values for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.

        Returns
        -------
        beam : np.ndarray (nfreq*svd_len)
        """

        svdfile = h5py.File(self._svdfile(mi), "r")
        sv = svdfile["singularvalues"][:]
        svdfile.close()

        return sv

    # ===================================================

    # ====== Generation of all the cache files ==========

    def _generate_svdfiles(self, regen=False, skip_svd_inv=False):

        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print(
                    "m index %i. File: %s exists. Skipping..."
                    % (mi, (self._svdfile(mi)))
                )
                continue
            else:
                print("m index %i. Creating SVD file: %s" % (mi, self._svdfile(mi)))
                self._generate_svdfile_m(mi, skip_svd_inv)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()

        # Make marker file to indicate that full-freq SVD has been used
        open(self.directory + "/beam_m/FULLFREQ_SVD_USED", "a").close()


    def _generate_svdfile_m(self, mi, skip_svd_inv=False):

        # Fetch number of frequencies
        nfreq = self.telescope.nfreq

        ## Read in the beam transfer matrix, expand it to full-frequency form,
        ## perform SVDs, and save products.

        # Read the positive and negative m beams, and combine into one.
        # B is originally packed as [freq,msign,base,pol,ell].
        fm = h5py.File(self._mfile(mi), "r")
        b_diag = fm["beam_m"][:]
        b_diag_shape = b_diag.shape
        fm.close()

        # Expand beam transfer matrix to 2 freq axes,
        # packed as [freq,msign,base,freq,pol,ell]
        b_full_shape = b_diag_shape[:3] + (b_diag_shape[0], b_diag_shape[3], b_diag_shape[4],)
        b_full = np.zeros(b_full_shape, dtype=np.complex128)
        for i in range(nfreq):
            b_full[i,:,:,i,:,:] = b_diag[i]

        # Perform any preprocessing of beam transfer matrix that is desired
        # prior to prewhitening and SVDs, and return beam transfer matrix
        # and any ancillary information used for this preprocessing.
        # In derived classes, this will be used for various filters that are
        # applied to telescope-basis data
        b_full, pp_info = self._preprocess_full_beam_transfer_matrix(mi, b_full)

        # Prewhiten beam transfer matrix and reshape to [freq*msign*nbase,freq*pol*ell]
        b_full = self._prewhiten_beam_transfer_matrix(b_full)
        bfr = b_full.reshape(nfreq * self.ntel, -1)

        success = False

        # If unpolarized, skip first 2 SVDs
        if self.telescope.num_pol_sky == 1:
            bf2 = bfr
            ut2 = np.identity(nfreq * bt.ntel, dtype=np.complex128)
        else:
            ## SVD 1 - coarse projection onto sky-modes
            u1, s1 = matrix_image(
                bfr, rtol=1e-10, errmsg=("SVD1 m=%i" % (mi))
            )

            ut1 = u1.T.conj()
            bf1 = np.dot(ut1, bfr)

            ## SVD 2 - project onto polarisation null space
            bfp = bf1.reshape(
                bf1.shape[0],
                nfreq,
                self.telescope.num_pol_sky,
                self.telescope.lmax + 1,
            )[:, :, 1:]
            bfp = bfp.reshape(
                bf1.shape[0],
                nfreq * (self.telescope.num_pol_sky - 1) * (self.telescope.lmax + 1),
            )
            u2, s2 = matrix_nullspace(
                bfp, rtol=self.polsvcut, errmsg=("SVD2 m=%i" % (mi))
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
                -1, nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1
            )[:, :, 0].reshape(-1, nfreq * (self.telescope.lmax + 1))

            u3, s3 = matrix_image(
                bft, rtol=0.0, errmsg=("SVD3 m=%i" % (mi))
            )
            ut3 = np.dot(u3.T.conj(), ut2)

            nmodes = ut3.shape[0]

            # Final products
            ut = ut3
            sig = s3[:nmodes]
            beam = np.dot(ut3, bfr)

            # Apply prewhitening to U^T, so the saved U^T includes the
            # prewhitening operation
            ut = self._apply_prewhitening_to_beam_ut(ut)

            # If any preprocessing of beam transfer matrix has been performed,
            # we need to apply the same preprocessing to U^T from the right.
            # (bfr includes the preprocessing and prewhitening, so the beam
            # variable above also includes all that)
            ut = self._apply_preprocessing_to_beam_ut(mi, ut, pp_info)

            # Set flag that saves products to files later
            success = True


        # Open file to write SVD results into.
        fs = h5py.File(self._svdfile(mi), "w")

        # Create a chunked dataset for writing the SVD beam matrix into.
        dsize_bsvd = (
            nfreq * self.svd_len,
            nfreq,
            self.telescope.num_pol_sky,
            self.telescope.lmax + 1,
        )
        csize_bsvd = (
            min(10, self.svd_len),
            1,
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
                nfreq,
                self.telescope.num_pol_sky,
                self.telescope.lmax + 1,
                nfreq * self.svd_len,
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
        dsize_ut = (nfreq * self.svd_len, nfreq * self.ntel)
        csize_ut = (min(10, self.svd_len), self.ntel)
        dset_ut = fs.create_dataset(
            "beam_ut",
            dsize_ut,
            chunks=csize_ut,
            compression="lzf",
            dtype=np.complex128,
        )

        # Create a dataset for the singular values.
        dsize_sig = (nfreq * self.svd_len, )
        dset_sig = fs.create_dataset("singularvalues", dsize_sig, dtype=np.float64)

        # Write a few useful attributes and close SVD file
        fs.attrs["baselines"] = self.telescope.baselines
        fs.attrs["m"] = mi
        fs.attrs["frequencies"] = self.telescope.frequencies

        if success:

            # Save the combined U^T matrix (for transforming from the telescope
            # basis into the SVD basis).
            dset_ut[:nmodes] = ut

            # Save the modified beam matrix (for transforming from the sky into
            # the SVD basis)
            dset_bsvd[:nmodes] = beam.reshape(
                nmodes, nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1
            )

            if not skip_svd_inv:
                ibeam = la.pinv(beam)
                # Find the pseudo-inverse of the beam matrix and save to disk.
                dset_ibsvd[:, :, :, :nmodes] = ibeam.reshape(
                    nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1, nmodes
                )

            # Save the singular values
            dset_sig[:nmodes] = sig

        fs.close()


    def _preprocess_beam_transfer_matrix(self, mi, b):
        """Preprocess beam transfer matrix before prewhitening.

        This assumes b is packed as [freq,msign,base,freq,pol,ell].

        In the base BeamTransferFullFreq class, this routine does nothing,
        but derived classes can use it for telescope-basis filtering
        """
        return b, None

    def _apply_preprocessing_to_beam_ut(self, mi, ut, pp_info):
        """Apply beam transfer preprocessing to beam U^T from the right.

        In the base BeamTransferFullFreq class, this routine does nothing,
        but derived classes can use it for telescope-basis filtering
        """
        return ut


    def _prewhiten_beam_transfer_matrix(self, b):
        """Prewhiten beam transfer matrix using instrumental noise.

        Assuming beam transfer matrix is packed as
        [freq,msign,base,freq,pol,ell], but is diagonal in freq,
        we prewhiten by applying
        N^-1/2 B.
        """

        # Reshape b into convenient form for prewhitening
        b_local_shape = b.shape
        b_local = b.reshape(self.telescope.nfreq, self.ntel, self.telescope.nfreq, -1)

        # Loop over frequencies
        for fi in range(self.telescope.nfreq):
            # Make N^-1/2 for this frequency
            noisew = self.telescope.noisepower(
                np.arange(self.telescope.npairs), fi
            ).flatten() ** (-0.5)
            # Double up, accounting for 2 m signs
            noisew = np.concatenate([noisew, noisew])
            # Apply N^-1/2 to frequency-diagonal elements of B
            b_local[fi,:,fi,:] *= noisew[:, np.newaxis]

        # Reshape b back into original form, and return
        return b_local.reshape(b_local_shape)


    def _apply_prewhitening_to_beam_ut(self, ut):
        """Apply prewhitening to U^T matrix from telescope-SVD decomposition.

        Assumes ut is packed as [freq*svd_len, freq*ntel]
        """

        # Make array to hold all noise weights
        noisew = np.zeros(self.telescope.nfreq * self.ntel)

        # Fill up array frequency by frequency
        for fi in range(self.telescope.nfreq):
            # Make N^-1/2 for this frequency
            noisew_f = self.telescope.noisepower(
                np.arange(self.telescope.npairs), fi
            ).flatten() ** (-0.5)
            # Double up, accounting for 2 m signs
            noisew_f = np.concatenate([noisew_f, noisew_f])
            # Include in total array
            noisew[fi * self.ntel : (fi+1) * self.ntel] = noisew_f

        # Apply weights to U^T from the right
        ut = ut * noisew[np.newaxis, :]
        return ut


    def _collect_svd_spectrum(self):
        """Gather the SVD spectrum into a single file."""

        svd_func = lambda mi: self.beam_singularvalues(mi)

        svdspectrum = kltransform.collect_m_array(
            list(range(self.telescope.mmax + 1)),
            svd_func,
            (self.nfreq * self.svd_len,),
            np.float64,
        )

        if mpiutil.rank0:

            with h5py.File(self.directory + "/svdspectrum.hdf5", "w") as f:

                f.create_dataset("singularvalues", data=svdspectrum)

        mpiutil.barrier()

    # ===================================================

    # ====== Projections between spaces ==================

    def _svd_num(self, mi):
        ## Calculate the number of SVD modes meeting the cut,
        ## and return the number. Unlike in the frequency-diagonal case,
        ## this only returns a number, because SVD modes are defined
        ## across all frequencies

        # Get the array of singular values for each mode
        sv = self.beam_singularvalues(mi)

        # Number of significant sv modes at each frequency
        svnum = (sv > sv.max() * self.svcut).sum()

        return svnum

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

        # Number of significant SV modes
        svnum = self._svd_num(mi)

        # Create the output matrix
        matf = np.zeros((svnum, svnum), dtype=np.complex128)

        # Should it be a +=?
        for pi in range(npol):
            for pj in range(npol):
                # beam is packed as [nfreq*svd_len, nfreq, pol, ell],
                # mat is packed as [pol, pol, ell, freq, freq]

                for fi in range(self.telescope.nfreq):

                    fibeam = beam[
                        :svnum, fi, pi, :
                    ]  # Beam for this pol, freq, and svcut (i)

                    for fj in range(self.telescope.nfreq):

                        fjbeam = beam[
                            :svnum, fj,  pj, :
                        ]  # Beam for this pol, freq, and svcut (j)

                        lmat = mat[
                            pi, pj, :, fi, fj
                        ]  # Local section of the sky matrix (i.e C_l part)

                        matf += np.dot(fibeam * lmat, fjbeam.T.conj())

        return matf


    def project_matrix_diagonal_telescope_to_svd(self, mi, dmat):
        """Project a diagonal matrix from the telescope basis into the SVD basis.

        This slightly specialised routine is for projecting the noise
        covariance into the SVD space.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [nfreq, ntel]

        Returns
        -------
        tmat : np.ndarray [nsvd, nsvd]
            Covariance in SVD basis.
        """

        svdfile = h5py.File(self._svdfile(mi), "r")

        # Get the SVD beam matrix, packed as [nfreq*svd_len, nfreq*ntel]
        beam = svdfile["beam_ut"]

        # Number of significant SV modes
        svnum = self._svd_num(mi)

        # Reshape the input matrix into a vector of its diagonals, packed
        # as [nfreq*ntel]
        mat_in = dmat.reshape(-1)

        # Sandwich input matrix between U^T matrices
        matf = np.dot((beam[:svnum, :] * mat_in), beam[:svnum, :].T.conj())

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
            Telescope data vector packed as [freq, ntel, polarisation]

        Returns
        -------
        svec : np.ndarray[svdnum]
            SVD vector to return.
        """

        # Number of significant SV modes
        svnum = self._svd_num(mi)

        # Get the SVD beam matrix, packed as [nfreq*svd_len, nfreq*ntel]
        beam = self.beam_ut(mi)

        # # Create the output matrix (shape is calculated from input shape)
        # vecf = np.zeros(svnum,) + vec.shape[2:], dtype=np.complex128)

        # Reshape input vector to [nfreq*ntel, pol]
        vec_in = vec.reshape(self.telescope.nfreq * self.ntel, -1)

        # Apply SVD projection
        vecf = np.dot(beam[:svnum], vec_in)

        return vecf


    def project_matrix_telescope_to_svd(self, mi, mat):
        """Map a matrix from the telescope space into the SVD basis.

        This projection may be lose information about the sky, depending on
        the polarisation filtering.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Telescope-basis matrix, packed as [freq, baseline, freq, baseline].

        Returns
        -------
        out_mat : np.ndarray
            SVD-basis matrix, packed as [svdnum, svdnum].
        """

        # Number of significant SV modes
        svnum = self._svd_num(mi)

        # Get SVD beam matrix, packed as [nfreq*svd_len, nfreq*ntel]
        beam = self.beam_ut(mi)

        # Reshape input matrix to [nfreq*ntel, nfreq*ntel]
        in_mat = mat.reshape(
            self.telescope.nfreq * self.ntel,
            self.telescope.nfreq * self.ntel
        )

        # Sandwich matrix between U^T matrices
        out_mat = np.dot(beam[:svnum], np.dot(in_mat, beam[:svnum].T.conj()))

        return out_mat


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
        svec : np.ndarray[freq, sign, baseline]
            Data vector to return.
        """

        # Number of significant SV modes
        svnum = self._svd_num(mi)

        # Get the SVD beam matrix, packed as [nfreq*svd_len, nfreq*ntel]
        beam = self.beam_ut(mi)

        # # Create the output matrix
        # vecf = np.zeros((self.nfreq, self.ntel), dtype=np.complex128)

        # Make array to hold all noise weights
        noise = np.zeros(self.telescope.nfreq * self.ntel)

        # Fill up array frequency by frequency
        for fi in range(self.telescope.nfreq):
            # Make N for this frequency
            noise_f = self.telescope.noisepower(
                np.arange(self.telescope.npairs), fi
            ).flatten()
            # Double up, accounting for 2 m signs
            noise_f = np.concatenate([noise_f, noise_f])
            # Include in total array
            noise[fi * self.ntel : (fi+1) * self.ntel] = noise_f

        # As the form of the forward projection is simply a scaling and then
        # projection onto an orthonormal basis, the pseudo-inverse is simply
        # related.
        vec_out = noise * np.dot(beam[:svnum].T.conj(), svec)

        # Reshape vec_out to [freq, 2, npairs]
        vec_out = vec_out.reshape(self.nfreq, 2, self.telescope.npairs)

        return vec_out


    def project_vector_sky_to_svd(self, mi, vec, temponly=False):
        """Project a vector from the the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, pol, lmax+1]
        temponly: boolean
            Force projection of temperature part only (default: False)

        Returns
        -------
        svec : np.ndarray
            SVD vector to return.
        """
        npol = 1 if temponly else self.telescope.num_pol_sky

        # Number of significant SV modes
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix, packed as [nfreq*svd_len, nfreq, pol, ell]
        beam = self.beam_svd(mi)

        # Create the output array
        vec_out = np.zeros((svbounds[-1],) + vec.shape[3:], dtype=np.complex128)

        for pi in range(npol):

            # Get beam for this polarization, and reshape to [sv, freq*ell]
            pbeam = beam[:svnum, :, pi, :].reshape((svnum, -1))

            # Get input vector for this polarization, and reshape to [freq*ell, ...]
            pvec = vec[:, pi].reshape((self.telescope.nfreq * (self.telescope.lmax+1), -1))

            # Accumulate beam dot vec in output array
            vec_out += np.dot(pbeam, pvec)

        return vec_out

    def project_vector_svd_to_sky(self, mi, svec, temponly=False, conj=False):
        """Project a vector from the the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            SVD vector, packed as [nsvd, ...]
        temponly: boolean
            Force projection of temperature part only (default: False)
        conj: boolean
            Reverse projection by applying conjugation (as opposed to pseudo-
            inverse). Default is False.

        Returns
        -------
        svec : np.ndarray
            Sky vector to return, packed as [nfreq, pol, lmax+1, ...]
        """

        npol = 1 if temponly else self.telescope.num_pol_sky

        # Number of significant SV modes
        svnum = self._svd_num(mi)

        # Get the SVD beam matrix, packed as [nfreq*svd_len, nfreq, npol_sky, lmax+1],
        # or inverse SVD beam matrix, packed as [nfreq, npol_sky, lmax+1, nfreq*svd_len]
        beam = self.beam_svd(mi) if conj else self.invbeam_svd(mi)

        # Create the output matrix
        out_vec = np.zeros(
            (self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1)
            + svec.shape[1:],
            dtype=np.complex128,
        )

        for pi in range(npol):

            # Get inverse beam for this polarization, and reshape to [freq*ell, svnum]
            if conj:
                pbeam = beam[:svnum, :, pi, :].reshape(
                    svnum,
                    self.telescope.nfreq * (self.telescope.lmax + 1)
                ).T.conj()
            else:
                pbeam = beam[:, pi, :, :svnum].reshape(
                    self.telescope.nfreq * (self.telescope.lmax + 1),
                    svnum
                )

            # Dot inverse beam into input SV-basis vector, and reshape to
            # [freq, ell, ...]
            out_vec[:, pi, :] = np.dot(pbeam, svec).reshape(
                self.telescope.nfreq,
                self.telescope.lmax + 1,
                -1
            )

        return out_vec


    # ===================================================

    # ====== Dimensionality of the various spaces =======

    @property
    def svd_len(self):
        """The size of the SVD output per frequency."""
        return min(self.telescope.lmax + 1, self.ntel)

    @property
    def ndofmax(self):
        return self.svd_len * self.nfreq

    def ndof(self, mi):
        """The number of degrees of freedom at a given m."""
        return self._svd_num(mi)




class BeamTransferFullFreqExtSVD(BeamTransferFullFreq):
    """BeamTransfer class that allows for extra SVD filtering in telescope basis.

    Before the beam transfer SVD decompositions take place, a certain number
    of externally-defined frequency modes are filtered out at each m. The
    "beam_svd" and "beam_ut" products are then defined to include this filtering,
    while the non-SVD'ed beam transfer matrices are untouched.

    Parameters
    ----------
    directory : string
        Path of directory to read and write Beam Transfers from.
    telescope : drift.core.telescope.TransitTelescope, optional
        Telescope object to use for calculation. If `None` (default), try to
        load a cached version from the given directory.

    Attributes
    ----------
    svcut
    polsvcut
    ntel
    nsky
    nfreq
    svd_len
    ndofmax
    external_svd_basis_dir
    external_svthreshold_global
    external_svthreshold_local
    external_sv_mode_cut
    external_global_max_sv

    Methods
    -------
    ndof
    beam_m
    invbeam_m
    beam_svd
    beam_ut
    invbeam_svd
    beam_singularvalues
    generate
    project_vector_sky_to_telescope
    project_vector_telescope_to_sky
    project_vector_sky_to_svd
    project_vector_svd_to_sky
    project_vector_telescope_to_svd
    project_matrix_sky_to_telescope
    project_matrix_sky_to_svd
    """

    # Directory containing files defining external SVD basis (determined
    # directly from measured visibilities, using draco.analysis.svdfilter.SVDFilter)
    external_svd_basis_dir = None

    # Thresholds for filtering modes defined in external SVD basis:
    #  global ->    Remove modes with singular value higher than external_svthreshold_global
    #               times the largest mode on any m
    #  local  ->    Remove modes with singular value higher than external_svthreshold_local
    #               times the largest mode on each m
    # Default values are such that no modes are filtered out - user must specify something
    # for filtering to take place!
    external_svthreshold_global = 1000.
    external_svthreshold_local = 1000.
    external_sv_mode_cut = None

    def _external_svdfile(self, mi):
        """File containing external SVD basis for a given m.
        """
        if self.external_svd_basis_dir is None:
            raise RuntimeError("Directory containing external SVD basis not specified!")
        else:
            return external_svdfile(self.external_svd_basis_dir, mi, self.telescope.mmax)


    # ===================================================

    # ====== Generation of all the cache files ==========

    def _generate_svdfiles(self, regen=False, skip_svd_inv=False):

        # Loop over all m's to find the maximum singular value for ext-SVD basis
        max_sv = 0.

        for mi in mpiutil.mpirange(self.telescope.mmax + 1):
            fe = h5py.File(self._external_svdfile(mi))
            ext_sig = fe["sig"][:]
            max_sv = max(ext_sig[0], max_sv)
            fe.close()

        self.external_global_max_sv = mpiutil.world.allreduce(max_sv, op=MPI.MAX)
        # print("Rank %d: max_sv=%g" % (mpiutil.rank,self.external_global_max_sv ))

        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print(
                    "m index %i. File: %s exists. Skipping..."
                    % (mi, (self._svdfile(mi)))
                )
                continue
            else:
                print("m index %i. Creating SVD file: %s" % (mi, self._svdfile(mi)))
                self._generate_svdfile_m(mi, skip_svd_inv)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()

        # Make marker file to indicate that full-freq SVD has been used
        open(self.directory + "/beam_m/FULLFREQ_SVD_USED", "a").close()


    def _preprocess_beam_transfer_matrix(self, mi, b):
        """Preprocess beam transfer matrix before prewhitening.

        This assumes b is packed as [freq,msign,base,freq,pol,ell].

        This is the first time ext-SVD information will be used for this
        m, so we define the frequency projection matrix here and return it
        and the second return value.
        """

        # Open file for this m, and read vh and singular values
        fe = h5py.File(self._external_svdfile(mi))
        ext_vh = fe["vh"][:]
        ext_sig = fe["sig"][:]
        fe.close()

        # Determine how many modes to cut, based on global and local thresholds
        global_ext_sv_cut = (ext_sig > self.external_svthreshold_global \
                             * self.external_global_max_sv).sum()
        local_ext_sv_cut = (ext_sig > self.external_svthreshold_local * ext_sig[0]).sum()
        cut = max(global_ext_sv_cut, local_ext_sv_cut)
        if self.external_sv_mode_cut is not None:
            cut = self.external_sv_mode_cut

        # Define vector of ones with same length as ext_sig, put zeros
        # for modes we want to cut, and convert to a diagonal matrix
        Z = np.ones(ext_sig.shape[0])
        Z[:cut] = 0.0
        Z = np.diag(Z)

        # Define a projection matrix at V.Z.V^dagger.
        # Note that this needs to be transposed when applied to the beam
        # transfer matrix or telescope-basis data vector, because of how it
        # was constructed (as the V instead of U matrix in the SVD)
        proj = np.dot(ext_vh.T.conj(), np.dot(Z, ext_vh))

        # Apply this projection matrix to the beam transfer matrix from the left
        b_shape
        b = np.dot(proj.T, b.reshape(self.telescope.nfreq, -1)).reshape(b_shape)

        # Return filtered B and projection matrix
        return b, proj


    def _apply_preprocessing_to_beam_ut(self, mi, ut, pp_info):
        """Apply beam transfer preprocessing to beam U^T from the right.

        pp_info will be the frequency projection matrix defined in
        _preprocess_beam_transfer_matrix, and we can simply apply it to U^T
        from the right to incorporating this projection into the U^T we save.
        """

        # U^T will be packed as [freq*svd_len, freq*ntel], so we need to
        # reshape it to have frequency as the last axis, dot it with
        # the (transposed) projector, and the reshape back to the original
        # format
        nmodes = ut.shape[0]
        ut = np.dot(
            ut.reshape(nmodes, self.telescope.nfreq, self.ntel).transpose(0,2,1),
            proj.T
        ).transpose(0,2,1).reshape(nmodes, -1)

        return ut


    def _prewhiten_beam_transfer_matrix(self, b):
        """Prewhiten beam transfer matrix using instrumental noise.

        We assume that b is packed as [freq,msign,base,freq,pol,ell],
        but is not necessarily diagonal in frequency.
        """

        # Reshape b into convenient form for prewhitening
        b_local_shape = b.shape
        b_local = b.reshape(self.telescope.nfreq, self.ntel, self.telescope.nfreq, -1)

        # Make array to hold all noise weights
        noisew = np.zeros(self.telescope.nfreq * self.ntel)

        # Fill up array frequency by frequency
        for fi in range(self.telescope.nfreq):
            # Make N^-1/2 for this frequency
            noisew_f = self.telescope.noisepower(
                np.arange(self.telescope.npairs), fi
            ).flatten() ** (-0.5)
            # Double up, accounting for 2 m signs
            noisew_f = np.concatenate([noisew_f, noisew_f])
            # Include in total array
            noisew[fi * self.ntel : (fi+1) * self.ntel] = noisew_f

        # Multiply noise weights into b_local elementwise,
        # adding second axis so that weights are multiplied along
        # first axis
        b_local *= noisew[:, np.newaxis]

        # Reshape back to original format, and return
        return b_local.reshape(b_local_shape)

    # ===================================================
