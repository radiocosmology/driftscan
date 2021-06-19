import os
import time

import numpy as np
from mpi4py import MPI

from cora.util import nputil

from caput import mpiutil, config, mpiarray

from drift.core import psestimation


class PSMonteCarlo(psestimation.PSEstimation):
    """An extension of the PSEstimation class to support estimation of the
    Fisher matrix via Monte-Carlo simulations.

    This uses the fact that the covariance of the q-estimator is the Fisher
    matrix to Monte-Carlo the Fisher matrix and the bias. See Padmanabhan and
    Pen (2003), and Dillon et al. (2012).

    Attributes
    ----------
    nsamples : integer
        The number of samples to draw from each band.
    """

    nsamples = config.Property(proptype=int, default=500)
    noiseonly = config.Property(proptype=bool, default=False)

    def gen_sample(self, mi, nsamples=None):
        """Generate a random set of KL-data for this m-mode.

        Found by drawing from the eigenvalue distribution.

        Parameters
        ----------
        mi : integer
            The m-mode to draw from.

        Returns
        -------
        x : np.ndarray[nmodes, self.nsamples]
            The random KL-data. The number of samples is set by the objects
            attribute.
        """

        nsamples = self.nsamples if nsamples is None else nsamples

        evals, evecs = self.kltrans.modes_m(mi)

        # Calculate C**(1/2), this is the weight to generate a draw from C
        w = np.ones_like(evals) if self.noiseonly else (evals + 1.0) ** 0.5

        # Calculate x
        x = nputil.complex_std_normal((evals.shape[0], nsamples)) * w[:, np.newaxis]

        return x

    def _work_fisher_bias_m(self, mi):
        """Worker routine for calculating the Fisher and bias for a given m.

        This method estimates both quantities using Monte-Carlo estimation,
        and the fact that Cov(q_a, q_b) = F_ab.

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

        qa = np.zeros((self.nbands, self.nsamples))

        # Split calculation into subranges to save on memory usage
        num, starts, ends = mpiutil.split_m(self.nsamples, (self.nsamples // 1000) + 1)

        for n, s, e in zip(num, starts, ends):

            x = self.gen_sample(mi, n)
            qa[:, s:e] = self.q_estimator(mi, x)

        fisher = np.cov(qa)
        bias = qa.mean(axis=1)

        return fisher, bias


class PSMonteCarloAlt(psestimation.PSEstimation):
    """An extension of the PSEstimation class to support estimation of the
    Fisher matrix via Monte-Carlo simulations.

    This uses a stochastic estimation of the trace which allows us to compute
    a reduced set of products between the four covariance matrices.

    Attributes
    ----------
    nswitch : integer
        The threshold number of eigenmodes above which we switch to Monte-Carlo
        estimation.
    nsamples : integer
        The number of samples to draw from each band.
    """

    nsamples = config.Property(proptype=int, default=500)
    nswitch = config.Property(proptype=int, default=0)  # 200

    def gen_vecs(self, mi):
        """Generate a cache of sample vectors for each bandpower.
        """

        # Delete cache
        self.vec_cache = []

        bt = self.kltrans.beamtransfer
        evals, evecs = self.kltrans.modes_m(mi)
        nbands = len(self.bands) - 1

        # Set of S/N weightings
        cf = (evals + 1.0) ** -0.5

        # Generate random set of Z_2 vectors
        xv = (
            2 * (np.random.rand(evals.size, self.nsamples) <= 0.5).astype(np.float)
            - 1.0
        )

        # Multiply by C^-1 factorization
        xv1 = cf[:, np.newaxis] * xv

        # Project vector from eigenbasis into telescope basis
        xv2 = np.dot(evecs.T.conj(), xv1).reshape(bt.ndof(mi), self.nsamples)

        # Project back into sky basis
        xv3 = self.kltrans.beamtransfer.project_vector_svd_to_sky(
            mi, xv2, conj=True, temponly=True
        )

        for bi in range(nbands):

            # Product with sky covariance C_l(z, z')
            xv4 = np.zeros_like(xv3)
            for li in range(self.telescope.lmax + 1):
                xv4[:, 0, li, :] = np.dot(
                    self.clarray[bi][0, 0, li], xv3[:, 0, li, :]
                )  # TT only.

            # Projection from sky back into SVD basis
            xv5 = self.kltrans.beamtransfer.project_vector_sky_to_svd(
                mi, xv4, temponly=True
            )

            # Projection into eigenbasis
            xv6 = np.dot(evecs, xv5.reshape(bt.ndof(mi), self.nsamples))
            xv7 = cf[:, np.newaxis] * xv6

            # Push set of vectors into cache.
            self.vec_cache.append(xv7)

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

        fisher = np.zeros((self.nbands, self.nbands), dtype=np.complex128)
        bias = np.zeros(self.nbands, dtype=np.complex128)

        self.gen_vecs(mi)

        ns = self.nsamples

        for ia in range(self.nbands):
            # Estimate diagonal elements (including bias correction)
            va = self.vec_cache[ia]

            fisher[ia, ia] = np.sum(va * va.conj()) / ns

            # Estimate diagonal elements
            for ib in range(ia):
                vb = self.vec_cache[ib]

                fisher[ia, ib] = np.sum(va * vb.conj()) / ns
                fisher[ib, ia] = np.conj(fisher[ia, ib])

        return fisher, bias


class PSMonteCarloLarge(PSMonteCarlo):
    """An extension of the PSEstimation class to support estimation of the
    Fisher matrix via Monte-Carlo simulations for very large arrays.

    Attributes
    ----------
    nsamples : integer
        The number of samples to draw from each band.
    """

    nsamples = config.Property(proptype=int, default=500)

    # ==== Calculate the total Fisher matrix/bias =======

    def make_clzz_array(self):
        """Calculate the response of the angular powerspectrum to each power spectrum band using an MPI array distributed over the band axis.

        Uses the lmax and frequencies from the telescope object.
        """
        nbands = self.nbands
        nfreq = self.telescope.nfreq
        lmax = self.telescope.lmax

        self.clarray = mpiarray.MPIArray(
            (nbands, lmax + 1, nfreq, nfreq),
            axis=0,
            dtype=np.float64,
            comm=MPI.COMM_WORLD,
        )

        self.clarray[:] = 0.0

        for bl, bg in self.clarray.enumerate(axis=0):
            self.clarray[bl] = self.make_clzz(self.band_pk[bg])

    def generate(self, regen=False):
        """Calculate the total Fisher matrix and bias and save to a file.

        Parameters
        ----------
        regen : boolean, optional
            Force regeneration if products already exist (default `False`).
        """
        if mpiutil.rank0:
            st = time.time()
            print("======== Starting PS calculation LARGE ========")

        ffile = self.psdir + "/fisher.hdf5"

        if os.path.exists(ffile) and not regen:
            print("Fisher matrix file: %s exists. Skipping..." % ffile)
            return

        mpiutil.barrier()

        # Pre-compute all the angular power spectra for the bands.
        self.genbands()

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Split up all m-modes such that each MPI process receives one m-mode
        m_chunks, low_bound, upp_bound = self.split_single(
            (self.telescope.mmax + 1), size
        )
        num_chunks = m_chunks.shape[0]

        n = self.nsamples

        # Create an MPI array for the q-estimator distributed over bands
        self.qa = mpiarray.MPIArray(
            (self.telescope.mmax + 1, self.nbands, n),
            axis=1,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
        )

        self.qa[:] = 0.0

        # This loop is designed such that the total number of m's is split up in chunks.
        # Each chunk contains as many m's as there are ranks such that each rank gets exactly one m-mode.
        for ci, num_m in enumerate(m_chunks):
            if mpiutil.rank0:
                print("Starting chunk %i of %i" % (ci, num_chunks))

            loc_num, loc_start, loc_end = mpiutil.split_local(num_m)
            mi = np.arange(low_bound[ci], upp_bound[ci])[loc_start:loc_end]

            # At last iteration of m's, it is necessary to assign m's to remaining ranks otherwise the MPI communication crashes.
            if len(mi) == 0:
                mi = np.array([low_bound[ci] + rank])

            if len(mi) != 0:
                print("Processing m-mode %i on rank %i" % (mi, rank))

            nfreq = self.telescope.nfreq
            lmax = self.telescope.lmax

            if loc_num > 0:
                if self.num_evals(mi) > 0:
                    # Generate random KL data
                    x = self.gen_sample(mi, n)
                    vec1 = self.project_vector_kl_to_sky(mi, x)
                    # Make array contiguous
                    vec1 = np.ascontiguousarray(vec1.reshape(nfreq, lmax + 1, n))

                # If I don't have evals - return zero vector. Each MPI process must have some data to pass around. Otherwise send and receive won't work.
                else:
                    vec1 = np.zeros((nfreq, lmax + 1, n), dtype=np.complex128)

            else:
                vec1 = np.zeros((nfreq, lmax + 1, n), dtype=np.complex128)

            if mpiutil.rank0:
                st_q = time.time()

            # Loop over total number of ranks given by `size`.
            for ir in range(size):
                # Only fill qa if we haven't reached mmax
                if mi < (self.telescope.mmax + 1):
                    self.q_estimator(mi, vec1)

                # We do the MPI communications between ranks only (size - 1) times
                if ir == (size - 1):
                    break

                recv_buffer = self.recv_send_data(vec1, axis=-1)
                vec1 = recv_buffer

                # Send data to (rank + 1) % size, hence local mi must be updated to (mi -1) % size
                mi = low_bound[ci] + (mi - 1) % size

            if mpiutil.rank0:
                et_q = time.time()
                print(
                    "Time needed for quadratic estimation on all ranks for this m-chunk",
                    et_q - st_q,
                    )

        if mpiutil.rank0:
            et_allm = time.time()
            print(
                "Time needed for quadratic estimation on all ranks for all m-chunks ",
                et_allm - st,
                )

        # Once done with all the m's, redistribute qa array over m's
        self.qa = self.qa.redistribute(axis=0)

        # Make an array for local fisher an bias
        fisher_loc = np.zeros((self.nbands, self.nbands), dtype=np.float64)
        bias_loc = np.zeros((self.nbands,), dtype=np.float64)

        # Calculate fisher for each m
        for ml, mg in self.qa.enumerate(axis=0):
            fisher_m = np.cov(self.qa[ml])
            bias_m = np.mean(self.qa[ml], axis=1)

            # Sum over all local m-modes to get the overall fisher and bias per parallel process
            fisher_loc += fisher_m.real  # be careful with the real
            bias_loc += bias_m.real

        self.fisher = mpiutil.allreduce(fisher_loc, op=MPI.SUM)
        self.bias = mpiutil.allreduce(bias_loc, op=MPI.SUM)

        if mpiutil.rank0:
            et = time.time()
            print("======== Ending PS calculation (time=%f) ========" % (et - st))

        self.write_fisher_file()

    def q_estimator(self, mi, x2, y2=None):
        """Estimate the q-parameters from given data (see paper).

        Parameters
        ----------
        mi : integer
            The m-mode we are calculating for.
        x2, y2 : np.ndarrays[nfreq, lmax+1, num_realisations]
            The vector(s) of data we are estimating from in the sky basis,
            Passing in `y2` different from `x2` is for cross power spectrum calculation.

        Returns
        -------
        qa : np.ndarray[numbands]
            Array of q-parameters. If noise=True then the array is one longer, and the last parameter is the projection against the noise.
        """

        # If y2 is None set it to x2
        if y2 is None:
            y2 = x2

        # if data vector is filled with zeros, return q = 0.0 for this m
        if np.all(x2 == 0) or np.all(y2 == 0):
            self.qa[mi, :] = 0.0

        # If one of the data vectors is empty, return q = 0.0 for this m
        elif x2.shape[0] == 0:
            self.qa[mi, :] = 0.0

        else:
            lside = self.telescope.lmax + 1
            for bi, bg in self.qa.enumerate(axis=1):
                for li in range(lside):
                    lxvec = x2[:, li]
                    lyvec = y2[:, li]
                    self.qa[mi, bi] += np.sum(
                        lyvec.conj()
                        * np.matmul(self.clarray[bi][li].astype(np.complex128), lxvec),
                        axis=0,
                    ).astype(np.float64)

    def recv_send_data(self, data, axis):
        """Send data from one MPI process to the next by using a vector buffer.

        Parameters
        ----------
        data : np.ndarray, complex
            Data vector that is sent to the next rank. Expecting
            a complex data type.
        axis : int
            Axis over which the data is devided into chunks of 4GB.

        Returns
        -------
        recv_buffer : np.ndarray
            The data vector received from the previous rank.
        """

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        shape = data.shape
        dtype = data.dtype

        recv_rank = (rank - 1) % size
        send_rank = (rank + 1) % size

        recv_buffer = np.zeros(shape, dtype=dtype)

        # Need to send in 4GB chunks due to some MPI library.
        message_size = 4 * 2 ** 30.0
        dsize = np.prod(shape) * 16.0
        num_messages = int(np.ceil(dsize / message_size))

        num, sm, em = mpiutil.split_m(shape[axis], num_messages)

        for i in range(num_messages):
            slc = [slice(None)] * len(shape)
            slc[axis] = slice(sm[i], em[i])

            di = np.ascontiguousarray(data[slc], dtype=dtype)
            bi = np.zeros(di.shape, dtype=dtype)

            # Initiate non-blocking receive
            request = comm.Irecv(
                [bi, MPI.DOUBLE_COMPLEX], source=recv_rank, tag=(i * size) + recv_rank
            )
            # Initiate send
            comm.Send([di, MPI.DOUBLE_COMPLEX], dest=send_rank, tag=(i * size) + rank)
            # Wait for receive
            request.Wait()

            # Fill recv buffer with messages
            recv_buffer[slc] = bi

        return recv_buffer

    def split_single(self, m, size):
        """Split number of m elements into chunks of `size`. Also return the lower and upper bounds of `size` elements in m.

        Parameters
        -----------
        m : int
            Number of total elements.
        size: size
            Number of MPI processes running.

        Returns
        -------
        m_chunks : list
            List of chunk sizes.
        low_bound : list
            Lower bound of where chunk starts.
        upp_bound : list
            Upper bound where chunk stops.
        """

        quotient = m // size
        rem = m % size

        if rem != 0:
            m_chunks = np.append((size * np.ones(quotient, dtype=int)), rem)
        else:
            m_chunks = size * np.ones(quotient, dtype=int)

        bound = np.cumsum(np.insert(m_chunks, 0, 0))
        ms = m_chunks.shape[0]

        low_bound = bound[:ms]
        upp_bound = bound[1 : (ms + 1)]

        return m_chunks, low_bound, upp_bound


def sim_skyvec(trans, n):
    """Simulate a set of alm(\nu)'s for a given m.

    Generated as if m=0. For greater m, just ignore entries for l < abs(m).

    Parameters
    ----------
    trans : np.ndarray
        Transfer matrix generated by `block_root` from a a particular C_l(z,z').

    Returns
    -------
    gaussvars : np.ndarray
       Vector of alms.
    """

    lside = trans.shape[0]
    nfreq = trans.shape[1]

    matshape = (lside, nfreq, n)

    gaussvars = (
        np.random.standard_normal(matshape) + 1.0j * np.random.standard_normal(matshape)
    ) / 2.0 ** 0.5

    for i in range(lside):
        gaussvars[i] = np.dot(trans[i], gaussvars[i])

    return gaussvars  # .T.copy()


def block_root(clzz):
    """Calculate the 'square root' of an angular powerspectrum matrix (with
    nulls).
    """

    trans = np.zeros_like(clzz)

    for i in range(trans.shape[0]):
        trans[i] = nputil.matrix_root_manynull(clzz[i], truncate=False)

    return trans
