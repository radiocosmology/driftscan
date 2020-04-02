# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np

from cora.util import nputil

from caput import mpiutil, config

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

    def gen_sample(self, mi, nsamples=None, noiseonly=False):
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
        w = np.ones_like(evals) if noiseonly else (evals + 1.0) ** 0.5

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

        ft = np.cov(qa)

        fisher = np.cov(qa)  # ft[:self.nbands, :self.nbands]
        # bias = ft[-1, :self.nbands]
        bias = qa.mean(axis=1)  # [:self.nbands]

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


class PSMonteCarloXLarge(psestimation.PSEstimation, PSMonteCarlo):
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
        """Store clarray for power spectrum bands in parallel with MPIArray."""

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
            print("======== Starting PS calculation ========")

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

        st = time.time()

        m_chunks, low_bound, upp_bound = self.split_single(
            (self.telescope.mmax + 1), size
        )
        num_chunks = m_chunks.shape[0]

        if mpiutil.rank0:
            print(
                "M chunks, lower bounds, upper bounds", m_chunks, low_bound, upp_bound
            )

        n = self.nsamples

        # This is designed such that at each iteration one rank gets one m-mode.
        for ci, num_m in enumerate(m_chunks):
            if mpiutil.rank0:
                print("Starting chunk %i of %i" % (ci, num_chunks))

            loc_num, loc_start, loc_end = mpiutil.split_local(num_m)
            mi = np.arange(low_bound[ci], upp_bound[ci])[loc_start:loc_end]

            # At last iteration of m's, it is necessary to assign m's to remaining ranks otherwise the MPI communication crashes.
            if len(mi) != 0:
                print("Processing m-mode %i on rank %i" % (mi, rank))

            if len(mi) == 0:
                mi = np.array([low_bound[ci] + rank])

            nfreq = self.telescope.nfreq
            lmax = self.telescope.lmax

            if loc_num > 0:
                if self.num_evals(mi) > 0:
                    # Generate random KL data
                    x = self.gen_sample(mi, n)
                    vec1, vec2 = self.project_vector_kl_to_sky(mi, x)
                    # Make array contiguous
                    vec1 = np.ascontiguousarray(
                        vec1.reshape(loc_num, nfreq, lmax + 1, n)
                    )
                    vec2 = np.ascontiguousarray(
                        vec2.reshape(loc_num, nfreq, lmax + 1, n)
                    )

                # If I don't have evals - return zero vector
                else:
                    vec1 = np.zeros((loc_num, nfreq, lmax + 1, n), dtype=np.complex128)
                    vec2 = vec1

            else:
                vec1 = np.zeros((1, nfreq, lmax + 1, n), dtype=np.complex128)
                vec2 = vec1

            self.qa = mpiarray.MPIArray(
                (self.telescope.mmax + 1, self.nbands, n),
                axis=1,
                comm=MPI.COMM_WORLD,
                dtype=np.float64,
            )

            self.qa[:] = 0.0

            et = time.time()

            dsize = np.prod(vec1.shape)

            for ir in range(size):
                st_ir = time.time()
                # Only fill qa if we haven't reached mmax
                if mi < (self.telescope.mmax + 1):
                    self.q_estimator(mi, vec1, vec2)

                etq = time.time()
                print("Time needed for calculating qa one round", etq - st_ir)

                # We do the MPI communications only (size - 1) times
                if ir == (size - 1):
                    break

                recv_buffer = self.recv_send_data(vec1, axis=-1)
                vec1 = recv_buffer

                # Sent data to (rank + 1) % size, hence local mi must be updated to (mi -1) % size
                mi = low_bound[ci] + (mi - 1) % size

            etallq = time.time()
            print("Time needed for qa calculation all ranks ", etallq - et)

        em = time.time()
        print("Time needed for qa calculation all m chunks ", em - et)

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

        self.write_fisher_file()

    def q_estimator(self, mi, vec1, vec2, noise=False):
        """Calculate the quadratic estimator for this mi with data vec1 and vec2"""
        lside = self.telescope.lmax + 1

        for bi, bg in self.qa.enumerate(axis=1):
            for li in range(lside):
                lxvec = vec1[:, :, li]
                lyvec = vec2[:, :, li]
                self.qa[mi, bi, :] += np.sum(
                    lyvec.conj()
                    * np.matmul(self.clarray[bi][li].astype(np.complex128), lxvec),
                    axis=1,
                ).astype(np.float64)

        # Calculate q_a for noise power (x0^H N x0 = |x0|^2)
        if noise:
            # If calculating crosspower don't include instrumental noise
            noisemodes = 0.0 if self.crosspower else 1.0
            evals, evecs = self.kltrans.modes_m(mi)
            noisemodes = noisemodes + (evals if self.zero_mean else 0.0)

            qa.global_slice[mi, -1] = np.sum(
                (vec1 * vec2.conj()).T.real * noisemodes, axis=-1
            )

    def recv_send_data(self, data, axis):

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
        print("Number of messages: %i" % num_messages)
        # If dsize =0.0 you get 0 num_messages and it throws error. hack.
        if num_messages == 0:
            num_messages = 1

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
