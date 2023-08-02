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
        """Generate a cache of sample vectors for each bandpower."""

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


def sim_skyvec(trans, n):
    """Simulate a set of :math:`alm(\\nu)` 's for a given :math:`m`.

    Generated as if :math:`m=0`. For greater :math:`m`, just ignore entries for :math:`l < abs(m)`.

    Parameters
    ----------
    trans : np.ndarray
        Transfer matrix generated by `block_root` from a a particular :math:`C_l(z,z\')`.

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
    ) / 2.0**0.5

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
