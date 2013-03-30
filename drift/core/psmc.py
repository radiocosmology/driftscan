import os
import time


import numpy as np
import scipy.linalg as la
import h5py

from drift.core import psestimation
from drift.util import mpiutil, config

from cosmoutils import nputil



        




class PSMonteCarlo(psestimation.PSEstimation):
    """An extension of the PSEstimation class to support estimation of the
    Fisher matrix via Monte-Carlo simulations.

    This uses the fact that the covariance of the q-estimator is the Fisher
    matrix to Monte-Carlo the Fisher matrix and the bias.

    Attributes
    ----------
    nsamples : integer
        The number of samples to draw from each band.
    """
    
    nsamples = config.Property(proptype=int, default=500)

    fisher = None
    bias = None

    zero_mean = config.Property(proptype=bool, default=True)


    def gen_sample(self, mi):
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

        evals, evecs = self.kltrans.modes_m(mi)

        # Calculate C**(1/2), this is the weight to generate a draw from C
        w = (evals + 1.0)**0.5
    
        # Calculate x
        x = nputil.complex_std_normal((evals.shape[0], self.nsamples)) * w[:, np.newaxis] 

        return x


    def q_estimator(self, mi, vec, noise=False):
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
        x0 = (vec.T / (evals + 1.0)).T

        # Project back into SVD basis
        x1 = np.dot(evecs.T.conj(), x0)

        # Project back into sky basis
        x2 = self.kltrans.beamtransfer.project_vector_svd_to_sky(mi, x1, conj=True)

        # Create empty q vector (length depends on if we're calculating the noise term too)
        qa = np.zeros((self.nbands + 1 if noise else self.nbands,) + vec.shape[1:])

        lside = self.telescope.lmax + 1

        # Calculate q_a for each band
        for bi in range(self.nbands):

            for li in range(lside):

                lvec = x2[:, 0, li]

                qa[bi] += np.sum(lvec.conj() * np.dot(self.clarray[bi][0, 0, li], lvec), axis=0) # TT only.

        # Calculate q_a for noise power (x0^H N x0 = |x0|^2)
        if noise:
            if self.zero_mean:
                qa[-1] = np.sum((x0 * x0.conj()).T * (evals + 1.0), axis=-1)
            else:
                qa[-1] = np.sum(x0 * x0.conj(), axis=0)

        return qa



    def fisher_m(self, mi, retbias=True):

            
        fab = np.zeros((self.nbands, self.nbands), dtype=np.complex128)

        if self.num_evals(mi) > 0:
            print "Making fisher (for m=%i)." % mi

            x = self.gen_sample(mi)
            qa = self.q_estimator(mi, x, noise=True)
            ft = np.cov(qa)

            fisher = ft[:self.nbands, :self.nbands]
            bias = ft[-1, :self.nbands]

        else:
            print "No evals (for m=%i), skipping." % mi

            fisher = np.zeros((self.nbands, self.nbands), dtype=np.complex128)
            bias = np.zeros((self.nbands,), dtype=np.complex128)

        if not retbias:
            return fisher
        else:
            return fisher, bias




    def generate(self, mlist = None, regen=False):
        """Generate the Fisher matrix and bias required for
        forecasting and powerspectrum estimation.

        Parameters
        ----------
        mlist : array_like
            Restricted set of m's to compute for. If None (default) use all
            m's.

        regen : boolean
            If True force recalculation over. Default is False.
        """

        if mpiutil.rank0:
            st = time.time()
            print "======== Starting PS calculation ========"


        if mlist is None:
            mlist = range(self.telescope.mmax + 1)

        ffile = self.psdir +'fisher.hdf5'

        if os.path.exists(ffile) and not regen:
            print ("Fisher matrix file: %s exists. Skipping..." % ffile)
            return

        mpiutil.barrier()

        self.genbands()

        # Use parallel map to distribute Fisher calculation
        fisher_bias = mpiutil.parallel_map(self.fisher_m, mlist)

        # Unpack into separate lists of the Fisher matrix and bias
        fisher, bias = zip(*fisher_bias)

        # Sum over all m-modes to get the over all Fisher and bias
        self.fisher = np.sum(np.array(fisher), axis=0).real # Be careful of the .real here
        self.bias = np.sum(np.array(bias), axis=0).real # Be careful of the .real here


        if mpiutil.rank0:
            et = time.time()
            print "======== Ending PS calculation (time=%f) ========" % (et - st)

            f = h5py.File(self.psdir + '/fisher.hdf5', 'w')

            cv = la.inv(self.fisher)
            err = cv.diagonal()**0.5
            cr = cv / np.outer(err, err)

            f.create_dataset('fisher/', data=self.fisher)
            f.create_dataset('bias/', data=self.bias)
            f.create_dataset('covariance/', data=cv)
            f.create_dataset('error/', data=err)
            f.create_dataset('correlation/', data=cr)


            f.create_dataset('bandpower/', data=self.bpower)
            f.create_dataset('bandstart/', data=self.bstart)
            f.create_dataset('bandend/', data=self.bend)
            f.create_dataset('bandcenter/', data=self.bcenter)
            f.create_dataset('psvalues/', data=self.psvalues)
            f.close()


    def fisher_bias(self):

        with h5py.File(self.psdir + '/fisher.hdf5', 'r') as f:

            return f['fisher'][:], f['bias'][:]
        


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
    nswitch = config.Property(proptype=int, default=0) #200


    def gen_vecs(self, mi):
        """Generate a cache of sample vectors for each bandpower.
        """

        # Delete cache
        self.vec_cache = []

        bt = self.kltrans.beamtransfer
        evals, evecs = self.kltrans.modes_m(mi)
        nbands = len(self.bands) - 1

        # Set of S/N weightings
        cf = (evals + 1.0)**-0.5

        # Generate random set of Z_2 vectors
        xv = 2*(np.random.rand(evals.size, self.nsamples) <= 0.5).astype(np.float) - 1.0

        # Multiply by C^-1 factorization
        xv1 = cf[:, np.newaxis] * xv

        # Project vector from eigenbasis into telescope basis
        xv2 = np.dot(evecs.T.conj(), xv1).reshape(bt.ndof(mi), self.nsamples)

        # Project back into sky basis
        xv3 = self.kltrans.beamtransfer.project_vector_svd_to_sky(mi, xv2, conj=True, temponly=True)

        for bi in range(nbands):

            # Product with sky covariance C_l(z, z')
            xv4 = np.zeros_like(xv3)
            for li in range(self.telescope.lmax + 1):
                xv4[:, 0, li, :] = np.dot(self.clarray[bi][0, 0, li], xv3[:, 0, li, :]) # TT only.

            # Projection from sky back into SVD basis
            xv5 = self.kltrans.beamtransfer.project_vector_sky_to_svd(mi, xv4, temponly=True)

            # Projection into eigenbasis
            xv6 = np.dot(evecs, xv5.reshape(bt.ndof(mi), self.nsamples))
            xv7 = cf[:, np.newaxis] * xv6

            # Push set of vectors into cache.
            self.vec_cache.append(xv7)



    def fisher_m_mc(self, mi):
        """Calculate the Fisher Matrix by Monte-Carlo.
        """
            
        fab = np.zeros((self.nbands, self.nbands), dtype=np.complex128)

        if self.num_evals(mi) > 0:
            print "Making fisher (for m=%i)." % mi

            self.gen_vecs(mi)

            ns = self.nsamples

            for ia in range(self.nbands):
                # Estimate diagonal elements (including bias correction)
                va = self.vec_cache[ia]

                fab[ia, ia] = np.sum(va * va.conj()) / ns

                # Estimate diagonal elements
                for ib in range(ia):
                    vb = self.vec_cache[ib]

                    fab[ia, ib] = np.sum(va * vb.conj()) / ns
                    fab[ib, ia] = np.conj(fab[ia, ib])
            
        else:
            print "No evals (for m=%i), skipping." % mi

        return fab


    def fisher_m(self, mi):
        """Calculate the Fisher Matrix for a given m.

        Decides whether to use direct evaluation or Monte-Carlo depending on the
        number of eigenvalues required.
        """
        if self.num_evals(mi) < self.nswitch:
            return super(PSMonteCarlo, self).fisher_m(mi)
        else:
            return self.fisher_m_mc(mi)
        




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

    gaussvars = (np.random.standard_normal(matshape)
                 + 1.0J * np.random.standard_normal(matshape)) / 2.0**0.5

    for i in range(lside):
        gaussvars[i] = np.dot(trans[i], gaussvars[i])

    return gaussvars   #.T.copy()
        

def block_root(clzz):
    """Blah.
    """

    trans = np.zeros_like(clzz)

    for i in range(trans.shape[0]):
        trans[i] = nputil.matrix_root_manynull(clzz[i], truncate=False)

    return trans
