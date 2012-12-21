import numpy as np

from cylsim import psestimation

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
    
    nsamples = 500

    __config_table_ =   {   'nsamples'  : [ int,    'nsamples'] }


    def __init__(self, *args, **kwargs):

        super(PSMonteCarlo, self).__init__(*args, **kwargs)

        # Add configuration options                
        self.add_config(self.__config_table_)




    def gen_sample(self, mi):
        """Get a set of random samples from the specified band `bi` for a given
        `mi`.
        """

        evals, evecs = self.kltrans.modes_m(mi)

        # Calculate C**(1/2), this is the weight to generate a draw from C
        w = (evals + 1.0)**0.5
    
        # Calculate x
        x = nputil.complex_std_normal((evals.shape[0], self.nsamples)) * w[:, np.newaxis] 

        return x


    def q_estimator(self, mi, vec):


        evals, evecs = self.kltrans.modes_m(mi)

        # Weight by C**-1 (transposes are to ensure broadcast works for 1 and 2d vecs)
        x0 = (vec.T / (evals + 1.0)).T

        # Project back into SVD basis
        x1 = np.dot(evecs.T.conj(), x0)

        # Project back into sky basis
        x2 = self.kltrans.beamtransfer.project_vector_svd_to_sky(mi, x1, conj=True)

        qa = np.zeros((self.nbands + 1,) + vec.shape[1:])

        lside = self.telescope.lmax + 1

        # Calculate q_a for each band
        for bi in range(self.nbands):

            for li in range(lside):

                lvec = x2[:, 0, li]

                qa[bi] += np.sum(lvec.conj() * np.dot(self.clarray[bi][0, 0, li], lvec), axis=0) # TT only.

        # Calculate q_a for noise power (x0^H N x0 = |x0|^2)
        qa[-1] = np.sum(x0 * x0.conj(), axis=0)

        return qa



    def fisher_m(self, mi, retbias=False):

            
        fab = np.zeros((self.nbands, self.nbands), dtype=np.complex128)

        if self.num_evals(mi) > 0:
            print "Making fisher (for m=%i)." % mi

            x = self.gen_sample(mi)
            qa = self.q_estimator(mi, x)
            ft = np.cov(qa)

            fisher = ft[:self.nbands, :self.nbands]
            bias = ft[-1, self.nbands]

        else:
            print "No evals (for m=%i), skipping." % mi

            fisher = np.zeros((self.nbands, self.nbands), dtype=np.complex128)
            bias = np.zeros((self.nbands,), dtype=np.complex128)

        if not retbias:
            return fisher
        else:
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
    
    nsamples = 500
    nswitch = 0 #200

    __config_table_ =   {   'nsamples'  : [ int,    'nsamples'],
                            'nswitch'   : [ int,    'nswitch'],
                        }


    def __init__(self, *args, **kwargs):

        super(PSMonteCarloAlt, self).__init__(*args, **kwargs)

        # Add configuration options                
        self.add_config(self.__config_table_)





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
        xv2 = np.dot(evecs.T.conj(), xv1).reshape(bt.nfreq, bt.ntel, self.nsamples)

        # Get projection matrix from stokes I to telescope
        bp = bt.beam_m(mi)[:, :, :, :, 0, :].reshape(bt.nfreq, bt.ntel, -1)
        lside = bp.shape[-1]

        # Project with transpose B matrix
        xv3 = np.zeros((bt.nfreq, lside, self.nsamples), dtype=np.complex128)
        for fi in range(bt.nfreq):
            xv3[fi] = np.dot(bp[fi].T.conj(), xv2[fi])

        for bi in range(nbands):

            # Product with sky covariance C_l(z, z')
            xv4 = np.zeros_like(xv3)
            for li in range(lside):
                xv4[:, li, :] = np.dot(self.clarray[bi][0, 0, li], xv3[:, li, :]) # TT only.

            # Projection from sky vector into telescope
            xv5 = np.zeros_like(xv2)
            for fi in range(bt.nfreq):
                xv5[fi] = np.dot(bp[fi], xv4[fi])

            # Projection into eigenbasis
            xv6 = np.dot(evecs, xv5.reshape(bt.nfreq * bt.ntel, self.nsamples))
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
