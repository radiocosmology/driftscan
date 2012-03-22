import numpy as np

from cylsim import kltransform

class DoubleKL(kltransform.KLTransform):


    foreground_threshold = 100.0

    def transform_m(self, mi):

        nside = self.telescope.nbase * self.telescope.num_pol_telescope * self.telescope.nfreq
        cs, cn = [ cv.reshape(nside, nside) for cv in self.sn_covariance(mi, noise=False) ]

        ## Solve for *** F/S *** ratio, much more numerically well behaved
        evals, evecs, ac = kltransform.eigh_gen(cn, cs)

        i_ev = np.searchsorted(evals, 1.0 / self.foreground_threshold)
        evals = evals[:i_ev]
        evecs = evecs[:, :i_ev]

        cs, cn = [ cv.reshape(nside, nside) for cv in self.sn_covariance(mi) ]

        cs = np.identity(i_ev, dtype=np.complex128)
        cn = np.dot(evecs.T.conj(), np.dot(cn, evecs))

        evals2, evecs2, ac = kltransform.eigh_gen(cs, cn)

        return evals2, np.dot(evecs, evecs2), ac


