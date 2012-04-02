import os

import numpy as np
import h5py

from cylsim import mpiutil
from cylsim import kltransform
from cylsim import blockla


def svd_trans(bp, thr=1e-15):
    u, s, vh = blockla.svd_dm(bp, full_matrices=False)

    ts = s.max() * thr
    ns = (s > ts).sum()

    ta = np.zeros((ns, bp.shape[0], bp.shape[1]), dtype=np.complex128)

    ci = 0
    for i in range(bp.shape[0]):
        lns = (s[i] > ts).sum()
        ta[ci:(ci+lns), i] = u[i].T.conj()[:lns]

        ci = ci + lns

    return ta.reshape((ns, -1))


class DoubleKL(kltransform.KLTransform):
    """Modified KL technique that performs a first transformation to remove
    foreground modes, and a subsequent transformation to diagonalise the full
    noise (remaining foregrounds+instrumental space).

    Attributes
    ----------
    foreground_threshold : scalar
        Ratio of S/F power below which we throw away modes as being foreground
        contaminated.
    """

    foreground_threshold = 100.0
    
    def _transform_m(self, mi):

        inv = None

        nside = self.beamtransfer.ntel * self.telescope.nfreq

        # Construct S and F matrices and regularise foregrounds
        cs, cn = [ cv.reshape(nside, nside) for cv in self.sn_covariance(mi, noise=False) ]
        
        # Find joint eigenbasis and transformation matrix
        evals, evecs2, ac = kltransform.eigh_gen(cs, cn)
        evecs = evecs2.T.conj()

        # Get the indices that extract the high S/F ratio modes
        ind = np.where(evals > self.foreground_threshold)

        # Construct evextra dictionary (holding foreground ratio)
        evextra = { 'ac' : ac, 'f_evals' : evals.copy() }

        # Construct inverse transformation if required
        if self.inverse:
            inv = kltransform.inv_gen(evecs).T

        # Construct the foreground removed subset of the space
        evals = evals[ind]
        evecs = evecs[ind]
        inv = inv[ind] if self.inverse else None

        if evals.size > 0:
            # Generate the full S and N covariances in the truncated basis
            cs, cn = [ cv.reshape(nside, nside) for cv in self.sn_covariance(mi) ]
            cs = np.dot(evecs, np.dot(cs, evecs.T.conj()))
            cn = np.dot(evecs, np.dot(cn, evecs.T.conj()))

            # Find the eigenbasis and the transformation into it.
            evals, evecs2, ac = kltransform.eigh_gen(cs, cn)
            evecs = np.dot(evecs2.T.conj(), evecs)

            # Construct the inverse if required.
            if self.inverse:
                inv2 = kltransform.inv_gen(evecs2)
                inv = np.dot(inv2, inv)

        return evals, evecs, inv, evextra


    def _ev_save_hook(self, f, evextra):

        kltransform.KLTransform._ev_save_hook(self, f, evextra)

        # Save out S/F ratios
        f.create_dataset('f_evals1', data=evextra['f_evals'])


    def _collect(self):
        
        evfunc = lambda mi: h5py.File(self._evfile % mi, 'r')['evals_full'][:]
        fvfunc = lambda mi: h5py.File(self._evfile % mi, 'r')['f_evals'][:]

        if mpiutil.rank0:
            print "Creating eigenvalues file (process 0 only)."
        
        mlist = range(self.telescope.mmax+1)
        shape = (self.beamtransfer.ntel * self.telescope.nfreq, )
        
        evarray = kltransform.collect_m_array(mlist, evfunc, shape, np.float64)
        fvarray = kltransform.collect_m_array(mlist, fvfunc, shape, np.float64)
        
        if mpiutil.rank0:
            if os.path.exists(self.evdir + "/evals.hdf5"):
                print "File: %s exists. Skipping..." % (self.evdir + "/evals.hdf5")
                return

            f = h5py.File(self.evdir + "/evals.hdf5", 'w')
            f.create_dataset('evals', data=evarray)
            f.create_dataset('f_evals1', data=fvarray)
            f.close()

