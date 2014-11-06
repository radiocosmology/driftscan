import os

import numpy as np
import h5py

from drift.core import kltransform
from drift.util import mpiutil, config


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

    foreground_threshold = config.Property(proptype=float, default=100.0)
    
    def _transform_m(self, mi):

        inv = None

        nside = self.beamtransfer.ndof(mi)

        # Ensure that number of SVD degrees of freedom is non-zero before proceeding
        if nside == 0:
            return np.array([]), np.array([[]]), np.array([[]]), { 'ac' : 0.0, 'f_evals' : np.array([]) }

        # Construct S and F matrices and regularise foregrounds
        self.use_thermal = False
        cs, cn = [ cv.reshape(nside, nside) for cv in self.sn_covariance(mi) ]
        
        # Find joint eigenbasis and transformation matrix
        evals, evecs2, ac, inv = kltransform.eigh_gen(cs, cn, invert=self.inverse)
        evecs = evecs2.T.conj()

        if self.inverse:
            inv.imag *= -1   # conjugate but no transpose

        # Get the indices that extract the high S/F ratio modes
        ind = np.where(evals > self.foreground_threshold)

        # Construct evextra dictionary (holding foreground ratio)
        evextra = { 'ac' : ac, 'f_evals' : evals.copy() }

        # Construct the foreground removed subset of the space
        evals = evals[ind]
        evecs = evecs[ind]
        inv = inv[ind] if self.inverse else None

        if evals.size > 0:
            # Generate the full S and N covariances in the truncated basis
            self.use_thermal = True
            cs, cn = [ cv.reshape(nside, nside) for cv in self.sn_covariance(mi) ]
            cs = np.dot(evecs, np.dot(cs, evecs.T.conj()))
            cn = np.dot(evecs, np.dot(cn, evecs.T.conj()))

            # Find the eigenbasis and the transformation into it.
            evals, evecs2, ac, inv2 = kltransform.eigh_gen(cs, cn, invert=self.inverse)
            evecs = np.dot(evecs2.T.conj(), evecs)

            # Construct the inverse if required.
            if self.inverse:
                inv2.imag *= -1    # conjugate but no transpose
                inv = np.dot(inv2, inv)

        return evals, evecs, inv, evextra


    def _ev_save_hook(self, f, evextra, context=None):
        kltransform.KLTransform._ev_save_hook(self, f, evextra, context)

        # Save out S/F ratios
        rank0 = (context is None) or (context.mpi_comm.rank == 0)
        if rank0:
            f.create_dataset('f_evals', data=evextra['f_evals'])


    def _collect(self):
        
        def evfunc(mi):


            ta = np.zeros(shape, dtype=np.float64)

            f = h5py.File(self._evfile % mi, 'r')

            if f['evals_full'].shape[0] > 0:
                ev = f['evals_full'][:]
                fev = f['f_evals'][:]
                ta[0, -ev.size:] = ev
                ta[1, -fev.size:] = fev

            f.close()

            return ta

        if mpiutil.rank0:
            print "Creating eigenvalues file (process 0 only)."
        
        mlist = range(self.telescope.mmax+1)
        shape = (2, self.beamtransfer.ndofmax)
        
        evarray = kltransform.collect_m_array(mlist, evfunc, shape, np.float64)
        
        if mpiutil.rank0:
            if os.path.exists(self.evdir + "/evals.hdf5"):
                print "File: %s exists. Skipping..." % (self.evdir + "/evals.hdf5")
                return

            f = h5py.File(self.evdir + "/evals.hdf5", 'w')
            f.create_dataset('evals', data=evarray[:, 0])
            f.create_dataset('f_evals', data=evarray[:, 1])
            f.close()

