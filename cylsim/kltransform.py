import time

import numpy as np
import scipy.linalg as la

import h5py
import healpy


from simulations.foregroundmap import matrix_root_manynull

from cylsim import mpiutil
from cylsim import beamtransfer
from cylsim import util
from cylsim import skymodel

from simulations import foregroundsck, corr21cm
from utils import units

def eigh_gen(A, B):
    """Solve the generalised eigenvalue problem. :math:`\mathbf{A} \mathbf{v} =
    \lambda \mathbf{B} \mathbf{v}`
    
    This routine will attempt to correct for when `B` is not positive definite
    (usually due to numerical precision), by adding a constant diagonal to make
    all of its eigenvalues positive.
    
    Parameters
    ----------
    A, B : np.ndarray
        Matrices to operate on.
        
    Returns
    -------
    evals : np.ndarray
        Eigenvalues of the problem.
    evecs : np.ndarray
        2D array of eigenvectors (packed column by column).
    add_const : scalar
        The constant added on the diagonal to regularise.
    """
    add_const = 0.0
    
    try:
        evals, evecs = la.eigh(A, B, overwrite_a=True, overwrite_b=True)
    except la.LinAlgError:
        
        print "Matrix probabaly not positive definite due to numerical issues. \
        Trying to a constant...."
        
        add_const = -la.eigvalsh(B, eigvals=(0, 0))[0] * 1.1
        
        B[np.diag_indices(B.shape[0])] += add_const
        evals, evecs = la.eigh(A, B, overwrite_a=True, overwrite_b=True)
        
    return evals, evecs, add_const



class KLTransform(object):
    """Perform KL transform.
    """

    subset = True
    threshold = 1.0

    directory = ""

    @property
    def _evfile(self):
        # Pattern to form the `m` ordered file.
        return self.beamtransfer.directory + "/" + self.evsubdir + "/ev_m_" + util.intpattern(self.telescope.mmax) + ".hdf5"
    

    def __init__(self, bt, evsubdir = "ev"):
        self.beamtransfer = bt
        self.telescope = self.beamtransfer.telescope

        self.evsubdir = evsubdir



    def foreground(self):
        """Compute the foreground covariance matrix.

        Returns
        -------
        cv_fg : np.ndarray[pol2, pol1, l, freq1, freq2]
        """

        npol = self.telescope.num_pol_sky

        if npol != 1 and npol != 3:
            raise Exception("Can only handle unpolarised only (num_pol_sky = 1), or I, Q and U (num_pol_sky = 3).")
        
        cv_fg = skymodel.foreground_model(self.telescope.lmax, self.telescope.frequencies, npol)
        return cv_fg


    def signal(self):
        """Compute the signal covariance matrix.

        Returns
        -------
        cv_fg : np.ndarray[pol2, pol1, l, freq1, freq2]
        """
        npol = self.telescope.num_pol_sky

        if npol != 1 and npol != 3:
            raise Exception("Can only handle unpolarised only (num_pol_sky = 1), or I, Q and U (num_pol_sky = 3).")
        
        cv_sg = skymodel.im21cm_model(self.telescope.lmax, self.telescope.frequencies, npol)
        return cv_sg


    def signal_covariance(self, mi):
        
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        npol = self.telescope.num_pol_sky
        nfreq = self.telescope.nfreq
        lside = self.telescope.lmax + 1

        beam = self.beamtransfer.beam_m(mi).reshape((nfreq, ntel, npol, lside))
        
        cvb_s = np.zeros((nfreq, ntel, nfreq, ntel), dtype=np.complex128)
        cvb_n = np.zeros_like(cvb_s)

        cv_sg = self.signal()
        cv_fg = self.foreground()

        for fi in range(nfreq):
            for fj in range(nfreq):
                for pi in range(npol):
                    for pj in range(npol):
                        cvb_n[fi, :, fj, :] = np.dot((beam[fi, :, pi, :] * cv_fg[..., fi, fj]), beam[fj, :, pj, :].T.conj())
                        cvb_s[fi, :, fj, :] = np.dot((beam[fi, :, pi, :] * cv_sg[..., fi, fj]), beam[fj, :, pj, :].T.conj())
            
            noisebase = np.diag(self.telescope.noisepower(np.arange(self.telescope.nbase), fi).reshape(ntel))
            cvb_n[fi, :, fi, :] += noisebase

        return cvb_s, cvb_n


    def transform_m(self, mi):
        
        print "Solving for Eigenvalues...."

        st = time.time()

        nside = self.telescope.nbase * self.telescope.num_pol_telescope * self.telescope.nfreq
        
        cvb_sr, cvb_nr = [cv.reshape(nside, nside) for cv in self.signal_covariance(mi)]
        
        et = time.time()
        print "Time =", (et-st)

        st = time.time()

        res = eigh_gen(cvb_sr, cvb_nr)
        
        et=time.time()
        print "Time =", (et-st)

        return res


    def transform_all(self):

        # Iterate list over MPI processes.
        for mi in mpiutil.mpirange(-self.telescope.mmax, self.telescope.mmax+1):
            #for mi in [-100]:
            
            st = time.time()
            
            print "Constructing signal and noise covariances for m = %i ..." % (mi)
            evals, evecs, ac = self.transform_m(mi)
            
            ## Write out Eigenvals and Vectors
            print "Creating file %s ...." % (self._evfile % mi)
            f = h5py.File(self._evfile % mi, 'w')
            f.attrs['m'] = mi
            f.attrs['SUBSET'] = self.subset

            if self.subset:
                i_ev = np.searchsorted(evals, self.threshold)

                f.create_dataset('evals_full', data=evals)
                evalsf = evals

                evals = (evals[:i_ev])[::-1]
                evecs = (evecs[:, i_ev:])[::-1]
                print "Modes with S/N > %f: %i of %i" % (self.threshold, evals.size, evalsf.size)



            f.create_dataset('evals', data=evals)
            f.create_dataset('evecs', data=evecs.T, compression='gzip')
                
            if ac != 0.0:
                f.attrs['add_const'] = ac
                f.attrs['FLAGS'] = 'NotPositiveDefinite'
            else:
                f.attrs['FLAGS'] = 'Normal'
                
            f.close()





