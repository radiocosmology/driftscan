import time
import os

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

    evdir = ""

    _cvfg = None
    _cvsg = None

    @property
    def _evfile(self):
        # Pattern to form the `m` ordered file.
        return self.evdir + "/ev_m_" + util.intpattern(self.telescope.mmax) + ".hdf5"
    

    def __init__(self, bt, evsubdir = "ev"):
        self.beamtransfer = bt
        self.telescope = self.beamtransfer.telescope
                
        # Create directory if required
        self.evdir = self.beamtransfer.directory + "/" + evsubdir
        if mpiutil.rank0 and not os.path.exists(self.evdir):
            os.makedirs(self.evdir)



    def foreground(self):
        """Compute the foreground covariance matrix.

        Returns
        -------
        cv_fg : np.ndarray[pol2, pol1, l, freq1, freq2]
        """

        if self._cvfg is None:

            npol = self.telescope.num_pol_sky

            if npol != 1 and npol != 3:
                raise Exception("Can only handle unpolarised only (num_pol_sky = 1), or I, Q and U (num_pol_sky = 3).")
            
            self._cvfg = skymodel.foreground_model(self.telescope.lmax, self.telescope.frequencies, npol)

        return self._cvfg


    def signal(self):
        """Compute the signal covariance matrix.

        Returns
        -------
        cv_fg : np.ndarray[pol2, pol1, l, freq1, freq2]
        """
        
        if self._cvsg is None:
            npol = self.telescope.num_pol_sky

            if npol != 1 and npol != 3:
                raise Exception("Can only handle unpolarised only (num_pol_sky = 1), or I, Q and U (num_pol_sky = 3).")
        
            self._cvsg = skymodel.im21cm_model(self.telescope.lmax, self.telescope.frequencies, npol)

        return self._cvsg


    def signal_covariance(self, mi):
        
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        nfreq = self.telescope.nfreq

        cvb_s = self.beamtransfer.project_matrix_forward(mi, self.signal())
        cvb_n = self.beamtransfer.project_matrix_forward(mi, self.foreground())

        for fi in range(nfreq):
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




    def transform_save(self, mi):
        st = time.time()
        
        print "Constructing signal and noise covariances for m = %i ..." % (mi)
        evals, evecs, ac = self.transform_m(mi)
    
        ## Write out Eigenvals and Vectors
        print "Creating file %s ...." % (self._evfile % mi)
        f = h5py.File(self._evfile % mi, 'w')
        f.attrs['m'] = mi
        f.attrs['SUBSET'] = self.subset

        f.create_dataset('evals_full', data=evals)

        if self.subset:
            i_ev = np.searchsorted(evals, self.threshold)
            
            evalsf = evals
            
            evals = evals[i_ev:]
            evecs = evecs[:, i_ev:]
            print "Modes with S/N > %f: %i of %i" % (self.threshold, evals.size, evalsf.size)
            
        f.create_dataset('evals', data=evals)
        f.create_dataset('evecs', data=evecs.T, compression='gzip')
        
        if ac != 0.0:
            f.attrs['add_const'] = ac
            f.attrs['FLAGS'] = 'NotPositiveDefinite'
        else:
            f.attrs['FLAGS'] = 'Normal'
                
        f.close()

        return evals, evecs.T


    def evals_all(self):

        nside = self.telescope.nbase * self.telescope.num_pol_telescope * self.telescope.nfreq
        evarray = np.zeros((2*self.telescope.mmax+1, nside))
        
        for mi in range(-self.telescope.mmax, self.telescope.mmax+1):

            f = h5py.File(self._evfile % mi, 'r')
            evarray[mi] = f['evals_full']

        return evarray


    def generate(self, mlist = None):

        # Iterate list over MPI processes.
        for mi in mpiutil.mpirange(-self.telescope.mmax, self.telescope.mmax+1):
            self.transform_save(mi)

        if mpiutil.rank0:
            print "Creating eigenvalues file (process 0 only)."
            evals = self.evals_all()

            f = h5py.File(self.evdir + "/evals.hdf5", 'w')
            f.create_dataset('evals', data=evals)
            f.close()


    def modes_m(self, mi):

        if not os.path.exists(self._evfile % mi):
            modes = self.transform_save(mi)
        else:
            f = h5py.File(self._evfile % mi, 'r')
            modes = ( f['evals'][:], f['evecs'][:] )
            f.close()

        return modes

    def skymodes_m(self, mi):

        evals, evecs = self.modes_m(mi)

        nfreq = self.telescope.nfreq
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        nsky = self.telescope.num_pol_sky * (self.telescope.lmax + 1)

        beam = self.beamtransfer.beam_m(mi).reshape((nfreq, ntel, nsky))
        evecs = evecs.reshape((-1, nfreq, ntel)).conj()

        evsky = np.zeros((evecs.shape[0], nfreq, nsky), dtype=np.complex128)
        
        for fi in range(nfreq):
            evsky[:, fi, :] = np.dot(evecs[:, fi, :], beam[fi])

        return evsky
            
        
    def skymodes_m_c(self, mi):

        evals, evecs = self.modes_m(mi)

        nfreq = self.telescope.nfreq
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        nsky = self.telescope.num_pol_sky * (self.telescope.lmax + 1)

        beam = self.beamtransfer.beam_m(mi).reshape((nfreq, ntel, nsky))
        evecs = evecs.reshape((-1, nfreq, ntel)).conj()

        evsky = np.zeros((evecs.shape[0], nfreq, nsky), dtype=np.complex128)
        
        for fi in range(nfreq):
            evsky[:, fi, :] = np.dot(evecs[:, fi, :], beam[fi])

        return evsky
                    


    def project_tel_vector_forward(self, mi, vec):

        evals, evecs = self.modes_m(mi)

        if vec.shape[0] != evecs.shape[1]:
            raise Exception("Vectors are incompatible.")

        return np.dot(evecs, vec)


    def project_sky_vector_forward(self, mi, vec):

        tvec = self.beamtransfer.project_vector_forward(mi, vec).flat

        return self.project_tel_vector_forward(mi, tvec)

    def project_tel_matrix_forward(self, mi, mat):

        evals, evecs = self.modes_m(mi)

        if (mat.shape[0] != evecs.shape[1]) or (mat.shape[0] != mat.shape[1]):
            raise Exception("Matrix size incompatible.")

        return np.dot(np.dot(evecs.conj(), mat), evecs.T)

    def project_sky_matrix_forward(self, mi, mat):

        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1
        nfreq = self.telescope.nfreq

        evsky = self.skymodes_m(mi).reshape((-1, nfreq, npol, lside))

        matf = np.zeros((evsky.shape[0], evsky.shape[0]), dtype=np.complex128)
        
        for li in range(lside):
            for pi in range(npol):
                for pj in range(npol):
                    matf += np.dot(np.dot(evsky[..., pi, li], mat[pi, pj, li, ...]), evsky[..., pj, li].T.conj())

        return matf


    def project_sky_matrix_forward_c(self, mi, mat):

        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1
        nfreq = self.telescope.nfreq

        evsky = self.skymodes_m_c(mi).reshape((-1, nfreq, npol, lside))

        matf = np.zeros((evsky.shape[0], evsky.shape[0]), dtype=np.complex128)
        
        for li in range(lside):
            for pi in range(npol):
                for pj in range(npol):
                    matf += np.dot(np.dot(evsky[..., pi, li], mat[pi, pj, li, ...]), evsky[..., pj, li].T.conj())

        return matf            
        
        
        

        

        
