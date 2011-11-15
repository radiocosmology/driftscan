

import numpy as np
import scipy.linalg as la

import h5py
import healpy


from simulations.foregroundmap import matrix_root_manynull

from cylsim import mpiutil
from cylsim import cylinder
from cylsim import beamtransfer
from cylsim import util
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
        
        add_const = -la.eigvalsh(B, eigvals=(0,0))[0] * 1.1
        
        B[np.diag_indices(nside)] += add_const
        evals, evecs = la.eigh(A, B, overwrite_a=True, overwrite_b=True)

    return evals, evecs, add_const



class KLTransform(object):
    """Perform KL transform.
    """

    subset = True
    threshold = 1.0

    @property
    def _evfile(self):
        # Pattern to form the `m` ordered file.
        return self.directory + "/ev_m_" + util.intpattern(self.telescope.mmax) + ".hdf5"
    

    def __init__(self, beamtransfer):
        self.beamtransfer = beamtransfer
        self.telescope = self.beamtransfer.telescope



    def foreground(self):
        ## Construct foreground matrix C[l,nu1,nu2]
        fsyn = foregroundsck.Synchrotron()
        fps = foregroundsck.PointSources()
        
        cv_fg = np.zeros((3, self.telescope.lmax+1, self.telescope.nfreq, self.telescope.nfreq))

        cv_fg[0] = (fsyn.angular_powerspectrum(np.arange(self.telescope.lmax+1))[:,np.newaxis,np.newaxis]
                    * fsyn.frequency_covariance(*np.meshgrid(self.telescope.frequencies, self.telescope.frequencies))[np.newaxis,:,:] * 1e-6)

        cv_fg[1] = 0.3**2 * (fsyn.angular_powerspectrum(np.arange(self.telescope.lmax+1))[:,np.newaxis,np.newaxis]
                             * fsyn.frequency_covariance(*np.meshgrid(self.telescope.frequencies, self.telescope.frequencies))[np.newaxis,:,:] * 1e-6)

        cv_fg[2] = cv_fg[1]

        cv_fg[0] += (fps.angular_powerspectrum(np.arange(self.telescope.lmax+1))[:,np.newaxis,np.newaxis]
                     * fps.frequency_covariance(*np.meshgrid(self.telescope.frequencies, self.telescope.frequencies))[np.newaxis,:,:] * 1e-6)



    def signal(self):
        cv_sg = np.zeros((3, self.telescope.lmax+1, self.telescope.nfreq, self.telescope.nfreq))

        ## Construct signal matrix C_l(nu, nu')
        cr = corr21cm.Corr21cm()
        za = units.nu21 / self.telescope.frequencies - 1.0
        cv_sg[0] = cr.angular_powerspectrum_fft(np.arange(self.telescope.lmax+1)[:,np.newaxis,np.newaxis], za[np.newaxis,:,np.newaxis], za[np.newaxis,np.newaxis,:]) * 1e-6



    def beam(self, mi):
        beam = self.beamtransfer.beam_m(mi)
        self.ntel = self.telescope.nbase
        self.nsky = self.telescope.lmax + 1
        


    def signal_covariance(self, mi):
                
        cvb_s = np.zeros((self.telescope.nfreq, self.self.ntel, self.telescope.nfreq, self.self.ntel),
                         dtype=np.complex128)
        cvb_n = np.zeros_like(cvb_s)

        cv_sg = self.signal()
        cv_fg = self.foreground()

        for fi in range(self.telescope.nfreq):
            for fj in range(self.telescope.nfreq):
                cvb_n[fi,:,fj,:] = np.dot((beam[fi] * cv_fg[..., fi, fj]).reshape((self.ntel, self.nsky)),
                                          beam[fj].reshape((self.ntel, self.nsky)).T.conj())
                cvb_s[fi,:,fj,:] = np.dot((beam[fi] * cv_sg[..., fi, fj]).reshape((self.ntel, self.nsky)),
                                          beam[fj].reshape((self.ntel, self.nsky)).T.conj())
            
            noisebase = np.diag(self.telescope.noisepower(np.arange(self.telescope.nbase), fi, ndays).reshape(self.ntel))
            cvb_n[fi,:,fi,:] += noisebase

        return cvb_s, cvb_n


    def transform_m(self, mi):
        
        print "Solving for Eigenvalues...."

        st = time.time()

        cvb_sr, cvb_nr = [cv.reshape(nside, nside) for cv in self.signal_covariance()]
        
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
            beam = bt.beam_m(mi)
            
            print "Constructing signal and noise covariances for m = %i ..." % (mi)
            evals, evecs, ac = self.transform_m(mi)
            
            ## Write out Eigenvals and Vectors
            print "Creating file %s ...." % (self.ev_pat % mi)
            f = h5py.File(self.ev_pat % mi, 'w')
            f.attrs['m'] = mi
            f.attrs['SUBSET'] = self.subset

            if self.subset:
                i_ev = np.search_sorted(evals, self.threshold)

                f.create_dataset('evals_full', data=evals)

                evals = (evals[:i_ev])[::-1]
                evecs = (evecs[:,i_ev:])[::-1]
                print "Modes with S/N > %f: %i of %i" % (self.threshold, evals_ss.size, evals.size)



            f.create_dataset('evals', data=evals)
            f.create_dataset('evecs', data=evecs.T, compression='gzip')
                
            if add_const != 0.0:
                f.attrs['add_const'] = add_const
                f.attrs['FLAGS'] = 'NotPositiveDefinite'
            else:
                f.attrs['FLAGS'] = 'Normal'
                
            f.close()





ev_pat = args.rootdir + "/" + args.evdir + "/ev_" + util.intpattern(self.telescope.mmax) + ".hdf5"



ndays = 730

self.nsky = 3 * (self.telescope.lmax + 1)
self.ntel = 3 * self.telescope.nbase

nside = self.telescope.nfreq * self.ntel

