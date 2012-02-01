import time
import os
import warnings

import numpy as np
import scipy.linalg as la

import h5py
import healpy


from simulations.foregroundmap import matrix_root_manynull

from cylsim import mpiutil
from cylsim import beamtransfer
from cylsim import util
from cylsim import skymodel
from cylsim import hputil

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

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()


    def foreground(self):
        """Compute the foreground covariance matrix (on the sky).

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
        """Compute the signal covariance matrix (on the sky).

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


    def sn_covariance(self, mi):
        """Compute the signal and noise covariances (on the telescope).

        The signal is formed from the 21cm signal, whereas the noise includes
        both foregrounds and instrumental noise. This is for a single m-mode.

        Parameters
        ----------
        mi : integer
            The m-mode to calculate at.

        Returns
        -------
        s, n : np.ndarray[nfreq, npol*nbase, nfreq, npol*nbase]
            Signal and noice covariance matrices.
        """

        
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        nfreq = self.telescope.nfreq

        # Project the signal and foregrounds from the sky onto the telescope.
        cvb_s = self.beamtransfer.project_matrix_forward(mi, self.signal())
        cvb_n = self.beamtransfer.project_matrix_forward(mi, self.foreground())

        # Add in the instrumental noise. Assumed to be diagonal for now.
        for fi in range(nfreq):
            noisebase = np.diag(self.telescope.noisepower(np.arange(self.telescope.nbase), fi).reshape(ntel))
            cvb_n[fi, :, fi, :] += noisebase

        return cvb_s, cvb_n


    def transform_m(self, mi):
        """Perform the KL-transform for a single m.

        Parameters
        ----------
        mi : integer
            The m-mode to calculate for.

        Returns
        -------
        evals, evecs : np.ndarray
            The KL-modes. The evals correspond to the diagonal of the
            covariances in the new basis, and the evecs define the basis.
        """
        
        print "Solving for Eigenvalues...."

        # Fetch the covariance matrices to diagonalise
        st = time.time()
        nside = self.telescope.nbase * self.telescope.num_pol_telescope * self.telescope.nfreq
        cvb_sr, cvb_nr = [cv.reshape(nside, nside) for cv in self.sn_covariance(mi)]
        et = time.time()
        print "Time =", (et-st)

        # Perform the generalised eigenvalue problem to get the KL-modes.
        st = time.time()
        res = eigh_gen(cvb_sr, cvb_nr)
        et=time.time()
        print "Time =", (et-st)

        return res




    def transform_save(self, mi):
        """Save the KL-modes for a given m.

        Perform the transform and cache the results for later use.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Results
        -------
        evals, evecs : np.ndarray
            See `transfom_m` for details.
        """
        
        # Perform the KL-transform
        print "Constructing signal and noise covariances for m = %i ..." % (mi)
        evals, evecs, ac = self.transform_m(mi)
    
        ## Write out Eigenvals and Vectors

        # Create file and set some metadata
        print "Creating file %s ...." % (self._evfile % mi)
        f = h5py.File(self._evfile % mi, 'w')
        f.attrs['m'] = mi
        f.attrs['SUBSET'] = self.subset

        # Write out the full spectrum of eigenvalues
        f.create_dataset('evals_full', data=evals)

        # Discard eigenmodes with S/N below threshold if requested.
        if self.subset:
            i_ev = np.searchsorted(evals, self.threshold)
            
            evalsf = evals
            
            evals = evals[i_ev:]
            evecs = evecs[:, i_ev:]
            print "Modes with S/N > %f: %i of %i" % (self.threshold, evals.size, evalsf.size)

        # Write out potentially reduced eigen spectrum.
        f.create_dataset('evals', data=evals)
        f.create_dataset('evecs', data=evecs.T.conj(), compression='gzip')
        f.attrs['num_modes'] = evals.size

        # If we had to regularise because the noise spectrum is numerically ill
        # conditioned, write out the constant we added to the diagonal (see
        # eigh_gen).
        if ac != 0.0:
            f.attrs['add_const'] = ac
            f.attrs['FLAGS'] = 'NotPositiveDefinite'
        else:
            f.attrs['FLAGS'] = 'Normal'
                
        f.close()

        return evals, evecs.T


    def evals_all(self):
        """Collects the full eigenvalue spectrum for all m-modes.

        Reads in from files on disk.

        Returns
        -------
        evarray : np.ndarray
            The full set of eigenvalues across all m-modes.
        """

        nside = self.telescope.nbase * self.telescope.num_pol_telescope * self.telescope.nfreq
        evarray = np.zeros((2*self.telescope.mmax+1, nside))

        # Iterate over all m's, reading file and extracting the eigenvalues.
        for mi in range(-self.telescope.mmax, self.telescope.mmax+1):

            f = h5py.File(self._evfile % mi, 'r')
            evarray[mi] = f['evals_full']
            f.close()

        return evarray



    def generate(self, mlist = None, regen=False):
        """Perform the KL-transform for all m-modes and save the result.

        Uses MPI to distribute the work (if available).

        Parameters
        ----------
        mlist : array_like, optional
            Set of m's to calculate KL-modes for By default do all m-modes.
        """
        
        # Iterate list over MPI processes.
        for mi in mpiutil.mpirange(-self.telescope.mmax, self.telescope.mmax+1):
            if os.path.exists(self._evfile % mi) and not regen:
                print "m index %i. File: %s exists. Skipping..." % (mi, (self._evfile % mi))
                continue

            self.transform_save(mi)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Create combined eigenvalue file.
        if mpiutil.rank0:
            if os.path.exists(self.evdir + "/evals.hdf5") and not regen:
                print "File: %s exists. Skipping..." % (self.evdir + "/evals.hdf5")
                return

            print "Creating eigenvalues file (process 0 only)."
            evals = self.evals_all()

            f = h5py.File(self.evdir + "/evals.hdf5", 'w')
            f.create_dataset('evals', data=evals)
            f.close()


    _last_mode_m = None
    _last_mode = None
    olddatafile = False

    def modes_m(self, mi, threshold=None):
        """Fetch the KL-modes for a particular m.

        This attempts to read in the results from disk, if available and if not
        will create them.

        Also, it will cache the previous m-mode in memory, so as to avoid disk
        access in many cases. However *this* is not sensitive to changes in the
        threshold, be careful.

        Parameters
        ----------
        mi : integer
            m to fetch KL-modes for.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        evals, evecs : np.ndarray
            KL-modes with S/N greater than some threshold. Both evals and evecs
            are potentially `None`, if there are no modes either in the file, or
            satisfying S/N > threshold.
        """
        
        # See if requested m was the previous one (thus cached in memory).
        if self._last_mode_m == mi:
            return self._last_mode

        # If modes not already saved to disk, create file.
        if not os.path.exists(self._evfile % mi):
            modes = self.transform_save(mi)
        else:
            f = h5py.File(self._evfile % mi, 'r')

            # If no modes are in the file, return None, None
            if f['evals'].shape[0] == 0:
                modes = None, None
            else:
                # Find modes satisfying threshold (if required).
                evals = f['evals'][:]
                startind = np.searchsorted(evals, threshold) if threshold is not None else 0

                if startind == evals.size:
                    modes = None, None
                else:
                    modes = ( evals[startind:], f['evecs'][startind:] )
                    
                    # If old data file perform complex conjugate
                    modes = modes if not self.olddatafile else ( modes[0], modes[1].conj() )
            f.close()

        # Cache the results of this call to reduce disk access.
        self._last_mode = modes
        self._last_mode_m = mi
        return modes


    _last_invmode_m = None
    _last_invmode = None
    def invmodes_m(self, mi, threshold=None):
        # See if requested m was the previous one (thus cached in memory).
        if self._last_invmode_m == mi:
            return self._last_invmode

        evals, modes = self.modes_m(mi, threshold)

        try:
            #invmode = la.pinv2(modes)
            invmode = la.pinv(modes)
        except la.LinAlgError:
            warnings.Warn("Pseudo-inverse by SVD failed. Trying alternative.")
            #invmode = la.pinv(modes)
            invmode = la.pinv2(modes)
        
        self._last_invmode_m = mi
        self._last_invmode = invmode

        return invmode


    _last_skymode_m = None
    _last_skymode = None

    def skymodes_m(self, mi, threshold=None):
        """Find the representation of the KL-modes on the sky.

        Use the beamtransfers to rotate the SN-modes onto the sky. This routine
        is based on `modes_m`, as such the same caching and caveats apply.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        skymodes : np.ndarray
            The modes as found in a_{lm}(\nu) space. Note this routine does not
            return the evals.

        See Also
        --------
        `modes_m`
        """
        if self._last_skymode_m == mi:
            return self._last_skymode

        # Fetch the modes in the telescope basis.
        evals, evecs = self.modes_m(mi, threshold=threshold)

        if evals is None:
            raise Exception("Don't seem to be any evals to use.")

        ## Rotate onto the sky basis. Slightly complex as need to do
        ## frequency-by-frequency
        nfreq = self.telescope.nfreq
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        nsky = self.telescope.num_pol_sky * (self.telescope.lmax + 1)

        beam = self.beamtransfer.beam_m(mi).reshape((nfreq, ntel, nsky))
        evecs = evecs.reshape((-1, nfreq, ntel))

        evsky = np.zeros((evecs.shape[0], nfreq, nsky), dtype=np.complex128)
        
        for fi in range(nfreq):
            evsky[:, fi, :] = np.dot(evecs[:, fi, :], beam[fi])

        # Cache results and return
        self._last_skymode = evsky
        self._last_skymode_m = mi
        return evsky
            



    def project_tel_vector_forward(self, mi, vec, threshold=None):
        """Project a telescope data vector into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Telescope data vector.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        evals, evecs = self.modes_m(mi, threshold)

        if evals is None:
            return None

        if vec.shape[0] != evecs.shape[1]:
            raise Exception("Vectors are incompatible.")

        return np.dot(evecs, vec)


    def project_tel_vector_backward(self, mi, vec, threshold=None):
        """Project a vector in the Eigenbasis back into the telescope space.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Eigenbasis data vector.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        evals, evecs = self.modes_m(mi, threshold)

        if evals is None:
            return np.zeros(self.telescope.num_pol_telescope * self.telescope.nbase * self.telescope.nfreq, dtype=np.complex128)

        if vec.shape[0] != evecs.shape[0]:
            raise Exception("Vectors are incompatible.")

        # Construct the pseudo inverse
        invmodes = self.invmodes_m(mi, threshold)

        return np.dot(self.invmodes_m(mi, threshold), vec)


    def project_sky_vector_forward(self, mi, vec, threshold=None):
        """Project an m-vector from the sky into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [freq, pol, l]
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        tvec = self.beamtransfer.project_vector_forward(mi, vec).flatten()

        return self.project_tel_vector_forward(mi, tvec, threshold)


    def project_tel_matrix_forward(self, mi, mat, threshold=None):

        evals, evecs = self.modes_m(mi, threshold)

        if (mat.shape[0] != evecs.shape[1]) or (mat.shape[0] != mat.shape[1]):
            raise Exception("Matrix size incompatible.")

        return np.dot(np.dot(evecs, mat), evecs.T.conj())


    def project_sky_matrix_forward(self, mi, mat, threshold=None):

        npol = self.telescope.num_pol_sky
        nbase = self.telescope.nbase
        nfreq = self.telescope.nfreq
        nside = npol * nbase * nfreq

        mproj = self.beamtransfer.project_matrix_forward(mi, mat)

        return self.project_tel_matrix_forward(mi, mproj.reshape((nside, nside)), threshold)


    def project_sky_matrix_forward_old(self, mi, mat, threshold=None):

        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1
        nfreq = self.telescope.nfreq

        st = time.time()

        evsky = self.skymodes_m(mi, threshold).reshape((-1, nfreq, npol, lside))
        et = time.time()
        
        print "Evsky: %f" % (et-st)

        st = time.time()
        ev1n = np.transpose(evsky, (2, 3, 0, 1)).copy()
        ev1h = np.transpose(evsky, (2, 3, 1, 0)).conj()
        matf = np.zeros((evsky.shape[0], evsky.shape[0]), dtype=np.complex128)

        for pi in range(npol):
            for pj in range(npol):
                for li in range(lside):
                    matf += np.dot(np.dot(ev1n[pi, li], mat[pi, pj, li]), ev1h[pj, li])

        et = time.time()
        
        print "Rest: %f" % (et-st)


        return matf


    def project_sky(self, sky, mlist = None, threshold=None, harmonic=False):

        # Set default list of m-modes (i.e. all of them), and partition
        if mlist is None:
            mlist = range(-self.telescope.mmax, self.telescope.mmax + 1)
        mpart = mpiutil.partition_list_mpi(mlist)
        
        # Total number of sky modes.
        nmodes = self.telescope.num_pol_telescope * self.telescope.nbase * self.telescope.nfreq

        # If sky is alm fine, if not perform spherical harmonic transform.
        alm = sky if harmonic else hputil.sphtrans_sky(sky, lmax=self.telescope.lmax)


        ## Routine to project sky onto eigenmodes
        def _proj(mi):
            p1 = self.project_sky_vector_forward(mi, alm[:, :, mi], threshold)
            p2 = np.zeros(nmodes, dtype=np.complex128)
            p2[-p1.size:] = p1
            return p2

        # Map over list of m's and project sky onto eigenbasis
        proj_sec = [(mi, _proj(mi)) for mi in mpart]

        # Gather projections onto the rank=0 node.
        proj_all = mpiutil.world.gather(proj_sec, root=0)

        proj_arr = None
        
        if mpiutil.rank0:
            # Create array to put projections into
            proj_arr = np.zeros((2*self.telescope.mmax + 1, nmodes), dtype=np.complex128)

            # Iterate over all gathered projections and insert into the array
            for proc_rank in proj_all:
                for pm in proc_rank:
                    proj_arr[pm[0]] = pm[1]

        # Return the projections (rank=0) or None elsewhere.
        return proj_arr

            

