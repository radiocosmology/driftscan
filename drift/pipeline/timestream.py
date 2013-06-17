import pickle
import os

import h5py
import numpy as np

from cosmoutils import hputil

from drift.core import manager
from drift.util import util, mpiutil


class Timestream(object):

    directory = None
    output_directory = None
    beamtransfer_dir = None




    #============ Constructor etc. =====================

    def __init__(self, tsdir, prodconfig):
        """Create a new Timestream object.

        Parameters
        ----------
        tsdir : string
            Directory to create the Timestream in.
        btdir : string
            Directory that the BeamTransfer files are stored in.
        """
        self.directory = os.path.abspath(tsdir)
        self.output_directory = self.directory
        #self.beamtransfer_dir = os.path.abspath(btdir)
        self.manager = manager.ProductManager.from_config(prodconfig)
    
    #====================================================


    #===== Accessing the BeamTransfer and Telescope =====

    _beamtransfer = None
    @property
    def beamtransfer(self):
        """The BeamTransfer object corresponding to this timestream.
        """
        # if self._beamtransfer is None:
        #     self._beamtransfer = beamtransfer.BeamTransfer(self.beamtransfer_dir)

        # return self._beamtransfer

        return self.manager.beamtransfer

    @property
    def telescope(self):
        """The telescope object corresponding to this timestream.
        """
        return self.beamtransfer.telescope

    #====================================================


    #======== Fetch and generate the f-stream ===========


    def _fdir(self, fi):
        # Pattern to form the `freq` ordered file.
        pat = self.directory + "/timestream_f/" + util.natpattern(self.telescope.nfreq)
        return pat % fi


    def _ffile(self, fi):
        # Pattern to form the `freq` ordered file.
        return self._fdir(fi) + "/timestream.hdf5"


    def timestream_f(self, fi):
        """Fetch the timestream for a given frequency.

        Parameters
        ----------
        fi : integer
            Frequency to load.

        Returns
        -------
        timestream : np.ndarray[npairs, ntime]
            The visibility timestream.
        """

        with h5py.File(self._ffile(fi), 'r') as f:
            ts = f['timestream'][:]
        return ts

    #====================================================


    #======== Fetch and generate the m-modes ============

    def _mdir(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self.output_directory + "/mmodes/" + util.natpattern(self.telescope.mmax)
        return pat % abs(mi)


    def _mfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + '/mode.hdf5'


    def mmode(self, mi):
        """Fetch the timestream m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        timestream : np.ndarray[nfreq, pm, npairs]
            The visibility m-modes.
        """

        with h5py.File(self._mfile(mi), 'r') as f:
            return f['mmode'][:]


    def generate_mmodes(self):
        """Calculate the m-modes corresponding to the Timestream.

        Perform an MPI transpose for efficiency.
        """


        if os.path.exists(self.output_directory + "/mmodes/COMPLETED_M"):
            if mpiutil.rank0:
                print "******* m-files already generated ********"
            return

        tel = self.telescope
        mmax = tel.mmax
        nfreq = tel.nfreq

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
        lm, sm, em = mpiutil.split_local(mmax + 1)

        ntime = 2*mmax+1

        # Load in the local frequencies of the time stream
        tstream = np.zeros((lfreq, tel.npairs, ntime), dtype=np.complex128)
        for lfi, fi in enumerate(range(sfreq, efreq)):
            tstream[lfi] = self.timestream_f(fi)

        # FFT to calculate the m-modes for the timestream
        row_mmodes = np.fft.fft(tstream, axis=-1) / (2*mmax + 1.0)

        ## Combine positive and negative m parts.
        row_mpairs = np.zeros((lfreq, 2, tel.npairs, mmax+1), dtype=np.complex128)

        row_mpairs[:, 0, ..., 0] = row_mmodes[..., 0]
        for mi in range(1,mmax+1):
            row_mpairs[:, 0, ..., mi] = row_mmodes[...,  mi]
            row_mpairs[:, 1, ..., mi] = row_mmodes[..., -mi].conj()

        # Transpose to get the entirety of an m-mode on each process (i.e. all frequencies)
        col_mmodes = mpiutil.transpose_blocks(row_mpairs, (nfreq, 2, tel.npairs, mmax + 1))

        # Transpose the local section to make the m's first
        col_mmodes = np.transpose(col_mmodes, (3, 0, 1, 2))

        for lmi, mi in enumerate(range(sm, em)):

            # Make directory for each m-mode
            if not os.path.exists(self._mdir(mi)):
                os.makedirs(self._mdir(mi))

            # Create the m-file and save the result.
            with h5py.File(self._mfile(mi), 'w') as f:
                f.create_dataset('/mmode', data=col_mmodes[lmi])
                f.attrs['m'] = mi

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(self.output_directory + "/mmodes/COMPLETED_M", 'a').close()

    #====================================================


    #======== Make and fetch SVD m-modes ================

    def _svdfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + '/svd.hdf5'


    def mmode_svd(self, mi):
        """Fetch the SVD m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        svd_mode : np.ndarray[nfreq, pm, npairs]
            The visibility m-modes.
        """

        with h5py.File(self._svdfile(mi), 'r') as f:
            return f['mmode_svd'][:]


    def generate_mmodes_svd(self):
        """Generate the SVD modes for the Timestream.
        """
        
        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)):
                print "File %s exists. Skipping..." % self._svdfile(mi)
                continue

            tm = self.mmode(mi).reshape(self.telescope.nfreq, 2*self.telescope.npairs)
            svdm = self.beamtransfer.project_vector_telescope_to_svd(mi, tm)

            with h5py.File(self._svdfile(mi), 'w') as f:
                f.create_dataset('mmode_svd', data=svdm)
                f.attrs['m'] = mi

    #====================================================


    #======== Make map from uncleaned stream ============

    def mapmake_full(self, nside, mapname):


        def _make_alm(mi):

            print "Making %i" % mi

            mmode = self.mmode(mi)
            sphmode = self.beamtransfer.project_vector_telescope_to_sky(mi, mmode)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, range(self.telescope.mmax + 1))

        if mpiutil.rank0:

            alm = np.zeros((self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            for mi in range(self.telescope.mmax + 1):

                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            with h5py.File(self.output_directory + '/' + mapname, 'w') as f:
                f.create_dataset('/map', data=skymap)


    def mapmake_svd(self, nside, mapname):

        self.generate_mmodes_svd()

        def _make_alm(mi):

            svdmode = self.mmode_svd(mi)

            sphmode = self.beamtransfer.project_vector_svd_to_sky(mi, svdmode)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, range(self.telescope.mmax + 1))

        if mpiutil.rank0:

            alm = np.zeros((self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            for mi in range(self.telescope.mmax + 1):

                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            with h5py.File(self.output_directory + '/' + mapname, 'w') as f:
                f.create_dataset('/map', data=skymap)

    #====================================================


    #========== Project into KL-mode basis ==============

    def set_kltransform(self, klname, threshold=None):

        self.klname = klname

        if threshold is None:
            kl = self.manager.kltransforms[self.klname]
            threshold = kl.threshold

        self.klthreshold = threshold 

    def _klfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + ('/klmode_%s_%f.hdf5' % (self.klname, self.klthreshold))




    def mmode_kl(self, mi):
        with h5py.File(self._klfile(mi), 'r') as f:
            if f['mmode_kl'].shape[0] == 0:
                return np.zeros((0,), dtype=np.complex128)
            else:
                return f['mmode_kl'][:]


    def generate_mmodes_kl(self):
        """Generate the KL modes for the Timestream.
        """

        kl = self.manager.kltransforms[self.klname]
        
        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._klfile(mi)):
                print "File %s exists. Skipping..." % self._klfile(mi)
                continue

            svdm = self.mmode_svd(mi) #.reshape(self.telescope.nfreq, 2*self.telescope.npairs)
            #svdm = self.beamtransfer.project_vector_telescope_to_svd(mi, tm)

            klm = kl.project_vector_svd_to_kl(mi, svdm, threshold=self.klthreshold)

            with h5py.File(self._klfile(mi), 'w') as f:
                f.create_dataset('mmode_kl', data=klm)
                f.attrs['m'] = mi


    def fake_kl_data(self):

        kl = self.manager.kltransforms[self.klname]

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            evals = kl.evals_m(mi)

            if evals is None:
                klmode = np.array([], dtype=np.complex128)
            else:
                modeamp = ((evals + 1.0) / 2.0)**0.5
                klmode = modeamp * (np.array([1.0, 1.0J]) * np.random.standard_normal((modeamp.shape[0], 2))).sum(axis=1)
            

            with h5py.File(self._klfile(mi), 'w') as f:
                f.create_dataset('mmode_kl', data=klmode)
                f.attrs['m'] = mi


    def mapmake_kl(self, nside, mapname, wiener=False):


        kl = self.manager.kltransforms[self.klname]

        if not kl.inverse:
            raise Exception("Need the inverse to make a meaningful map.")

        def _make_alm(mi):
            print "Making %i" % mi

            klmode = self.mmode_kl(mi)

            if wiener:
                evals = kl.evals_m(mi, self.klthreshold)

                if evals is not None:
                    klmode *= (evals / (1.0 + evals))

            isvdmode = kl.project_vector_kl_to_svd(mi, klmode, threshold=self.klthreshold)

            sphmode = self.beamtransfer.project_vector_svd_to_sky(mi, isvdmode)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, range(self.telescope.mmax + 1))

        if mpiutil.rank0:

            alm = np.zeros((self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            for mi in range(self.telescope.mmax + 1):

                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            with h5py.File(self.output_directory + '/' + mapname, 'w') as f:
                f.create_dataset('/map', data=skymap)

    #====================================================


    #======= Estimate powerspectrum from data ===========


    @property
    def _psfile(self):
        # Pattern to form the `m` ordered file.
        return self.output_directory + ('/ps_%s_%s.hdf5' % (self.klname, self.psname))



    def set_psestimator(self, psname):
        self.psname = psname


    def powerspectrum(self):

        import scipy.linalg as la
        

        if os.path.exists(self._psfile):
            print "File %s exists. Skipping..." % self._psfile
            return

        ps = self.manager.psestimators[self.psname]
        ps.genbands()

        def _q_estimate(mi):

            return ps.q_estimator(mi, self.mmode_kl(mi))

        qvals = mpiutil.parallel_map(_q_estimate, range(self.telescope.mmax + 1))

        qtotal = np.array(qvals).sum(axis=0)

        fisher, bias = ps.fisher_bias()

        powerspectrum =  np.dot(la.inv(fisher), qtotal - bias)


        if mpiutil.rank0:
            with h5py.File(self._psfile, 'w') as f:


                cv = la.inv(fisher)
                err = cv.diagonal()**0.5
                cr = cv / np.outer(err, err)

                f.create_dataset('fisher/', data=fisher)
#                f.create_dataset('bias/', data=self.bias)
                f.create_dataset('covariance/', data=cv)
                f.create_dataset('error/', data=err)
                f.create_dataset('correlation/', data=cr)

                f.create_dataset('bandpower/', data=ps.band_power)
                #f.create_dataset('k_start/', data=ps.k_start)
                #f.create_dataset('k_end/', data=ps.k_end)
                #f.create_dataset('k_center/', data=ps.k_center)
                #f.create_dataset('psvalues/', data=ps.psvalues)

                f.create_dataset('powerspectrum', data=powerspectrum)

        # Delete cache of bands for memory reasons
        del ps.clarray
        ps.clarray = None

        return powerspectrum




    #====================================================


    #======== Load and save the Pickle files ============

    def __getstate__(self):
        ## Remove the attributes we don't want pickled.
        state = self.__dict__.copy()

        for key in self.__dict__:
            #if (key in delkeys) or (key[0] == "_"):
            if (key[0] == "_"):
                del state[key]

        return state


    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.output_directory + "/timestreamobject.pickle"


    def save(self):
        """Save out the Timestream object information."""

        # Save pickled telescope object
        if mpiutil.rank0:
            with open(self._picklefile, 'w') as f:
                print "=== Saving Timestream object. ==="
                pickle.dump(self, f)


    @classmethod
    def load(cls, tsdir):
        """Load the Timestream object from disk.

        Parameters
        ----------
        tsdir : string
            Name of the directory containing the Timestream object.
        """

        # Create temporary object to extract picklefile property
        tmp_obj = cls(tsdir, tsdir)

        with open(tmp_obj._picklefile, 'r') as f:
            print "=== Loading Timestream object. ==="
            return pickle.load(f)

    #====================================================






