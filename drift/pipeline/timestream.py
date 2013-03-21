import pickle
import os

import h5py
import numpy as np

from drift.core import beamtransfer
from drift.util import util, mpiutil


class Timestream(object):

    directory = None

    beamtransfer_dir = None




    #============ Constructor etc. =====================

    def __init__(self, tsdir, btdir):
        """Create a new Timestream object.

        Parameters
        ----------
        tsdir : string
            Directory to create the Timestream in.
        btdir : string
            Directory that the BeamTransfer files are stored in.
        """
        self.directory = os.path.abspath(tsdir)
        self.beamtransfer_dir = os.path.abspath(btdir)
    
    #====================================================


    #===== Accessing the BeamTransfer and Telescope =====

    _beamtransfer = None
    @property
    def beamtransfer(self):
        """The BeamTransfer object corresponding to this timestream.
        """
        if self._beamtransfer is None:
            self._beamtransfer = beamtransfer.BeamTransfer(self.beamtransfer_dir)

        return self._beamtransfer

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
        pat = self.directory + "/mmodes/" + util.natpattern(self.telescope.mmax)
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
        """

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
        row_mmodes = np.fft.fft(tstream, axis=-1)

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
            f = h5py.File(self._mfile(mi), 'w')
            f.create_dataset('/mmode', data=col_mmodes[lmi])
            f.attrs['m'] = mi
            f.close()

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

            tm = self.mmode(mi).reshape(self.telescope.nfreq, 2*self.telescope.npairs)
            svdm = self.beamtransfer.project_vector_telescope_to_svd(mi, tm)

            with h5py.File(self._svdfile(mi), 'w') as f:
                f.create_dataset('mmode_svd', data=svdm)
                f.attrs['m'] = mi

    #====================================================


    #======== Make map from uncleaned stream ============

    def mapmake(self):
        pass


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
        return self.directory + "/timestreamobject.pickle"


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






