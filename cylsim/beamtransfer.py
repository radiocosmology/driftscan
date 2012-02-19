import pickle
import os

import numpy as np
import scipy.linalg as la
import h5py

import mpiutil
import util



class BeamTransfer(object):
    """A class for reading and writing Beam Transfer matrices from disk."""


    
    #====== Properties giving internal filenames =======
    
    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.directory + "/telescopeobject.pickle"

    @property
    def _mfile(self):
        # Pattern to form the `m` ordered file.
        return self.directory + "/beam_m_" + util.intpattern(self.telescope.mmax) + ".hdf5"

    @property
    def _ffile(self):
        # Pattern to form the `freq` ordered file.
        return self.directory + "/beam_f_" + util.natpattern(self.telescope.nfreq) + ".hdf5"

    #===================================================



    #=========== Patterns for HDF5 datasets ============

    @property
    def _msection(self):
        # The pattern for `m` datasets in freq files.
        return "m_section/" + util.intpattern(self.telescope.mmax)

    @property
    def _fsection(self):
        # The pattern for `freq` datasets in m files.
        return "freq_section/" + util.natpattern(self.telescope.nfreq)

    #===================================================


    @property
    def _telescope_pickle(self):
        # The pickled telescope object
        return pickle.dumps(self.telescope)


    def __init__(self, directory, telescope = None):
        
        self.directory = directory
        self.telescope = telescope
        
        # Create directory if required
        if mpiutil.rank0 and not os.path.exists(directory):
            os.makedirs(directory)

        mpiutil.barrier()
        
        if self.telescope == None:
            print "Attempting to read telescope from disk..."

            try:
                f =  open(self._picklefile, 'r')
                self.telescope = pickle.load(f)
            except IOError, UnpicklingError:
                raise Exception("Could not load Telescope object from disk.")
                

        
            
    def beam_m(self, mi):
        
        
        mfile = h5py.File(self._mfile % mi, 'r')
        sh = mfile[(self._fsection % 0)].shape

        beam = np.zeros((self.telescope.nfreq,) + sh, dtype=np.complex128)
        
        for fi in range(self.telescope.nfreq):
            beam[fi] = mfile[(self._fsection % fi)]
            
        mfile.close()
        
        return beam


    _ibeam = None
    _ibeam_m = None
    def invbeam_m(self, mi):
        
        if self._ibeam_m == mi:
            return self._ibeam
        
        nfreq = self.telescope.nfreq
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        nsky = self.telescope.num_pol_sky * (self.telescope.lmax + 1)
        
        beam = self.beam_m(mi).reshape((nfreq, ntel, nsky))
        
        ibeam = np.zeros((nfreq, nsky, ntel), dtype=np.complex128)

        for fi in range(nfreq):
            bh = beam[fi].T.conj()
            ibeam[fi] = la.pinv2(beam[fi])

        ibeam = ibeam.reshape((nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                               self.telescope.nbase, self.telescope.num_pol_telescope))

        self._ibeam = ibeam
        self._ibeam_m = mi
        return ibeam
        
        
    def beam_freq(self, fi, fullm = False):

        mside = 2*self.telescope.lmax+1 if fullm else 2*self.telescope.mmax+1
        
        
        ffile = h5py.File(self._ffile % fi, 'r')
        sh = ffile[(self._msection % 0)].shape
        beam = np.zeros(sh + (mside,), dtype=np.complex128)
        
        for mi in range(-self.telescope.mmax, self.telescope.mmax+1):
            beam[..., mi] = ffile[(self._msection % mi)]
            
        ffile.close()
        
        return beam


    def generate_cache(self, regen = False):

        # For each frequency, create the HDF5 file, and write in each `m` as a
        # seperate compressed dataset. Use MPI if available. 
        for fi in mpiutil.mpirange(self.telescope.nfreq):

            if os.path.exists(self._ffile % fi) and not regen:
                print "f index %i. File: %s exists. Skipping..." % (fi, (self._ffile % fi))
                continue
            else:
                print 'f index %i. Creating file: %s' % (fi, (self._ffile % fi))

            f = h5py.File(self._ffile % fi, 'w')
            f.create_group('m_section')

            # Calculate transfer matrices for each frequency
            btrans = self.telescope.transfer_for_frequency(fi)

            # Set a few useful attributes.
            f.attrs['baselines'] = self.telescope.baselines
            f.attrs['baseline_indices'] = np.arange(self.telescope.nbase)
            f.attrs['frequency_index'] = fi
            f.attrs['frequency'] = self.telescope.frequencies[fi]
            f.attrs['cylobj'] = self._telescope_pickle

            for mi in range(-self.telescope.mmax, self.telescope.mmax+1):
                
                dset = f.create_dataset((self._msection % mi), data=btrans[..., mi], compression='gzip')
                dset.attrs['m'] = mi
        
            f.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available. 
        for mi in mpiutil.mpirange(-self.telescope.mmax, self.telescope.mmax+1):

            if os.path.exists(self._mfile % mi) and not regen:
                print "m index %i. File: %s exists. Skipping..." % (mi, (self._mfile % mi))
                continue
            else:
                print 'm index %i. Creating file: %s' % (mi, self._mfile % mi)

            ## Create hdf5 file for each m-mode
            f = h5py.File(self._mfile % mi, 'w')
            f.create_group('freq_section')
            ## For each frequency read in the current m-mode and copy into file.
            for fi in np.arange(self.telescope.nfreq):

                ff = h5py.File(self._ffile % fi, 'r')

                # Check frequency is what we expect.
                if fi != ff.attrs['frequency_index']:
                    raise Exception("Bork.")
        
                dset = f.create_dataset(self._fsection % fi, data=ff[self._msection % mi], compression='gzip')
                dset.attrs['frequency_index'] = fi
                ff.close()
                
            # Write a few useful attributes.
            f.attrs['baselines'] = self.telescope.baselines
            f.attrs['m'] = mi
            f.attrs['frequencies'] = self.telescope.frequencies
            f.attrs['cylobj'] = self._telescope_pickle
            
            f.close()

        # Save pickled telescope object
        if mpiutil.rank0:
            with open(self._picklefile, 'w') as f:
                print "=== Saving Telescope object. ==="
                pickle.dump(self.telescope, f)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        

    def project_vector_forward(self, mi, vec):
                
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        nsky = self.telescope.num_pol_sky * (self.telescope.lmax + 1)
        nfreq = self.telescope.nfreq

        beam = self.beam_m(mi).reshape((nfreq, ntel, nsky))
        
        vecf = np.zeros((nfreq, ntel), dtype=np.complex128)

        for fi in range(nfreq):
            vecf[fi] = np.dot(beam[fi], vec[..., fi, :].reshape(nsky))

        return vecf

    
    def project_vector_backward(self, mi, vec):
                
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        nsky = self.telescope.num_pol_sky * (self.telescope.lmax + 1)
        nfreq = self.telescope.nfreq

        ibeam = self.invbeam_m(mi).reshape((nfreq, nsky, ntel))
        
        vecb = np.zeros((nfreq, nsky), dtype=np.complex128)
        vec = vec.reshape((nfreq, ntel))

        for fi in range(nfreq):
            vecb[fi] = np.dot(ibeam[fi], vec[fi, :].reshape(ntel))

        return vecb.reshape((nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1))
    

    def project_matrix_forward(self, mi, mat):
                
        ntel = self.telescope.nbase * self.telescope.num_pol_telescope
        npol = self.telescope.num_pol_sky
        nfreq = self.telescope.nfreq
        lside = self.telescope.lmax + 1

        beam = self.beam_m(mi).reshape((nfreq, ntel, npol, lside))
        
        matf = np.zeros((nfreq, ntel, nfreq, ntel), dtype=np.complex128)


        # Should it be a +=?
        for pi in range(npol):
            for pj in range(npol):
                for fi in range(nfreq):
                    for fj in range(nfreq):
                        matf[fi, :, fj, :] += np.dot((beam[fi, :, pi, :] * mat[pi, pj, :, fi, fj]), beam[fj, :, pj, :].T.conj())

        return matf

                    
            
        
