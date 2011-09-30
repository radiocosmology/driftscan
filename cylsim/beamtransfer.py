import pickle
import os

import numpy as np

import h5py


_rank = 0
_size = 1
_comm = None

## Try to setup MPI and get the comm, rank and size.
## If not they should end up as rank=0, size=1.
try:
    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    
    rank = _comm.Get_rank()
    size = _comm.Get_size()

    _rank = rank if rank else 0
    _size = size if size else 1
    
except ImportError:
    pass


def _intpattern(n):
    """Pattern that prints out a number upto `n` (integer - always shows sign)."""
    return ("%+0" + repr(int(np.ceil(np.log10(n + 1))) + 1) + "d")


def _natpattern(n):
    """Pattern that prints out a number upto `n` (natural number - no sign)."""
    return ("%0" + repr(int(np.ceil(np.log10(n + 1)))) + "d")


def partition_list_alternate(full_list, i, n):
    """Partition a list into `n` pieces. Return the `i`th partition."""
    return full_list[i::n]


def partition_list_mpi(full_list):
    """Return the partition of a list specific to the current MPI process."""
    return partition_list_alternate(full_list, _rank, _size)



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
        return self.directory + "/beam_m_" + _intpattern(self.telescope.mmax) + ".hdf5"

    @property
    def _ffile(self):
        # Pattern to form the `freq` ordered file.
        return self.directory + "/beam_f_" + _natpattern(self.telescope.nfreq) + ".hdf5"

    #===================================================



    #=========== Patterns for HDF5 datasets ============

    @property
    def _msection(self):
        # The pattern for `m` datasets in freq files.
        return "m_section/" + _intpattern(self.telescope.mmax)

    @property
    def _fsection(self):
        # The pattern for `freq` datasets in m files.
        return "freq_section/" + _natpattern(self.telescope.nfreq)

    #===================================================


    @property
    def _telescope_pickle(self):
        # The pickled telescope object
        return pickle.dumps(self.telescope)


    def __init__(self, directory, telescope = None):
        
        self.directory = directory
        self.telescope = telescope
        
        # Create directory if required
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        
        if self.telescope == None:
            print "Attempting to read telescope from disk..."

            try:
                f =  open(self._picklefile, 'r')
                self.telescope = pickle.load(f)
            except IOError, UnpicklingError:
                raise Exception("Could not load Telescope object from disk.")
                

        
            
    def beam_m(self, mi):
        
        beam = np.zeros((self.telescope.nfreq, self.telescope.nbase, self.telescope.lmax+1), dtype=np.complex128)
        
        mfile = h5py.File(self._mfile % mi, 'r')
        
        for fi in range(self.telescope.nfreq):
            beam[fi] = mfile[(self._fsection % fi)]
            
        mfile.close()
        
        return beam
        
        
    def beam_freq(self, fi):
        
        beam = np.zeros((self.telescope.nbase, self.telescope.lmax+1, 2*self.telescope.mmax+1), dtype=np.complex128)
        
        ffile = h5py.File(self._ffile % fi, 'r')
        
        for mi in range(-self.telescope.mmax, self.telescope.mmax+1):
            beam[:,:,mi] = ffile[(self._msection % mi)]
            
        ffile.close()
        
        return beam


    def generate_cache(self):

        # Get frequency channels this process will calculate
        flist = partition_list_mpi(np.arange(self.telescope.nfreq))

        # For each frequency, create the HDF5 file, and write in each `m` as a
        # seperate compressed dataset.
        for fi in flist:
            print 'f index %i. Creating file: %s' % (fi, self._ffile % fi)
            
            f = h5py.File(self._ffile % fi, 'w')

            # Calculate transfer matrices for each frequency
            btrans = self.telescope.transfer_for_frequency(fi)

            # Set a few useful attributes.
            f.attrs['baselines'] = self.telescope.baselines
            f.attrs['baseline_indices'] = np.arange(self.telescope.nbase)
            f.attrs['frequency_index'] = fi
            f.attrs['frequency'] = self.telescope.frequencies[fi]
            f.attrs['cylobj'] = self._telescope_pickle

            for mi in range(-self.telescope.mmax, self.telescope.mmax+1):
                
                dset = f.create_dataset((self._msection % mi), data=btrans[:, :, mi], compression='gzip')
                dset.attrs['m'] = mi
        
            f.close()

        # If we're part of an MPI run, synchronise here.
        if _comm:
            _comm.Barrier()

        # Get the set of `m`s this process will reorder.
        mlist = partition_list_mpi(np.arange(-self.telescope.mmax, self.telescope.mmax+1))

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file.
        for mi in mlist:

            print 'm index %i. Creating file: %s' % (mi, self._mfile % mi)

            ## Create hdf5 file for each m-mode
            f = h5py.File(self._mfile % mi, 'w')

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
        if _rank == 0:
            with open(self._picklefile, 'w') as f:
                print "=== Saving Telescope object. ==="
                pickle.dump(self.telescope, f)

