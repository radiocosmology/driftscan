import pickle

import numpy as np

import h5py

from cylsim import cylinder

def _intpattern(n):
    """Pattern that prints out a number upto `n`."""
    return ("%+0" + repr(int(np.ceil(np.log10(mmax+1)))+1) + "d")

def _natpattern(n):
    """Pattern that prints out a number upto `n`."""
    return ("%0" + repr(int(np.ceil(np.log10(mmax+1)))) + "d")


def partition_list_alternate(full_list, i, n):
    return full_list[i::n]

def partition_list_mpi(full_list):

    part = full_list

    try:
        # Attempt to get the MPI rank and size
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_rank()

        # If rank and size are None, then we are not running as an MPI process
        if rank and size:
            part = partition_list_alternate(full_list, rank, size)

    except ImportError:
        pass

    return part
    


class BeamTransfer(object):
    """A class for reading and writing Beam Transfer matrices from disk."""

    @property
    def _cylfilename(self):
        return self.directory + "/cylinderobject.pickle"

    @property
    def _mfile(self):
        return self.directory + "/beam_m_"+_intpattern(self.cylinder.mmax)+"d.hdf5"

    @property
    def _ffile(self):
        return self.directory + "/beam_f_"+_natpattern(self.cylinder.nfreq)+"d.hdf5"

    @property
    def _msection(self):
        return "m_section/"+_intpattern(self.cylinder.mmax)

    @property
    def _fsection(self):
        return "freq_section/"+_natpattern(self.cylinder.nfreq)


    def __init__(self, directory, cylinder = None):

        self.directory = directory

        self.cylinder = cylinder

        if self.cylinder == None:
            print "Attempting to read cylinder from disk..."

            with open(self._cylfilename, 'r') as f:
                self.cylinder = pickle.load(f)
                

        
        
    def beam_m(mi):
    
        beam = np.zeros((self.cylinder.nfreq, self.cylinder.nbase, self.cylinder.lmax+1), dtype=np.complex128)
        
        mfile = h5py.File(self._mfile % mi, 'r')

        for fi in range(self.cylinder.nfreq):
            beam[fi] = mfile[(self._fsection % fi)]

        mfile.close()

        return beam


     def beam_freq(fi):
    
        beam = np.zeros((2*self.cylinder.mmax+1, self.cylinder.nbase, self.cylinder.lmax+1), dtype=np.complex128)
        
        ffile = h5py.File(self._ffile % fi, 'r')

        for mi in range(-self.cylinder.mmax, self.cylinder.mmax+1):
            beam[mi] = mfile[(self._msection % mi)]

        ffile.close()

        return beam


    def generate_cache(self):

        flist = partition_list_mpi(np.arange(self.cylinder.nfreq))

        for fi in flist:
            f = h5py.File(self._ffile % fi, 'w')
            
            btrans = self.cylinder.transfer_for_frequency(fi)

            f.attrs['baselines'] = self.cylinder.baselines
            f.attrs['baseline_indices'] = np.arange(self.cylinder.nbase)
            f.attrs['frequency_index'] = fi
            f.attrs['frequency'] = self.cylinder.frequencies[fi]
            f.attrs['cylobj'] = self.cylinderpickle

            for mi in range(-self.cylinder.mmax, self.cylinder.mmax+1):
                
                dset = mgrp.create_dataset((self._msection % mi), data=btrans[mi,:,:], compression='gzip')
                dset.attrs['m'] = mi
        
            f.close()

        comm.Barrier() ## Barrier here

        mlist = partition_list_mpi(np.arange(-self.cylinder.mmax, self.cylinder.mmax+1))

        for mi in mlist:

            print 'm index %i. Creating file: %s' % (mi, fname)

            ## Create hdf5 file for each m-mode
            f = h5py.File(self._mfile % mi, 'w')

            ## For each frequency read in the current m-mode and copy into file.
            for fi in np.arange(self.cylinder.nfreq):

                ff = h5py.File(self._ffile % fi, 'r')

                # Check frequency is what we expect.
                if fi != ff.attrs['frequency_index']:
                    raise Exception("Bork.")
        
                dset = fgrp.create_dataset(self._fsection % fi, data=ff[self._msection % mi], compression='gzip')
                dset.attrs['frequency_index'] = fi
                ff.close()

            f.attrs['baselines'] = self.cylinder.baselines
            f.attrs['m'] = mi
            f.attrs['frequencies'] = self.cylinder.frequencies
            f.attrs['cylobj'] = self.cylinderpickle
            
            f.close()


