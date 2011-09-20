import pickle

import numpy as np

import h5py

from cylsim import cylinder

def _intpattern(n):
    """Pattern that prints out a number upto `n`."""
    return ("%+0" + repr(int(np.ceil(np.log10(mmax+1)))+1) + "d")



class BeamTransfer(object):
    """A class for reading and writing Beam Transfer matrices from disk."""

    @property
    def _cylfilename(self):
        return self.directory + "/cylinderobject.pickle"

    @property
    def _mfile(self):
        return self.directory + "beam_m_"+_intpattern(mmax)+"d.hdf5"

    @property
    def _ffile(self):
        return self.directory + "beam_f_"+_intpattern(nfreq)+"d.hdf5"



    def __init__(self, directory, cylinder = None):

        self.directory = directory

        self.cylinder = cylinder

        if self.cylinder == None:
            print "Attempting to read cylinder from disk..."

            with open(self._cylfilename, 'r') as f:
                self.cylinder = pickle.load(f)
                

        
        
    def beam_m(root, mi):

        cyl = get_cylinder(root)
        
        nbase = cyl.baselines.shape[0]
        nfreq = cyl.frequencies.shape[0]
    
        lmax, mmax = cyl.max_lm()
        mbase = root + "_m_%+0"+repr(int(np.ceil(np.log10(mmax+1)))+1)+"d.hdf5"

        ffmt = "%0"+repr(int(np.ceil(np.log10(nfreq+1))))+"d"
    
        beam = np.zeros((nfreq, nbase, lmax+1), dtype=np.complex128)
        
        mfile = h5py.File(mbase % mi, 'r')

        for fi in range(nfreq):
            
            fstr = 'freq_section/' + (ffmt % fi)
        
            beam[fi] = mfile[fstr]

        return beam



def beam_freq(root, fi):

    cyl = get_cylinder(root)

    nbase = cyl.baselines.shape[0]
    nfreq = cyl.frequencies.shape[0]

    lmax, mmax = cyl.max_lm()
    fbase = root + "_freq_%0"+repr(int(np.ceil(np.log10(nfreq+1))))+"d.hdf5"

    mfmt = "%+0"+repr(int(np.ceil(np.log10(mmax+1)))+1)+"d"
    
    
    beam = np.zeros((2*mmax+1, nbase, lmax+1), dtype=np.complex128)

    ffile = h5py.File(fbase % fi, 'r')

    for mi in range(-mmax,mmax+1):

        mstr = 'm_section/' + (mfmt % mi)
        
        beam[mi] = ffile[mstr]

    return beam


