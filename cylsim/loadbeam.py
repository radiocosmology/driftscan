import pickle
import numpy as np

from cylsim import cylinder

import h5py

def get_cylinder(root):
    with open(root+"_cylobj.pickle", 'r') as f:
        cyl = pickle.load(f)

    return cyl

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


