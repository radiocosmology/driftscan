import pickle
import numpy as np

from cylsim import cylinder

import h5py

import scipy.linalg

def get_cylinder(root):
    with open(root+"_cylobj.pickle", 'r') as f:
        cyl = pickle.load(f)

    return cyl

def beam_m(root, mi):

    cyl = get_cylinder(root)

    nbase = cyl.baselines.shape[0]
    nfreq = cyl.frequencies.shape[0]

    lmax, mmax = cyl.max_lm()
    mbase = root + "_m_%0"+repr(int(np.ceil(np.log10(mmax+1))))+"d.hdf5"

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

    mfmt = "%0"+repr(int(np.ceil(np.log10(mmax+1))))+"d"
    
    
    beam = np.zeros((mmax+1, nbase, lmax+1), dtype=np.complex128)

    ffile = h5py.File(fbase % fi, 'r')

    for mi in range(mmax):

        mstr = 'm_section/' + (mfmt % mi)
        
        beam[mi] = ffile[mstr]

    return beam


def block_svd(matrix, full_matrices=True):

    nblocks, n, m = matrix.shape
    dt = matrix.dtype
    k = min(n, m)
    sig = np.zeros((nblocks, k), dtype=dt)
    
    if full_matrices:
        u = np.zeros((nblocks, n, n), dtype=dt)
        v = np.zeros((nblocks, m, m), dtype=dt)
    else:
        u = np.zeros((nblocks, n, k), dtype=dt)
        v = np.zeros((nblocks, k, m), dtype=dt)


    for ib in range(nblocks):
        u[ib], sig[ib], v[ib] = scipy.linalg.svd(matrix[ib], full_matrices=full_matrices)

    return u, sig, v


def block_mv(matrix, vector, conj=False):

    if conj:
        nblocks, m, n = matrix.shape
    else:
        nblocks, n, m = matrix.shape

    if vector.shape != (nblocks, m):
        raise Exception("Shapes not compatible.")

    # Check dtype
    if conj:
        dt = np.dot(matrix[0].T.conj(), vector[0]).dtype
    else:
        dt = np.dot(matrix[0], vector[0]).dtype

    nvector = np.empty((nblocks, n), dtype=dt)

    if conj:
        for i in range(nblocks):
            nvector[i] = np.dot(matrix[i].T.conj(), vector[i])
    else:
        for i in range(nblocks):
            nvector[i] = np.dot(matrix[i], vector[i])

    return nvector
