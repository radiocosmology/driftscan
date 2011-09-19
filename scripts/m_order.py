


from mpi4py import MPI

import numpy as np
import h5py

from cylsim import cylinder

import sys
import os.path
import glob

import argparse
import os

import pickle

## Set up argument parser.
parser = argparse.ArgumentParser(description='MPI program to create m-ordered beam files, from frequency ordered.')
parser.add_argument('rootdir', help='Root directory to create files in.')
parser.add_argument('filestem', default='', help='Prefix to add created files.', nargs='?')
args = parser.parse_args()


## The filename root of the files we want.
root = args.rootdir + "/" + args.filestem + "beammatrix"

## Reconstruct cylinder object
cyl = None
with open(root+"_cylobj.pickle", 'r') as f:
    cyl = pickle.load(f)
    
cylpickle = pickle.dumps(cyl)


lmax, mmax = cyl.max_lm()

nfreq = cyl.frequencies.shape[0]

## Construct patterns for filenames
freqbase = root + "_freq_%0"+repr(int(np.ceil(np.log10(nfreq+1))))+"d.hdf5"
mbase = root + "_m_%+0"+repr(int(np.ceil(np.log10(mmax+1))) + 1)+"d.hdf5"

mfmt = "%+0"+repr(int(np.ceil(np.log10(mmax+1))) + 1)+"d"
ffmt = "%0"+repr(int(np.ceil(np.log10(nfreq+1))))+"d"

## Setup MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



data = None
if rank == 0:
    print "======= Distributing across %i processes =======\n" % size
    print " Splitting into %i m-modes from %i frequencies." % (mmax, nfreq)

    ## Calculate m-modes for each process to work on.
    m_ind = np.arange(-mmax, mmax+1)
    data = [m_ind[i::size] for i in range(size)]  # Split into alternating sublists

    
    
## Scatter m-modes to each process
local_mi = comm.scatter(data, root=0)

for mi in local_mi:

    fname = mbase % mi
    print 'm index %i. Creating file: %s' % (mi, fname)

    ## Create hdf5 file for each m-mode
    f = h5py.File(fname, 'w')
    fgrp = f.create_group('freq_section')

    ## For each frequency read in the current m-mode and copy into file.
    for fi in range(nfreq):
        freqfile = freqbase % fi
        ff = h5py.File(freqfile, 'r')

        # Check frequency is what we expect.
        if fi != ff.attrs['frequency_index']:
            raise Exception("Bork.")
        
        mstr = "m_section/" + (mfmt % mi)
        dset = fgrp.create_dataset((ffmt % fi), data=ff[mstr],  compression='gzip')
        dset.attrs['frequency_index'] = fi
        ff.close()

    f.attrs['baselines'] = cyl.baselines
    f.attrs['m'] = mi
    f.attrs['frequencies'] = cyl.frequencies
    f.attrs['cylobj'] = cylpickle
        
    f.close()

print "=== Process %i done. ===" % rank
