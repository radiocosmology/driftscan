
import numpy as np

import h5py

from cylsim import cylinder
from simulations import foregroundsck
import pickle

import argparse
import os

parser = argparse.ArgumentParser(description='MPI program to generate beam matrices frequency by frequency.')
parser.add_argument('rootdir', help='Root directory to create files in.')
parser.add_argument('filestem', default='', help='Prefix to add created files.', nargs='?')

args = parser.parse_args()

mi = 300

## The filename root of the files we want.
root = args.rootdir + "/" + args.filestem + "beammatrix"

## Reconstruct cylinder object
cyl = None
with open(root+"_cylobj.pickle", 'r') as f:
    cyl = pickle.load(f)

nbase = cyl.baselines.shape[0]
nfreq = cyl.frequencies.shape[0]

lmax, mmax = cyl.max_lm()


mbase = root + "_m_%0"+repr(int(np.ceil(np.log10(mmax+1))))+"d.hdf5"

ffmt = "%0"+repr(int(np.ceil(np.log10(nfreq+1))))+"d"

## Construct foreground matrix C[l,nu1,nu2]
fsyn = foregroundsck.Synchrotron()

cv_fg = (fsyn.angular_powerspectrum(np.arange(lmax+1))[:,np.newaxis,np.newaxis]
         * fsyn.frequency_covariance(*np.meshgrid(cyl.frequencies, cyl.frequencies))[np.newaxis,:,:])

beam = np.zeros((nfreq, nbase, lmax+1), dtype=np.complex128)

mfile = h5py.File(mbase % mi, 'r')

for fi in range(nfreq):

    fstr = 'freq_section/' + (ffmt % fi)

    beam[fi] = mfile[fstr]


cvb_fg = np.zeros((nfreq, nbase, nfreq, nbase), dtype=np.complex128)

for fi in range(nfreq):
    for fj in range(nfreq):
        cvb_fg[fi,:,fj,:] = np.dot((beam[fi] * cv_fg[:,fi,fj]), beam[fj].conj().T)

    
