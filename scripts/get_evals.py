import numpy as np

from cylsim import cylinder
from cylsim import beamtransfer
from cylsim import util
from simulations import foregroundsck, corr21cm
from utils import units

import scipy.linalg as la

import time

import h5py

import argparse
import os

from simulations.foregroundmap import matrix_root_manynull

import healpy

from cylsim import mpiutil


parser = argparse.ArgumentParser(description='MPI program to generate beam matrices frequency by frequency.')
parser.add_argument('rootdir', help='Root directory to create files in.')

args = parser.parse_args()

print "Reading beam matrices..."

bt = beamtransfer.BeamTransfer(args.rootdir)

cyl = bt.telescope

ev_pat = args.rootdir + "/ev_" + util.intpattern(cyl.mmax) + ".hdf5"

nside = cyl.nfreq*cyl.nbase

evarray = np.zeros((2*cyl.mmax+1, nside))
acarray = np.zeros(2*cyl.mmax+1)
# Iterate list over MPI processes.
for mi in mpiutil.mpirange(-cyl.mmax, cyl.mmax+1):

    print "Reading evals for m = %i" % mi

    f = h5py.File(ev_pat % mi, 'r')

    evals = f['evals']
    ac = f.attrs['add_const'] if 'add_const' in f.attrs else 0.0

    evarray[mi] = evals
    acarray[mi] = ac

    f.close()



f = h5py.File(args.rootdir + '/evals.hdf5', 'w')

f.create_dataset('evals', data=evarray)
f.create_dataset('add_const', data=acarray)

f.close()


