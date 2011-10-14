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

from progressbar import ProgressBar

parser = argparse.ArgumentParser(description='MPI program to generate beam matrices frequency by frequency.')
parser.add_argument('rootdir', help='Root directory to create files in.')

args = parser.parse_args()

print "Reading beam matrices..."

bt = beamtransfer.BeamTransfer(args.rootdir)

cyl = bt.telescope

print "Constructing C_l(z,z') matrices...."

## Construct foreground matrix C[l,nu1,nu2]
fsyn = foregroundsck.Synchrotron()
fps = foregroundsck.PointSources()

cv_fg = (fsyn.angular_powerspectrum(np.arange(cyl.lmax+1))[:,np.newaxis,np.newaxis]
         * fsyn.frequency_covariance(*np.meshgrid(cyl.frequencies, cyl.frequencies))[np.newaxis,:,:])

cv_fg += (fps.angular_powerspectrum(np.arange(cyl.lmax+1))[:,np.newaxis,np.newaxis]
         * fps.frequency_covariance(*np.meshgrid(cyl.frequencies, cyl.frequencies))[np.newaxis,:,:])

za = units.nu21 / cyl.frequencies - 1.0


## Construct signal matrix C_l(nu, nu')
cr = corr21cm.Corr21cm()

cv_sg = cr.angular_powerspectrum_fft(np.arange(cyl.lmax+1)[:,np.newaxis,np.newaxis], za[np.newaxis,:,np.newaxis], za[np.newaxis,np.newaxis,:])

## Construct thermal noise
tsys = 50.0
bw = np.abs(cyl.frequencies[1] - cyl.frequencies[0]) * 1e6
delnu = units.t_sidereal * bw / (2*np.pi)
ndays = 365
noisepower = tsys**2 / (2 * np.pi * delnu * ndays)
noisebase = noisepower * np.diag(1.0 / cyl.redundancy)
print "Noise: T_sys = %f K, Bandwidth %f MHz, %i days. Total %g" % (tsys, bw / 1e6, ndays, noisepower)

ev_pat = args.rootdir + "/ev_" + util.intpattern(cyl.mmax) + ".hdf5"

nside = cyl.nfreq*cyl.nbase

# Iterate list over MPI processes.
#for mi in mpiutil.mpirange(-cyl.mmax, cyl.mmax+1):
for mi in mpiutil.mpirange(3):

    beam = bt.beam_m(mi)

    print "Constructing signal and noise covariances for m = %i ..." % (mi)

    cvb_s = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)
    cvb_n = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)

    for fi in range(cyl.nfreq):
        for fj in range(cyl.nfreq):
            cvb_n[fi,:,fj,:] = np.dot((beam[fi] * cv_fg[:,fi,fj]), beam[fj].conj().T)
            cvb_s[fi,:,fj,:] = np.dot((beam[fi] * cv_sg[:,fi,fj]), beam[fj].conj().T)
        cvb_n[fi,:,fi,:] += noisebase
    
    print "Solving for Eigenvalues...."

    cvb_sr = cvb_s.reshape((nside,nside))
    cvb_nr = cvb_n.reshape((nside,nside))

    st = time.time()

    add_const = 0.0

    try:
        evals, evecs = la.eigh(cvb_sr, cvb_nr, overwrite_a=True, overwrite_b=True)
        break
    except LinAlgError:
        print "Matrix probabaly not positive definite due to numerical issues. Trying to a constant...."
        
        add_const = -la.eigvalsh(cvb_nr, eigvals=(0,0))[0] * 1.1
        
        cvb_nr[np.diag_indices(nside)] += add_const
        evals, evecs = la.eigh(cvb_sr, cvb_nr, overwrite_a=True, overwrite_b=True)
        
    et=time.time()
    print "Time =", (et-st)


    ## Write out Eigenvals and Vectors
    print "Creating file %s ...." % (ev_pat % mi)
    f = h5py.File(ev_pat % mi, 'w')
    f.attrs['m'] = mi

    f.create_dataset('evals', data=evals)
    f.create_dataset('evecs', data=evecs, compression='gzip')

    if add_const != 0.0:
        f.attrs['add_const'] = add_const
        f.attrs['FLAGS'] = 'NotPositiveDefinite'
    else:
        f.attrs['FLAGS'] = 'Normal'

    f.close()

