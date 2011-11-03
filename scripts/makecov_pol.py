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
parser.add_argument('evdir', help='Directory to save evals/evecs.', default='', nargs='?')
args = parser.parse_args()

print "Reading beam matrices..."

bt = beamtransfer.BeamTransfer(args.rootdir)

cyl = bt.telescope

print "Constructing C_l(z,z') matrices...."

## Construct foreground matrix C[l,nu1,nu2]
fsyn = foregroundsck.Synchrotron()
fps = foregroundsck.PointSources()

cv_fg = np.zeros((3, cyl.lmax+1, cyl.nfreq, cyl.nfreq))
cv_sg = np.zeros((3, cyl.lmax+1, cyl.nfreq, cyl.nfreq))

cv_fg[0] = (fsyn.angular_powerspectrum(np.arange(cyl.lmax+1))[:,np.newaxis,np.newaxis]
         * fsyn.frequency_covariance(*np.meshgrid(cyl.frequencies, cyl.frequencies))[np.newaxis,:,:] * 1e-6)

cv_fg[1] = 0.3**2 * (fsyn.angular_powerspectrum(np.arange(cyl.lmax+1))[:,np.newaxis,np.newaxis]
         * fsyn.frequency_covariance(*np.meshgrid(cyl.frequencies, cyl.frequencies))[np.newaxis,:,:] * 1e-6)

cv_fg[2] = cv_fg[1]
        

cv_fg[0] += (fps.angular_powerspectrum(np.arange(cyl.lmax+1))[:,np.newaxis,np.newaxis]
         * fps.frequency_covariance(*np.meshgrid(cyl.frequencies, cyl.frequencies))[np.newaxis,:,:] * 1e-6)




## Construct signal matrix C_l(nu, nu')
cr = corr21cm.Corr21cm()
za = units.nu21 / cyl.frequencies - 1.0
cv_sg[0] = cr.angular_powerspectrum_fft(np.arange(cyl.lmax+1)[:,np.newaxis,np.newaxis], za[np.newaxis,:,np.newaxis], za[np.newaxis,np.newaxis,:]) * 1e-6



ev_pat = args.rootdir + "/" + args.evdir + "/ev_" + util.intpattern(cyl.mmax) + ".hdf5"



ndays = 730

nsky = 3 * (cyl.lmax + 1)
ntel = 3 * cyl.nbase

nside = cyl.nfreq * ntel

# Iterate list over MPI processes.
for mi in mpiutil.mpirange(-cyl.mmax, cyl.mmax+1):
#for mi in [-100]:

    st = time.time()
    beam = bt.beam_m(mi)

    print "Constructing signal and noise covariances for m = %i ..." % (mi)

    cvb_s = np.zeros((cyl.nfreq, ntel, cyl.nfreq, ntel), dtype=np.complex128)
    cvb_n = np.zeros((cyl.nfreq, ntel, cyl.nfreq, ntel), dtype=np.complex128)

    for fi in range(cyl.nfreq):
        for fj in range(cyl.nfreq):
            cvb_n[fi,:,fj,:] = np.dot((beam[fi] * cv_fg[..., fi, fj]).reshape((ntel, nsky)), beam[fj].reshape((ntel, nsky)).T.conj())
            cvb_s[fi,:,fj,:] = np.dot((beam[fi] * cv_sg[..., fi, fj]).reshape((ntel, nsky)), beam[fj].reshape((ntel, nsky)).T.conj())
            
        noisebase = np.diag(cyl.noisepower(np.arange(cyl.nbase), fi, ndays).reshape(ntel))
        cvb_n[fi,:,fi,:] += noisebase
    
    print "Solving for Eigenvalues...."

    cvb_sr = cvb_s.reshape((nside,nside))
    cvb_nr = cvb_n.reshape((nside,nside))
        
    et=time.time()
    print "Time =", (et-st)

    st = time.time()

    add_const = 0.0


    try:
        evals, evecs = la.eigh(cvb_sr, cvb_nr, overwrite_a=True, overwrite_b=True)
    except la.LinAlgError:
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

