import numpy as np

from cylsim import cylinder
from cylsim import beamtransfer
from simulations import foregroundsck, corr21cm
from utils import units

import scipy.linalg as la

import time

import argparse
import os

from simulations.foregroundmap import matrix_root_manynull

import healpy

from progressbar import ProgressBar

#import matplotlib
#matplotlib.use('PDF')
#from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='MPI program to generate beam matrices frequency by frequency.')
parser.add_argument('rootdir', help='Root directory to create files in.')
parser.add_argument('filestem', default='', help='Prefix to add created files.', nargs='?')

args = parser.parse_args()

mi = 100

print "Reading beam matrices..."

## The filename root of the files we want.
root = args.rootdir + "/" + args.filestem + "beammatrix"

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

cr = corr21cm.Corr21cm()

cv_sg = cr.angular_powerspectrum_fft(np.arange(cyl.lmax+1)[:,np.newaxis,np.newaxis], za[np.newaxis,:,np.newaxis], za[np.newaxis,np.newaxis,:])

beam = bt.beam_m(mi)

#cvb_fg = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)
#cvb_sg = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)
#cvb_nm = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)

print "Constructing signal and noise covariances..."

cvb_s = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)
cvb_n = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)


tsys = 50.0
bw = np.abs(cyl.frequencies[1] - cyl.frequencies[0]) * 1e6
delnu = units.t_sidereal * bw / (2*np.pi)
ndays = 365
noisepower = tsys**2 / (2 * np.pi * delnu * ndays)

print "Noise: T_sys = %f K, Bandwidth %f MHz, %i days. Total %g" % (tsys, bw / 1e6, ndays, noisepower)

noisebase = noisepower * np.diag(1.0 / cyl.redundancy)

progress = ProgressBar()
for fi in progress(range(cyl.nfreq)):
    for fj in range(cyl.nfreq):
        cvb_n[fi,:,fj,:] = np.dot((beam[fi] * cv_fg[:,fi,fj]), beam[fj].conj().T)
        cvb_s[fi,:,fj,:] = np.dot((beam[fi] * cv_sg[:,fi,fj]), beam[fj].conj().T)
    cvb_n[fi,:,fi,:] += noisebase


nside = cyl.nfreq*cyl.nbase

#evals, evecs = la.eigh(cvb_s.reshape((nside,nside)), cvb_n.reshape((nside,nside)))

print "Solving for Eigenvalues...."

cvb_sr = cvb_s.reshape((nside,nside))
cvb_nr = cvb_n.reshape((nside,nside))

st = time.time()
evals, evecs = la.eigh(cvb_sr, cvb_nr, overwrite_a=True, overwrite_b=True)
et=time.time()
print "Time =", (et-st)

st = time.time()
#evals = la.eigvalsh(cvb_sr, cvb_nr, overwrite_a=True, overwrite_b=True)
et=time.time()
print "Time =", (et-st)


st = time.time()
#evals, evecs = la.eigh(cvb_sr, cvb_nr)
et=time.time()
print "Time =", (et-st)


st = time.time()
#evals, evecs = la.eigh(cvb_sr, overwrite_a=True, overwrite_b=True)
et=time.time()
print "Time =", (et-st)


#np.save("evals_save.npy", evals)

#evecs2 = evecs.reshape((cyl.nfreq,cyl.nbase,nside))
