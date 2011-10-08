import numpy as np

from cylsim import cylinder
from cylsim import beamtransfer
from simulations import foregroundsck, corr21cm
from utils import units

import scipy.linalg as la

import argparse
import os

from simulations.foregroundmap import matrix_root_manynull

import healpy

import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='MPI program to generate beam matrices frequency by frequency.')
parser.add_argument('rootdir', help='Root directory to create files in.')
parser.add_argument('filestem', default='', help='Prefix to add created files.', nargs='?')

args = parser.parse_args()

mi = 100

## The filename root of the files we want.
root = args.rootdir + "/" + args.filestem + "beammatrix"

bt = beamtransfer.BeamTransfer(args.rootdir)

cyl = bt.telescope

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

cvb_fg = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)
cvb_sg = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)
cvb_nm = np.zeros((cyl.nfreq, cyl.nbase, cyl.nfreq, cyl.nbase), dtype=np.complex128)

noisebase = 1e-9 * np.diag(1.0 / cyl.redundancy)

for fi in range(cyl.nfreq):
    for fj in range(cyl.nfreq):
        cvb_fg[fi,:,fj,:] = np.dot((beam[fi] * cv_fg[:,fi,fj]), beam[fj].conj().T)
        cvb_sg[fi,:,fj,:] = np.dot((beam[fi] * cv_sg[:,fi,fj]), beam[fj].conj().T)
    cvb_nm[fi,:,fi,:] = noisebase


nside = cyl.nfreq*cyl.nbase

evals, evecs = la.eigh(cvb_sg.reshape((nside,nside)), (cvb_fg + cvb_nm).reshape((nside,nside)))
    
evecs2 = evecs.reshape((cyl.nfreq,cyl.nbase,nside))


fg_ch = np.zeros_like(cv_fg)
sg_ch = np.zeros_like(cv_sg)

print "Rotating matrices..."
for i in range(fg_ch.shape[0]):
    #fg_ch[i] = np.linalg.cholesky(cv_fg[i])
    #sg_ch[i] = np.linalg.cholesky(cv_sg[i])
    fg_ch[i] = matrix_root_manynull(cv_fg[i], truncate = False)
    sg_ch[i] = matrix_root_manynull(cv_sg[i], truncate = False)


la, ma = healpy.Alm.getlm(cyl.lmax)
larr = la
marr = ma

print "Generating random variables..."
gv_fg = (np.random.standard_normal(la.shape + (cyl.nfreq,)) + 1.0J * np.random.standard_normal(la.shape + (cyl.nfreq,))) / 2.0**0.5
gv_sg = (np.random.standard_normal(la.shape + (cyl.nfreq,)) + 1.0J * np.random.standard_normal(la.shape + (cyl.nfreq,))) / 2.0**0.5

print "Transforming variables..."
for i, l in enumerate(la):
    gv_fg[i] = np.dot(fg_ch[l], gv_fg[i])
    gv_sg[i] = np.dot(sg_ch[l], gv_sg[i])

del fg_ch

#nside = cyl._best_nside(cyl.lmax)
#npix = healpy.nside2npix(nside)

#map_fg = np.empty((cyl.nfreq, npix))
#map_sg = np.empty((cyl.nfreq, npix))

print "Making maps..."

for i in range(cyl.nfreq):
    pass
#    map_fg[i,:] = healpy.alm2map(gv_fg[:,i].copy(), nside)
#    map_sg[i,:] = healpy.alm2map(gv_sg[:,i].copy(), nside)


#alm = np.zeros((cyl.lmax+1, cyl.lmax+1), dtype=np.complex128)
#alm[np.triu_indices(cyl.lmax+1)] = gv_fg[:,0]




mapfgm = np.zeros((cyl.nfreq, cyl.lmax+1), dtype=np.complex128)
mapfgm[:,100:] = gv_fg[np.where(ma == mi),:][0].T
projfg = loadbeam.block_mv(beam, mapfgm)


mapsgm = np.zeros((cyl.nfreq, cyl.lmax+1), dtype=np.complex128)
mapsgm[:,100:] = gv_sg[np.where(ma == mi),:][0].T
projsg = loadbeam.block_mv(beam, mapsgm)

projnm = (np.random.standard_normal((cyl.nfreq, cyl.nbase)) + 1.0J * np.random.standard_normal((cyl.nfreq, cyl.nbase)) ) * (cvb_nm.reshape((nside, nside)).diagonal().reshape((cyl.nfreq, cyl.nbase)) / 2.0)**0.5


ma_fg = np.dot(evecs.T.conj(), projfg.flat)
ma_sg = np.dot(evecs.T.conj(), projsg.flat)
ma_nm = np.dot(evecs.T.conj(), projnm.flat)


f = plt.figure(1)
ax = f.add_subplot(111)

ax.set_xlabel("Mode number")
ax.set_title("Signal/Noise amplitude")

ax.plot(evals**0.5)

f.savefig("evals.pdf")

ax.plot(np.abs(ma_fg))
f.savefig("evals+fg.pdf")

ax.plot(np.abs(ma_nm))
f.savefig("evals+fg+nm.pdf")

ax.plot(np.abs(ma_sg))
f.savefig("evals+fg+nm+sg.pdf")



if False:
    bf = loadbeam.beam_freq(root, 0)

    from cylsim import hputil
    import healpy

    u, sig, v = loadbeam.block_svd(bf, full_matrices=False)

    del u, sig

    v2 = np.zeros((2*cyl.lmax+1, cyl.nbase, cyl.lmax+1), dtype=np.complex128)

    v2[:mmax+1] = v[:mmax+1]
    v2[-mmax:] = v[-mmax:]

    del v

    almhalf = np.zeros((cyl.lmax+1, cyl.lmax+1), dtype=np.complex128)
    almhalf[np.triu_indices(cyl.lmax+1)] = gv_fg[:,0]

    almf = hputil._make_full_alm(almhalf).T.copy()

    almfproj = loadbeam.block_mv(v2, loadbeam.block_mv(v2, almf), conj=True)

    almhalf2 = np.zeros_like(almhalf)

    almhalf2[:,0] = almfproj[0]

    for mi in range(1,mmax+1):
        almhalf2[:,mi] = 0.5*(almfproj[mi] + (-1)**mi * almfproj[-mi].conj())

        almfold = almhalf2[np.triu_indices(cyl.lmax+1)]

    nside = cyl._best_nside(cyl.lmax)
    mapfull = healpy.alm2map(gv_fg[:,0].copy(), nside)
    mapproj = healpy.alm2map(almfold, nside)

