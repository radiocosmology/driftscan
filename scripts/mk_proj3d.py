

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


import sys

from cylsim import hputil, skysim

import h5py

import healpy

import numpy as np

mapfile = sys.argv[1]
stem = sys.argv[2]

f = h5py.File(mapfile, 'r')

def angular_slice(skymaps, theta, phi):
    nfreq = skymaps.shape[0]
    za = np.zeros((nfreq, phi.size), dtype=np.complex128)

    for i in range(nfreq):
        za[i] = healpy.get_interp_val(skymaps[i], theta, phi)
        
    return za


nphi = 100
phi0, phi1 = np.pi / 4, np.pi / 2.0
phi = np.linspace(phi0, phi1, nphi)
theta = np.pi / 2.0

for mapname in f.keys():

    sky = f[mapname][:]

    fig = plt.figure(1, figsize=(13,5))

    ax2 = fig.add_subplot(122)

    healpy.mollview(sky[0], fig=1, sub=121, title="")

    ax2.imshow(angular_slice(sky, theta, phi).real, interpolation='lanczos', aspect=1, origin='lower', extent=(45, 90, 400, 450))

    ax2.set_xlabel("$\phi$ / degrees")
    ax2.set_ylabel("$f$ / MHz")


    fig.savefig(stem + "_" + mapname + ".png")
    fig.clf()
