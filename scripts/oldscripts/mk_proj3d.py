from math import log10, floor
import sys


import h5py
import healpy
import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"], "size": 10.0})
rc("text", usetex=True)

from drift.core import hputil, skysim

mapfile = sys.argv[1]
stem = sys.argv[2]

f = h5py.File(mapfile, "r")


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(np.abs(x)))) - 1)


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


angslice = {}

skyslice = {}

arrmax = {}
arrmin = {}

for mapname in f.keys():

    sky = f[mapname][:]

    if len(sky.shape) == 3:
        sky = hputil.sphtrans_inv_sky(sky, 128).real
    else:
        sky = sky.real

    aslice = angular_slice(sky, theta, phi).real

    skyslice[mapname] = sky[0]
    angslice[mapname] = aslice

    arrmax[mapname] = max(aslice.max(), sky[0].max())
    arrmin[mapname] = max(aslice.min(), sky[0].min())

global_scale = True


for mapname in f.keys():

    fig = plt.figure(1, figsize=(13, 5))

    ax2 = fig.add_subplot(122)

    if global_scale:
        vmax = round_sig(max(arrmax.values()))
        vmin = round_sig(min(arrmin.values()))
    else:
        vmax = round_sig(arrmax[mapname])
        vmin = round_sig(arrmin[mapname])

    healpy.mollview(
        skyslice[mapname], fig=1, sub=121, title="400 MHz", min=vmin, max=vmax
    )

    ax2.imshow(
        angslice[mapname],
        interpolation="lanczos",
        aspect=1,
        origin="lower",
        extent=(45, 90, 400, 450),
        vmin=vmin,
        vmax=vmax,
    )

    ax2.set_xlabel("$\phi$ / degrees")
    ax2.set_ylabel("$f$ / MHz")

    fig.savefig(stem + "_" + mapname + ".png")
    fig.clf()
