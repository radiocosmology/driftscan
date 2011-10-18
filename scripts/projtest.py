import numpy as np


import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt

import healpy

from cylsim import beamtransfer

from cylsim import blockla

from cylsim import hputil
from cylsim import util


bt = beamtransfer.BeamTransfer('telescope2/')

bf = bt.beam_freq(0, fullm=True)

u, sig, vh = blockla.svd_dm(np.rollaxis(bf, -1), full_matrices=False)

sigmask = (sig > 1e-12*sig.max()).astype(np.int)

vh2 = vh*sigmask[:,:,np.newaxis]

haslam = healpy.read_map("haslam.fits")

angpos = hputil.ang_positions(healpy.npix2nside(haslam.size))
theta, phi =  healpy.Rotator(coord=['C', 'G'])(angpos[:,0], angpos[:,1])

hasrot = healpy.get_interp_val(haslam, theta, phi)

lmax = bt.telescope.lmax

#h_alm = healpy.map2alm(haslam, lmax=lmax)
#h_alm = hputil.sphtrans_complex(haslam, lmax=lmax).T

hasproj = util.proj_mblock(hasrot, vh2)

fig = plt.figure(1)

healpy.mollview(hasrot, title="Haslam map", fig=1)

fig.savefig("galaxy.pdf")

fig.clf()


healpy.mollview(hasproj, title="Haslam map - projected onto CHIME", fig=1)

fig.savefig("galaxyproj.pdf")


