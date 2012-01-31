
import numpy as np

import h5py

import healpy
from cylsim import hputil
from cylsim import util

from cylsim import skymodel

from simulations import foregroundsck

from os.path import join, dirname


_datadir = join(dirname(__file__), "data")





def galactic_synchrotron(nside, frequencies, debug=False, celestial=True):

    haslam = healpy.ud_grade(healpy.read_map(join(_datadir, "haslam.fits")), nside) #hputil.coord_g2c()
    
    h_nside = healpy.npix2nside(haslam.size)

    f = h5py.File(join(_datadir, 'skydata.hdf5'), 'r')

    s400 = healpy.ud_grade(f['/sky_400MHz'][:], nside)
    s800 = healpy.ud_grade(f['/sky_800MHz'][:], nside)
    nh = 512
    beam = 1.0

    f.close()

    syn = foregroundsck.Synchrotron()

    lmax = 3*nside - 1

    efreq = np.concatenate((np.array([400.0, 800.0]), frequencies))

    cla = skymodel.clarray(syn.aps, lmax, efreq) * 1e-6

    fg = util.mkfullsky(cla, nside)

    sub4 = healpy.smoothing(fg[0], sigma=beam, degree=True)
    sub8 = healpy.smoothing(fg[1], sigma=beam, degree=True)
    
    fgs = util.mkconstrained(cla, [(0, sub4), (1, sub8)], nside)

    fgt = fg - fgs

    sc = np.log(s800 / s400) / np.log(2.0)

    fg2 = (haslam[np.newaxis, :] * (((efreq / 400.0)[:, np.newaxis]**sc) + (0.25 * fgt / fgs[0].std())))[2:]

    if celestial:
        for i in range(fg2.shape[0]):
            fg2[i] = hputil.coord_g2c(fg2[i])
    
    if debug:
        return fg2, fgt, fg, fgs, sc, sub4, sub8, s400, s800
    else:
        return fg2

    
c_syn = galactic_synchrotron
