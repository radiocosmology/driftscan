
import numpy as np

import h5py

import healpy
from cylsim import hputil
from cylsim import util

from cylsim import skymodel

from simulations import foregroundsck

from os.path import join, dirname

#_haslam = hputil.coord_g2c(healpy.read_map("haslam.fits")) * 1e3
#_h_nside = healpy.npix2nside(_haslam.size)

_datadir = join(dirname(__file__), "data")

def constrained_syn(nside, frequencies):

    syn = foregroundsck.Synchrotron()

    lmax = 3*nside - 1

    cla = skymodel.clarray(syn.aps, lmax, np.concatenate((np.array([408.0]), frequencies)))

    fg = util.mkfullsky(cla, nside)
    

    sub = healpy.ud_grade(healpy.ud_grade(fg[0], _h_nside), nside)

    fg2 = (fg[1:] - sub[np.newaxis, :]) + (_haslam[np.newaxis, :] * ((frequencies / 408.0)**(-syn.alpha))[:, np.newaxis])

    #return fg

    return fg, fg2, sub, _haslam





def c_syn(nside, frequencies, debug=False):

    f = h5py.File(join(_datadir, 'skydata.hdf5'), 'r')

    s400 = healpy.ud_grade(f['/sky_400MHz'][:], nside)
    s800 = healpy.ud_grade(f['/sky_800MHz'][:], nside)
    nh = 512
    beam = 1.0

    f.close()

    syn = foregroundsck.Synchrotron()

    lmax = 3*nside - 1

    #efreq = np.concatenate((np.array([400.0, 800.0]), frequencies))

    cla = skymodel.clarray(syn.aps, lmax, frequencies) * 1e-6

    fg = util.mkfullsky(cla, nside)

    sub4 = healpy.smoothing(fg[0], sigma=beam, degree=True)
    sub8 = healpy.smoothing(fg[-1], sigma=beam, degree=True)
    
    fgs = util.mkconstrained(cla, [(0, sub4), (-1, sub8)], nside)

    fgt = fg - fgs

    sc = np.log(s800 / s400) / np.log(2.0)

    fg2 = s400[np.newaxis, :] * (((frequencies / 400.0)[:, np.newaxis]**sc) + (0.5 * fgt / fgs[0].std()))
    
    if debug:
        return fg2, fgt, fg, fgs, sc, sub4, sub8, s400, s800
    else:
        return fg2

    

    

    
