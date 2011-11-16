import numpy as np

from simulations import foregroundsck, corr21cm
from utils import units


def clarray(aps, lmax, zarray):

    clarray = aps(np.arange(lmax+1)[:, np.newaxis, np.newaxis],
                  zarray[np.newaxis, :, np.newaxis], zarray[np.newaxis, np.newaxis, :])

    return clarray



def foreground_model(lmax, frequencies, npol):

    fsyn = foregroundsck.Synchrotron()
    fps = foregroundsck.PointSources()
    
    nfreq = frequencies.size

    cv_fg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_fg[0, 0] = clarray(fsyn.aps, lmax, frequencies) * 1e-6

    if npol >= 3:
        cv_fg[1, 1] = 0.3**2 * cv_fg[0, 0]
        cv_fg[2, 2] = 0.3**2 * cv_fg[0, 0]

    cv_fg[0, 0] += clarray(fps.aps, lmax, frequencies) * 1e-6

    return cv_fg




def im21cm_model(lmax, frequencies, npol):

    nfreq = frequencies.size
    
    cr = corr21cm.Corr21cm()
    za = units.nu21 / frequencies - 1.0

    cv_sg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_sg[0, 0] = clarray(cr.angular_powerspectrum_fft, lmax, za) * 1e-6

    return cv_sg

