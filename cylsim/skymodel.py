import numpy as np

from simulations import foregroundsck, corr21cm
from utils import units

_cr = None


class FullskySynchrotron(foregroundsck.Synchrotron):
    """Increase synchrotron amplitude to see it alleviates the full sky issues."""
    A = 10000.0
    


def clarray(aps, lmax, zarray):

    clarray = aps(np.arange(lmax+1)[:, np.newaxis, np.newaxis],
                  zarray[np.newaxis, :, np.newaxis], zarray[np.newaxis, np.newaxis, :])

    return clarray



def foreground_model(lmax, frequencies, npol):

    #fsyn = foregroundsck.Synchrotron()
    fsyn = FullskySynchrotron()
    fps = foregroundsck.PointSources()
    
    nfreq = frequencies.size

    cv_fg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_fg[0, 0] = clarray(fsyn.aps, lmax, frequencies) * 1e-6

    if npol >= 3:
        cv_fg[1, 1] = 0.3**2 * cv_fg[0, 0]
        cv_fg[2, 2] = 0.3**2 * cv_fg[0, 0]

    cv_fg[0, 0] += clarray(fps.aps, lmax, frequencies) * 1e-6

    return cv_fg




def im21cm_model(lmax, frequencies, npol, cr = None):

    nfreq = frequencies.size

    if not cr:
        global _cr
        if not _cr:
            _cr = corr21cm.Corr21cm()
        cr = _cr
    
    za = units.nu21 / frequencies - 1.0

    cv_sg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_sg[0, 0] = clarray(cr.angular_powerspectrum_fft, lmax, za) * 1e-6

    return cv_sg

