import numpy as np

from simulations import foregroundsck, corr21cm, skysim
from cosmoutils import units

_cr = None


class FullskySynchrotron(foregroundsck.Synchrotron):
    """Increase synchrotron amplitude to see it alleviates the full sky issues."""
    A = 1.00e-2
    


def foreground_model(lmax, frequencies, npol):

    #fsyn = foregroundsck.Synchrotron()
    fsyn = FullskySynchrotron()
    fps = foregroundsck.PointSources()
    
    nfreq = frequencies.size

    cv_fg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_fg[0, 0] = skysim.clarray(fsyn.angular_powerspectrum, lmax, frequencies)

    if npol >= 3:
        cv_fg[1, 1] = 0.3**2 * cv_fg[0, 0]
        cv_fg[2, 2] = 0.3**2 * cv_fg[0, 0]

    cv_fg[0, 0] += skysim.clarray(fps.angular_powerspectrum, lmax, frequencies)
    return cv_fg




def im21cm_model(lmax, frequencies, npol, cr = None):

    nfreq = frequencies.size

    if not cr:
        global _cr
        if not _cr:
            _cr = corr21cm.Corr21cm()
        cr = _cr

    cv_sg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_sg[0, 0] = skysim.clarray(cr.angular_powerspectrum, lmax, frequencies)

    return cv_sg

