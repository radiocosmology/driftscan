import numpy as np

from simulations import foregroundsck, corr21cm, skysim
from cosmoutils import units

_cr = None


class FullskySynchrotron(foregroundsck.Synchrotron):
    """Match up Synchrotron amplitudes to thise found in La Porta et al. 2008,
    for galactic latitudes abs(b) > 5 degrees"""
    A = 6.6e-3
    beta = 2.8
    nu_0 = 408.0
    l_0 = 100.0

class PointSources(foregroundsck.PointSources):
    """Scale up point source amplitude to a higher S_{cut} = 0.1 Jy"""
    A = 3.55e-5
    nu_0 = 408.0
    l_0 = 100.0


def foreground_model(lmax, frequencies, npol):

    fsyn = FullskySynchrotron()
    fps = PointSources()
    
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

