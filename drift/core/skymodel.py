import numpy as np

from cora.core import skysim
from cora.signal import corr21cm
from cora.foreground import gaussianfg, galaxy

_cr = None

_reionisation = False


class PointSources(gaussianfg.PointSources):
    """Scale up point source amplitude to a higher S_{cut} = 0.1 Jy"""

    A = 3.55e-5
    nu_0 = 408.0
    l_0 = 100.0


def foreground_model(lmax, frequencies, npol, pol_frac=1.0, pol_length=None):
    fsyn = galaxy.FullSkySynchrotron()
    fps = PointSources()

    nfreq = frequencies.size

    cv_fg = np.zeros((npol, npol, lmax + 1, nfreq, nfreq))

    cv_fg[0, 0] = skysim.clarray(fsyn.angular_powerspectrum, lmax, frequencies)

    if npol >= 3:
        fpol = galaxy.FullSkyPolarisedSynchrotron()

        if pol_length is not None:
            fpol.zeta = pol_length

        cv_fg[1, 1] = pol_frac * skysim.clarray(
            fpol.angular_powerspectrum, lmax, frequencies
        )
        cv_fg[2, 2] = pol_frac * skysim.clarray(
            fpol.angular_powerspectrum, lmax, frequencies
        )

    cv_fg[0, 0] += skysim.clarray(fps.angular_powerspectrum, lmax, frequencies)
    return cv_fg


def im21cm_model(lmax, frequencies, npol, cr=None, temponly=False):
    nfreq = frequencies.size

    if not cr:
        global _cr
        if not _cr:
            if not _reionisation:
                _cr = corr21cm.Corr21cm()
            else:
                _cr = corr21cm.EoR21cm()
        cr = _cr

    # cr._freq_window = np.abs(cr.cosmology.comoving_distance(frequencies[0]) - cr.cosmology.comoving_distance(frequencies[1]))

    cv_t = skysim.clarray(cr.angular_powerspectrum, lmax, frequencies)

    if temponly:
        return cv_t
    else:
        cv_sg = np.zeros((npol, npol, lmax + 1, nfreq, nfreq))
        cv_sg[0, 0] = cv_t
        return cv_sg
