import numpy as np

from simulations import foregroundsck, corr21cm, skysim, galaxy
from cosmoutils import units

_cr = None

_reionisation = False

_zaverage = 5


def clarray(aps, lmax, zarray, zaverage=None):

    if zaverage == None:
        zaverage = _zaverage

    if zaverage == 1:
        return aps(np.arange(lmax+1)[:, np.newaxis, np.newaxis],
                   zarray[np.newaxis, :, np.newaxis], zarray[np.newaxis, np.newaxis, :])
    else:
        zhalf = np.abs(zarray[1] - zarray[0]) / 2.0
        zlen = zarray.size
        za = (zarray[:, np.newaxis] + np.linspace(-zhalf, zhalf, zaverage)[np.newaxis, :]).flatten()

        lsections = np.array_split(np.arange(lmax+1), lmax / 50)

        cla = np.zeros((lmax+1, zlen, zlen), dtype=np.float64)

        for lsec in lsections:
            clt = aps(lsec[:, np.newaxis, np.newaxis],
                      za[np.newaxis, :, np.newaxis], za[np.newaxis, np.newaxis, :])

            cla[lsec] = clt.reshape((-1, zlen, zaverage, zlen, zaverage)).sum(axis=4).sum(axis=2) / zaverage**2

        return cla


class PointSources(foregroundsck.PointSources):
    """Scale up point source amplitude to a higher S_{cut} = 0.1 Jy"""
    A = 3.55e-5
    nu_0 = 408.0
    l_0 = 100.0


def foreground_model(lmax, frequencies, npol):

    fsyn = galaxy.FullSkySynchrotron()
    fps = PointSources()
    
    nfreq = frequencies.size

    cv_fg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_fg[0, 0] = skysim.clarray(fsyn.angular_powerspectrum, lmax, frequencies)

    if npol >= 3:
        cv_fg[1, 1] = 0.3**2 * cv_fg[0, 0]
        cv_fg[2, 2] = 0.3**2 * cv_fg[0, 0]

    cv_fg[0, 0] += clarray(fps.angular_powerspectrum, lmax, frequencies)
    return cv_fg




def im21cm_model(lmax, frequencies, npol, cr = None):

    nfreq = frequencies.size

    if not cr:
        global _cr
        if not _cr:
            _cr = corr21cm.Corr21cm()
        cr = _cr

    cr._freq_window = np.abs(cr.cosmology.comoving_distance(frequencies[0]) - cr.cosmology.comoving_distance(frequencies[1]))

    cv_sg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_sg[0, 0] = clarray(cr.angular_powerspectrum, lmax, frequencies)

    if _reionisation:
        cv_sg[0, 0] = 1e5 * cv_sg[0, 0]

    return cv_sg

