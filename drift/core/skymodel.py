import numpy as np

import scipy.integrate

from simulations import foregroundsck, corr21cm, skysim, galaxy
from cosmoutils import units

_cr = None

_reionisation = False

# _zaverage = 5

# _endpoint = False


# def clarray(aps, lmax, zarray, zaverage=None):

#     if zaverage == None:
#         zaverage = _zaverage

#     if zaverage == 1:
#         return aps(np.arange(lmax+1)[:, np.newaxis, np.newaxis],
#                    zarray[np.newaxis, :, np.newaxis], zarray[np.newaxis, np.newaxis, :])
#     else:
#         zhalf = np.abs(zarray[1] - zarray[0]) / 2.0
#         zlen = zarray.size
#         if _endpoint:
#             za = (zarray[:, np.newaxis] + np.linspace(-zhalf, zhalf, zaverage)[np.newaxis, :]).flatten()
#         else:
#             za = np.linspace(zarray[0] - zhalf, zarray[-1]+ zhalf, zlen * zaverage, endpoint=False) + (zhalf / zaverage)

#         lsections = np.array_split(np.arange(lmax+1), lmax / 50)

#         cla = np.zeros((lmax+1, zlen, zlen), dtype=np.float64)

#         for lsec in lsections:
#             clt = aps(lsec[:, np.newaxis, np.newaxis],
#                       za[np.newaxis, :, np.newaxis], za[np.newaxis, np.newaxis, :])

#             cla[lsec] = clt.reshape((-1, zlen, zaverage, zlen, zaverage)).sum(axis=4).sum(axis=2) / zaverage**2

#         return cla



# def clarray_romb(aps, lmax, zarray, zromb=None):

#     if zromb == None:
#         zromb = 2

#     if zromb == 0:
#         return aps(np.arange(lmax+1)[:, np.newaxis, np.newaxis],
#                    zarray[np.newaxis, :, np.newaxis], zarray[np.newaxis, np.newaxis, :])

#     else:
#         zhalf = np.abs(zarray[1] - zarray[0]) / 2.0
#         zlen = zarray.size
#         zint = 2**zromb + 1
#         zspace = 2.0*zhalf / 2**zromb

#         za = (zarray[:, np.newaxis] + np.linspace(-zhalf, zhalf, zint)[np.newaxis, :]).flatten()

#         lsections = np.array_split(np.arange(lmax+1), lmax / 50)

#         cla = np.zeros((lmax+1, zlen, zlen), dtype=np.float64)

#         for lsec in lsections:
#             clt = aps(lsec[:, np.newaxis, np.newaxis],
#                       za[np.newaxis, :, np.newaxis], za[np.newaxis, np.newaxis, :])

#             clt = clt.reshape(-1, zlen, zint, zlen, zint)

#             clt = scipy.integrate.romb(clt, dx=zspace, axis=4)
#             clt = scipy.integrate.romb(clt, dx=zspace, axis=2)

#             cla[lsec] = clt

#         return cla


class PointSources(foregroundsck.PointSources):
    """Scale up point source amplitude to a higher S_{cut} = 0.1 Jy"""
    A = 3.55e-5
    nu_0 = 408.0
    l_0 = 100.0


def foreground_model(lmax, frequencies, npol, polfrac=0.5):

    fsyn = galaxy.FullSkySynchrotron()
    fps = PointSources()
    
    nfreq = frequencies.size

    cv_fg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_fg[0, 0] = skysim.clarray(fsyn.angular_powerspectrum, lmax, frequencies)

    if npol >= 3:
        fpol = galaxy.FullSkyPolarisedSynchrotron()
        cv_fg[1, 1] = skysim.clarray(fpol.angular_powerspectrum, lmax, frequencies)
        cv_fg[2, 2] = skysim.clarray(fpol.angular_powerspectrum, lmax, frequencies)

    cv_fg[0, 0] += skysim.clarray(fps.angular_powerspectrum, lmax, frequencies)
    return cv_fg




def im21cm_model(lmax, frequencies, npol, cr = None):

    nfreq = frequencies.size

    if not cr:
        global _cr
        if not _cr:
            if not _reionisation:    
                _cr = corr21cm.Corr21cm()
            else:
                _cr = corr21cm.EoR21cm()
        cr = _cr    

    cr._freq_window = np.abs(cr.cosmology.comoving_distance(frequencies[0]) - cr.cosmology.comoving_distance(frequencies[1]))

    cv_sg = np.zeros((npol, npol, lmax+1, nfreq, nfreq))

    cv_sg[0, 0] = skysim.clarray(cr.angular_powerspectrum, lmax, frequencies)

    return cv_sg

