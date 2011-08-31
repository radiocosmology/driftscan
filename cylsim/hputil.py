"""Convenience functions for dealing with Healpix maps.

Uses the healpy module.
"""

import healpy

import numpy as np

def ang_positions(nside):
    """Fetch the angular position of each pixel in a map.

    Parameters
    ----------
    nside : scalar
        The size of the map (nside is a definition specific to Healpix).

    Returns
    -------
    angpos : np.ndarray
        The angular position (in spherical polars), of each pixel in a
        Healpix map. Packed at [ [theta1, phi1], [theta2, phi2], ...]
    """
    npix = healpy.nside2npix(int(nside))
    
    angpos = np.empty([npix, 2], dtype = np.float64)

    angpos[:, 0], angpos[:, 1] = healpy.pix2ang(nside, np.arange(npix))

    return angpos


def sphtrans_real(hpmap, lmax = None):
    """Spherical Harmonic transform of a real map.

    Parameters
    ----------
    hpmap : np.ndarray
        A Healpix map.
    lmax : scalar, optional
        The maximum l to calculate. If `None` (default), calculate up
        to 3*nside - 1.

    Returns
    -------
    alm : np.ndarray
        A 2d array of alms, packed as alm[l,m].

    Notes
    -----
    This only includes m > 0. As this is the transform of a real field:

    .. math:: a_{l -m} = (-1)^m a_{lm}^*
    """
    if lmax == None:
        lmax = 3*healpy.npix2nside(hpmap.size) - 1

    alm = np.zeros([lmax+1, lmax+1], dtype=np.complex128)

    tlm = healpy.map2alm(np.ascontiguousarray(hpmap), lmax=lmax)

    alm[np.triu_indices(lmax+1)] = tlm

    return alm.T


def _make_full_alm(alm_half, centered = False):
    ## Construct an array of a_{lm} including both positive and
    ## negative m, from one including only positive m.
    lmax, mmax = alm_half.shape

    alm = np.zeros([lmax, 2*mmax - 1], dtype=alm_half.dtype)

    alm_neg = alm_half[:, :0:-1].conj()
    mfactor = (-1)**np.arange(mmax)[:0:-1][np.newaxis, :]
    alm_neg = mfactor *alm_neg

    if not centered:
        alm[:lmax, :mmax] = alm_half
        alm[:lmax, mmax:] = alm_neg
    else:
        alm[:lmax, (mmax-1):] = alm_half
        alm[:lmax, :(mmax-1)] = alm_neg

    return alm
    
def sphtrans_complex(hpmap, lmax = None, centered = False):
    """Spherical harmonic transform of a complex function.

    Parameters
    ----------
    hpmap : np.ndarray
        A complex Healpix map.
    lmax : scalar, optional
        The maximum l to calculate. If `None` (default), calculate up to 3*nside
        - 1.
    centered : boolean, optional
        If False (default) similar to an FFT, alm[l,:lmax+1] contains m >= 0,
        and the latter half alm[l,lmax+1:], contains m < 0. If True the first
        half opf alm[l,:] contains m < 0, and the second half m > 0. m = 0 is
        the central column.
        
    Returns
    -------
    alm : np.ndarray
        A 2d array of alms, packed as alm[l,m].
    """
    if lmax == None:
        lmax = 3*healpy.npix2nside(hpmap.size) - 1

    rlm = _make_full_alm(sphtrans_real(hpmap.real, lmax = lmax),
                         centered = centered)
    ilm = _make_full_alm(sphtrans_real(hpmap.imag, lmax = lmax),
                         centered = centered)

    alm = rlm + 1.0J * ilm

    return alm


def sphtrans_real_pol(hpmaps, lmax = None, lside=None):
    """Spherical Harmonic transform of polarisation functions on the sky.

    Accepts real T, Q and U like maps, and returns :math:`a^T_{lm}`
    :math:`a^E_{lm}` and :math:`a^B_{lm}`.

    Parameters
    ----------
    hpmaps : list of np.ndarray
        A list of Healpix maps, assumed to be T, Q, and U.
    lmax : scalar, optional
        The maximum l to calculate. If `None` (default), calculate up to 3*nside
        - 1.

    Returns
    -------
    alm_T, alm_E, alm_B : np.ndarray
        A 2d array of alms, packed as alm[l,m].

    Notes
    -----
    This only includes m > 0. As these are the transforms of a real field:

    .. math:: a_{l -m} = (-1)^m a_{lm}^*
    """
    if lmax == None:
        lmax = 3*healpy.npix2nside(hpmaps[0].size) - 1

    if lside == None or lside < lmax:
        lside = lmax

    alms = [np.zeros([lside+1, lside+1], dtype=np.complex128) for i in range(3)]

    tlms = healpy.map2alm([np.ascontiguousarray(hpmap) for hpmap in hpmaps],
                          lmax=lmax)

    for i in range(3):
        alms[i][np.triu_indices(lmax+1)] = tlms[i]

    return [alm.T for alm in alms]




def sphtrans_complex_pol(hpmaps, lmax = None, centered = False, lside=None):
    """Spherical harmonic transform of the polarisation on the sky (can be
    complex).

    Accepts complex T, Q and U like maps, and returns :math:`a^T_{lm}`
    :math:`a^E_{lm}` and :math:`a^B_{lm}`.
    
    Parameters
    ----------
    hpmaps : np.ndarray
         A list of complex Healpix maps, assumed to be T, Q, and U.
    lmax : scalar, optional
        The maximum l to calculate. If `None` (default), calculate up to 3*nside
        - 1.
    centered : boolean, optional
        If False (default) similar to an FFT, alm[l,:lmax+1] contains m >= 0,
        and the latter half alm[l,lmax+1:], contains m < 0. If True the first
        half opf alm[l,:] contains m < 0, and the second half m > 0. m = 0 is
        the central column.
        
    Returns
    -------
    alm_T, alm_E, alm_B : np.ndarray
        A 2d array of alms, packed as alm[l,m].
    """
    if lmax == None:
        lmax = 3*healpy.npix2nside(hpmaps[0].size) - 1

    rlms = [_make_full_alm(alm, centered = centered) for alm in
            sphtrans_real_pol([hpmap.real for hpmap in hpmaps], lmax = lmax, lside=lside)]
    ilms = [_make_full_alm(alm, centered = centered) for alm in
            sphtrans_real_pol([hpmap.imag for hpmap in hpmaps], lmax = lmax, lside=lside)]

    alms = [rlm + 1.0J * ilm for rlm, ilm in zip(rlms, ilms)]

    return alms
    
