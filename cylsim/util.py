import healpy

import hputil

import blockla

from simulations.foregroundmap import matrix_root_manynull

from simulations import pointsource

import scipy.linalg as la


import numpy as np

def intpattern(n):
    """Pattern that prints out a number upto `n` (integer - always shows sign)."""
    return ("%+0" + repr(int(np.ceil(np.log10(n + 1))) + 1) + "d")


def natpattern(n):
    """Pattern that prints out a number upto `n` (natural number - no sign)."""
    return ("%0" + repr(int(np.ceil(np.log10(n + 1)))) + "d")




def mkfullsky(corr, nside, alms = False):
    """Construct a set of correlated Healpix maps.
    
    Make a set of full sky gaussian random fields, given the correlation
    structure. Useful for constructing a set of different redshift slices.

    Parameters
    ----------
    corr : np.ndarray (lmax+1, numz, numz)
        The correlation matrix :math:`C_l(z, z')`.
    nside : integer
        The resolution of the Healpix maps.
    alms : boolean, optional
        If True return the alms instead of the sky maps.

    Returns
    -------
    hpmaps : np.ndarray (numz, npix)
        The Healpix maps. hpmaps[i] is the i'th map.
    """

    numz = corr.shape[1]
    maxl = corr.shape[0]-1

    if corr.shape[2] != numz:
        raise Exception("Correlation matrix is incorrect shape.")

    trans = np.zeros_like(corr)

    for i in range(maxl+1):
        trans[i] = matrix_root_manynull(corr[i], truncate=False)

    
    la, ma = healpy.Alm.getlm(maxl)

    matshape = la.shape + (numz,)

    # Construct complex gaussian random variables of unit variance
    gaussvars = (np.random.standard_normal(matshape)
                 + 1.0J * np.random.standard_normal(matshape)) / 2.0**0.5

    # Transform variables to have correct correlation structure
    for i, l in enumerate(la):
        gaussvars[i] = np.dot(trans[l], gaussvars[i])

    if alms:
        alm_freq = np.zeros((numz, maxl+1, 2*maxl+1), dtype=np.complex128)
        for i in range(numz):
            alm_freq[i] = hputil.unpack_alm(gaussvars[:, i], maxl, fullm=True)
        
        return alm_freq

    hpmaps = np.empty((numz, healpy.nside2npix(nside)))

    # Perform the spherical harmonic transform for each z
    for i in range(numz):
        hpmaps[i] = healpy.alm2map(gaussvars[:,i].copy(), nside)
        
    return hpmaps


def mkconstrained(corr, constraints, nside):
    """Construct a set of correlated Healpix maps.
    
    Make a set of full sky gaussian random fields, given the correlation
    structure. Useful for constructing a set of different redshift slices.

    Parameters
    ----------
    corr : np.ndarray (lmax+1, numz, numz)
        The correlation matrix :math:`C_l(z, z')`.
    nside : integer
        The resolution of the Healpix maps.

    Returns
    -------
    hpmaps : np.ndarray (numz, npix)
        The Healpix maps. hpmaps[i] is the i'th map.
    """

    numz = corr.shape[1]
    maxl = corr.shape[0]-1
    
    larr, marr = healpy.Alm.getlm(maxl)

    matshape = larr.shape + (numz,)

    nmodes = len(constraints)

    f_ind = zip(*constraints)[0]

    if corr.shape[2] != numz:
        raise Exception("Correlation matrix is incorrect shape.")

    trans = np.zeros((corr.shape[0], nmodes, corr.shape[2]))
    tmat = np.zeros((corr.shape[0], nmodes, nmodes))

    cmap = np.zeros(larr.shape + (nmodes, ), dtype=np.complex128)

    cv = np.zeros((numz,) + larr.shape, dtype=np.complex128)

    for i in range(maxl+1):
        trans[i] = la.eigh(corr[i])[1][:, -nmodes:].T
        tmat[i] = trans[i][:, f_ind]

    for i, cons in enumerate(constraints):
        cmap[:, i] = healpy.map2alm(cons[1], lmax=maxl)

    for i, l in enumerate(larr):
        if l == 0:
            cv[:, i] = 0.0
        else:
            cv[:, i] = np.dot(trans[l].T, la.solve(tmat[l].T, cmap[i]))

    hpmaps = np.empty((numz, healpy.nside2npix(nside)))

    for i in range(numz):
        hpmaps[i] = healpy.alm2map(cv[i], nside)

    return hpmaps




def fullskyps(nside, freq):

    npix = healpy.nside2npix(nside)

    sky = np.zeros((npix, freq.size))

    ps = pointsource.DiMatteo()
    
    fluxes = ps.generate_population(4*np.pi * (180 / np.pi)**2)

    print ps.size
    print ps.size * freq.size * 8.0 / (2**30.0)

    #sr = ps.spectral_realisation(fluxes[:,np.newaxis], freq[np.newaxis,:])
    #    
    #for i in xrange(npix):
    #    sky[npix,:] += sr[i,:]

    




def proj_mblock(hpmap, vec_mblock):

    lmax = vec_mblock.shape[-1] - 1
    alm = hputil.sphtrans_complex(hpmap, lmax=lmax).T.copy()

    almproj = blockla.multiply_dm_v(vec_mblock, blockla.multiply_dm_v(vec_mblock, alm), conj=True)

    hpproj = healpy.alm2map(hputil.pack_alm(almproj.T), healpy.npix2nside(hpmap.size))

    return hpproj
