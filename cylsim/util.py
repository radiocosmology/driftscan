import healpy

import hputil

import blockla

from simulations.foregroundmap import matrix_root_manynull

def mkfullsky(corr, nside):
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

    if corr.shape[2] != numz:
        raise Exception("Correlation matrix is incorrect shape.")

    trans = np.zeros_like(corr.shape)

    for i in range(maxl+1):
        trans[i] = matrix_root_manynull(corr[i], truncate=False)

    
    la, ma = healpy.Alm.getlm(lmax)

    matshape = la.shape + (numz,)

    # Construct complex gaussian random variables of unit variance
    gaussvars = (np.random.standard_normal(matshape)
                 + 1.0J * np.random.standard_normal(matshape)) / 2.0**0.5

    # Transform variables to have correct correlation structure
    for i, l in enumerate(la):
        gaussvars[i] = np.dot(trans[l], gaussvars[i])

    hpmaps = np.empty((numz, healpy.nside2npix(nside)))

    # Perform the spherical harmonic transform for each z
    for i in range(numz):
        hpmaps[i] = healpy.alm2map(gaussvars[:,i].copy(), nside)
        
    return hpmaps




def proj_mblock(hpmap, vec_mblock):

    lmax = vec_mblock.shape[-1] - 1
    alm = hputil.sphtrans_complex(hpmap, lmax=lmax).T.copy()

    almproj = blockla.multiply_dm_v(vec_mblock, blockla.multiply_dm_v(vec_mblock, alm), conj=True)

    hpproj = healpy.alm2map(hputil.pack_alm(almproj.T), healpy.npix2nside(hpmap.size))

    return hpproj
