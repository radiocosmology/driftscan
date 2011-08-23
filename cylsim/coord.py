
import numpy as np

def sph_to_cart(sph_arr):
    """Convert a vector in Spherical Polar coordinates to Cartesians.

    Parameters
    ----------
    sph_arr : np.ndarry
        A vector (or array of) in spherical polar co-ordinates. Values
        should be packed as [r, theta, phi] along the last
        axis. Alternatively they can be packed as [theta, phi] in
        which case r is assumed to be one.

    Returns
    -------
    cart_arr : np.ndarry
        Array of equivalent vectors in cartesian coordinartes.
    """

    # Create an array of the correct size for the output.
    shape = list(sph_arr.shape)
    shape[-1] = 3
    cart_arr = np.empty(shape)

    # Useful quantities
    if sph_arr.shape[-1] == 3:
        r = sph_arr[...,0]
    else:
        r = 1.0

    st = np.sin(sph_arr[...,-2])

    cart_arr[...,0] = r * st * np.cos(sph_arr[...,-1])  # x-axis
    cart_arr[...,1] = r * st * np.sin(sph_arr[...,-1])  # y-axis
    cart_arr[...,2] = r * np.cos(sph_arr[...,-2])       # z-axis

    return cart_arr


def sph_dot(arr1, arr2):
    """Take the scalar product in spherical polars.
    
    Parameters
    ----------
    arr, arr2 : np.ndarray
        Two arrays of vectors in spherical polars [theta, phi], (or
        alternatively as [theta, phi]). Should be broadcastable
        against each other.

    Returns
    -------
    dot : np.ndarray
        An array of the dotted vectors.

    """
    return np.inner(sph_to_cart(arr1), sph_to_cart(arr2))


def thetaphi_plane(sph_arr):
    """For each position, return the theta, phi unit vectors (in
    spherical polars).

    Parameters
    ----------
    sph_arr : np.ndarray
        Angular positions (in spherical polar co-ordinates).

    Returns
    -------
    thetahat, phihat : np.ndarray
        Unit vectors in the theta and phi directions (still in
        spherical polars).
    """
    thetahat = sph_arr.copy()
    thetahat[...,-2] += np.pi / 2.0

    phihat = sph_arr.copy()
    phihat[...,-2] = np.pi / 2.0
    phihat[...,-1] += np.pi / 2.0

    if sph_arr.shape[-1] == 3:
        thetahat[...,0] = 1.0
        phihat[...,0] = 1.0

    return thetahat, phihat


def thetaphi_plane_cart(sph_arr):
    """For each position, return the theta, phi unit vectors (in
    cartesians).

    Parameters
    ----------
    sph_arr : np.ndarray
        Angular positions (in spherical polar co-ordinates).

    Returns
    -------
    thetahat, phihat : np.ndarray
        Unit vectors in the theta and phi directions (in
        cartesian coordinates).
    """
    that, phat = thetaphi_plane(sph_arr)

    return sph_to_cart(that), sph_to_cart(phat)
    
    
    
