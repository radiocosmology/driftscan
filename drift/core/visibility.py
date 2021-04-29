"""Routines for calculating visibilities on the full sky."""

import numpy as np

from cora.util import coord
from ..util._fast_tools import fringe


def uv_plane_cart(zenith):
    r"""Fetch unit vectors in the UV plane.

    Parameters
    ----------
    zenith : np.ndarray
        The zenith vector in spherical polar-coordinates.

    Returns
    -------
    uhat, vhat : np.ndarray
        Unit vectors in the UV plane. `uhat` points East, and `vhat`
        points North.
    """
    t_hat, phat = coord.thetaphi_plane_cart(zenith)
    return phat, -t_hat


def horizon(sph_arr, zenith):
    r"""The horizon function at particular location.

    Parameters
    ----------
    sph_arr : np.ndarray
        Angular positions (in spherical polar co-ordinates).
    zenith : np.ndarray
        The zenith vector in spherical polar-coordinates.

    Returns
    -------
    horizon : np.ndarray
        The horizon function (including an angular projection term at
        each position).
    """

    proj = coord.sph_dot(sph_arr, zenith)

    return np.signbit(-proj)


def cylinder_beam(sph_arr, zenith, cylwidth):
    r"""The beam function for a cylinder aligned N-S.

    Beam will be a thin strip in the N-S direction (controlled by the
    cylinder width in the E-W direction).

    Parameters
    ----------
    sph_arr : np.ndarray
        Angular positions (in spherical polar co-ordinates).
    zenith : np.ndarray
        The zenith vector in spherical polar-coordinates.
    cylwidth : scalar
        The effective cylinder width.

    Returns
    -------
    beam : np.ndarray
        The beam function at each angular position.
    """
    # Construct uhat.
    uhatc, vhatc = uv_plane_cart(zenith)

    ## Note sinc function is normalised hence lack of pi
    return np.sinc(np.inner(coord.sph_to_cart(sph_arr), cylwidth * uhatc))


def pol_IQU(sph_arr, zenith, feed1, feed2):
    r"""The polarisation tensors at each point, projected onto two
    feeds.

    Parameters
    ----------
    sph_arr : np.ndarray
        Angular positions (in spherical polar co-ordinates).
    zenith : np.ndarray
        The zenith vector in spherical polar-coordinates.
    feed1, feed2 : np.ndarray
        Unit vectors for the two feeds. Should be given in (u,v)
        coordinates.

    Returns
    -------
    pI, pQ, pU : np.ndarray
        The projected polarisations at each angular position.

    Notes
    -----
    For each position :math:`\hat{n}` in `sph_arr` calculate:

    .. math:: f_1^a f_2^b \mathcal{P}^X_{ab}

    where X is one of I, Q or U.

    The co-ordinate system defining the polarisation at each point is
    :math:`(\hat{\theta}, \hat{\phi})`.
    """

    # Get theta, phi plane at each point.
    t_hat, p_hat = coord.thetaphi_plane_cart(sph_arr)

    # Get feed vectors in 3D Cartesian co-ordinates
    uhat, vhat = uv_plane_cart(zenith)
    f1c = feed1[0] * uhat + feed1[1] * vhat
    f2c = feed2[0] * uhat + feed2[1] * vhat

    # Project feed vectors into theta-phi plane.
    f1_t = np.inner(t_hat, f1c)
    f1_p = np.inner(p_hat, f1c)
    f2_t = np.inner(t_hat, f2c)
    f2_p = np.inner(p_hat, f2c)

    pI = 0.5 * (f1_t * f2_t + f1_p * f2_p)  # I
    pQ = 0.5 * (f1_t * f2_t - f1_p * f2_p)  # Q
    pU = 0.5 * (f1_t * f2_p + f1_p * f2_t)  # U

    return pI, pQ, pU
