cimport cython
from cython.parallel cimport prange, parallel

import numpy as np
cimport numpy

from libc.stdlib cimport abort, malloc, free
from libc.math cimport sin, cos, M_PI_2
from libc.math cimport sin, cos, hypot, M_PI, M_PI_2

from cora.util.coord import thetaphi_plane_cart

ctypedef double complex complex128


def fringe(sph_coords, zenith, baseline):
    r"""The fringe for a specified baseline at each angular position.

    Parameters
    ----------
    sph_coords : np.ndarray
        Angular positions (in spherical polar co-ordinates).
    zenith : np.ndarray
        The zenith vector in spherical polar-coordinates.
    baseline : array_like
        The baseline to calculate, given in (u,v) coordinates.

    Returns
    -------
    fringe : np.ndarray
        The fringe.

    Notes
    -----
    For each position :math:`\hat{n}` in `sph_arr` calculate:

    .. math:: \exp{\left(2\pi i * \hat{n}\cdot u_{12}\right)}

    """
    cdef Py_ssize_t i, n
    cdef double[::1] uhat, vhat, uv
    cdef complex128[::1] fringe_view
    cdef double[:, ::1] sph_view
    cdef double * cart_temp
    cdef double du, phase

    # Get the basis vectors in UV coordinates
    t_hat, p_hat = thetaphi_plane_cart(zenith)
    uhat = p_hat
    vhat = -t_hat
    uv = baseline[0] * uhat + baseline[1] * vhat

    fringe = np.empty(sph_coords.shape[:-1], dtype=np.complex128)
    fringe_view = fringe.ravel()

    sph_view = sph_coords.reshape(-1, 2)

    n = sph_view.shape[0]

    with cython.boundscheck(False), cython.wraparound(False), nogil, parallel():
    #with cython.boundscheck(False), cython.wraparound(False), nogil:

        cart_temp = <double *> malloc(sizeof(double) * 3)
        if cart_temp is NULL:
            abort()

        # share the work using the thread-local buffer(s)
        #for i in range(n):
        for i in prange(n):
            _s2c(sph_view[i, 0], sph_view[i, 1], cart_temp)

            du = cart_temp[0] * uv[0] + cart_temp[1] * uv[1] + cart_temp[2] * uv[2]

            phase = 2 * M_PI * du

            fringe_view[i] = cos(phase) + 1j * sin(phase)

        free(cart_temp)

    return fringe


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _s2c(double theta, double phi, double cart[3]) nogil:
    cdef double sintheta = sin(theta)
    cart[0] = sintheta * cos(phi)
    cart[1] = sintheta * sin(phi)
    cart[2] = cos(theta)