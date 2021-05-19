cimport cython
from cython.parallel cimport prange, parallel

import numpy as np
cimport numpy

from libc.stdlib cimport abort, malloc, free
from libc.math cimport sin, cos, tan, exp, hypot, M_PI, M_PI_2, M_LN2

from cora.util.coord import thetaphi_plane_cart

ctypedef double complex complex128

cdef extern from "complex.h" nogil:
    complex128 conj(complex128)


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


@cython.boundscheck(False)
@cython.wraparound(False)
def _construct_pol_real(
    double[:,  ::1] beami,
    double[:, ::1] beamj,
    complex128[::1] fringe,
    double[::1] horizon
):
    # Fast function for constructing the beam transfer matrices if the beams are *real*

    cdef Py_ssize_t i, n
    cdef double om_i = 0
    cdef double om_j = 0
    cdef double t, prefactor
    cdef complex128 tc

    cdef complex128[:, ::1] bt_view

    n = beami.shape[0]

    assert beamj.shape[0] == n
    assert fringe.shape[0] == n
    assert horizon.shape[0] == n
    assert beami.shape[1] == 2
    assert beamj.shape[1] == 2

    bt = np.empty((4, n), dtype=np.complex128)

    bt_view = bt

    for i in prange(n, nogil=True):
        # Accumulate power in beam_i
        t  = beami[i, 0] * beami[i, 0]
        t += beami[i, 1] * beami[i, 1]
        om_i += horizon[i] * t

        # Accumulate power in beam_j
        t  = beamj[i, 0] * beamj[i, 0]
        t += beamj[i, 1] * beamj[i, 1]
        om_j += horizon[i] * t

    # Scale by pixel areas to get beam area integrals
    om_i *= 4 * M_PI / n
    om_j *= 4 * M_PI / n

    prefactor = 1.0 / (om_i * om_j) ** 0.5

    for i in prange(n, nogil=True):

        tc = prefactor * fringe[i] * horizon[i]

        # Stokes I response
        bt_view[0, i] = tc * (
            beami[i, 0] * beamj[i, 0] + beami[i, 1] * beamj[i, 1]
        )

        # Stokes Q response
        bt_view[1, i] = tc * (
            beami[i, 0] * beamj[i, 0] - beami[i, 1] * beamj[i, 1]
        )

        # Stokes U response
        bt_view[2, i] = tc * (
            beami[i, 0] * beamj[i, 1] + beami[i, 1] * beamj[i, 0]
        )
        # Stokes V response
        bt_view[3, i] = 1j * tc * (
            beami[i, 0] * beamj[i, 1] - beami[i, 1] * beamj[i, 0]
        )

    return bt


@cython.boundscheck(False)
@cython.wraparound(False)
def _construct_pol_complex(
    complex128[:,  ::1] beami,
    complex128[:, ::1] beamj,
    complex128[::1] fringe,
    double[::1] horizon
):
    # Fast function for constructing the beam transfer matrices if the beams are
    # *complex*

    cdef Py_ssize_t i, n
    cdef double om_i = 0
    cdef double om_j = 0
    cdef double t, prefactor
    cdef complex128 tc

    cdef complex128[:, ::1] bt_view

    n = beami.shape[0]

    assert beamj.shape[0] == n
    assert fringe.shape[0] == n
    assert horizon.shape[0] == n
    assert beami.shape[1] == 2
    assert beamj.shape[1] == 2

    bt = np.empty((4, n), dtype=np.complex128)

    bt_view = bt

    for i in prange(n, nogil=True):
        # Accumulate power in beam_i
        t  = beami[i, 0].real * beami[i, 0].real
        t += beami[i, 0].imag * beami[i, 0].imag
        t += beami[i, 1].real * beami[i, 1].real
        t += beami[i, 1].imag * beami[i, 1].imag
        om_i += horizon[i] * t

        # Accumulate power in beam_j
        t  = beamj[i, 0].real * beamj[i, 0].real
        t += beamj[i, 0].imag * beamj[i, 0].imag
        t += beamj[i, 1].real * beamj[i, 1].real
        t += beamj[i, 1].imag * beamj[i, 1].imag
        om_j += horizon[i] * t

    # Scale by pixel areas to get beam area integrals
    om_i *= 4 * M_PI / n
    om_j *= 4 * M_PI / n

    prefactor = 1.0 / (om_i * om_j) ** 0.5

    for i in prange(n, nogil=True):

        tc = prefactor * fringe[i] * horizon[i]

        # Stokes I response
        bt_view[0, i] = tc * (
            beami[i, 0] * conj(beamj[i, 0]) + beami[i, 1] * conj(beamj[i, 1])
        )

        # Stokes Q response
        bt_view[1, i] = tc * (
            beami[i, 0] * conj(beamj[i, 0]) - beami[i, 1] * conj(beamj[i, 1])
        )

        # Stokes U response
        bt_view[2, i] = tc * (
            beami[i, 0] * conj(beamj[i, 1]) + beami[i, 1] * conj(beamj[i, 0])
        )
        # Stokes V response
        bt_view[3, i] = 1j * tc * (
            beami[i, 0] * conj(beamj[i, 1]) - beami[i, 1] * conj(beamj[i, 0])
        )

    return bt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def beam_exptan(double[::1] sintheta, double fwhm):
    """ExpTan beam.

    Note that this model was used in arXiv:1401.2095, however Eq 35 in that work
    contains a typo and is missing a factor of two in the tan factor on the denominator.
    The expression here is correct and is what was actually used in that paper.

    Parameters
    ----------
    sintheta : array_like
        Array of sin(angles) to return beam at.
    fwhm : scalar
        Beam width at half power (note that the beam returned is amplitude) as an
        angle (not sin(angle)).

    Returns
    -------
    beam : array_like
        The amplitude beam at each requested angle.
    """
    cdef double tan2, alpha
    cdef double[::1] et_view
    cdef Py_ssize_t i, n

    et = np.empty_like(sintheta)
    et_view = et

    alpha = M_LN2 / (2 * tan(fwhm / 2.0) ** 2)

    n = sintheta.shape[0]
    for i in prange(n, nogil=True):
        tan2 = sintheta[i]**2 / (1 - sintheta[i]**2 + 1e-100)
        et_view[i] = exp(-alpha * tan2)

    return et
