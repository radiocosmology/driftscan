"""Helpers for parameterised primary beams for pointed dish arrays.

The classes provide parameterised beams made use of by the 
:py:class:`..core.CustomDishArray` mixin.

These can be specified in the `beam_spec` section of the configuration file.

For example:

.. code-block:: yaml

    # In a drift-makeproducts configuration file:

    telescope:
      
      # Must be a TransitTelescope (sub)-class with CustomDishArray mixin.
      type: PolarisedDishArray 

      # Any TransitTelescope / CustomDishArray Parameters
      ...

      beam_spec:
        # Gaussian beam with FWHM = lambda/(6 m)
        type: gaussian
        diameter: 6 # effective dish diameter in metres
      
Other examples:

.. code-block:: yaml

    beam_spec:
        # Airy beam corresponding to a co-polar pattern of a uniformly 
        # illuminated aperture of diameter 6m and a cross-polar term 
        # -60 dB suppressed (in voltage) with the same pattern.
        type: airy
        diameter: 6 # effective dish diameter in metres
        crosspol_type: scaled
        crosspol_scale_dB: -60

.. code-block:: yaml

    beam_spec:
        # Read an arbitrary HEALPix file of the E_theta, E_phi
        # beam pattern stored in a draco.core.containers.HEALPixBeam
        # container.
        type: healpix
        filename: /path/to/beam_file.h5
        # Use the nearest frequency index of the beam file to that requested by
        # the telescope object.
        freq_index_type: nearest

Currently supported `beam_spec` types are:

- `gaussian` provided by :py:class:`GaussianBeam`
- `airy` provided by :py:class:`AiryBeam`
- `healpix` provided by :py:class:`HEALPixBeam` 

See their class and base class definitions for more parameter options. 
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from typing import Literal, Optional, Tuple, Union
import abc
import logging

from scipy.special import j1 as bessel_j1
import numpy as np
import healpy as hp

from caput import config
from caput.astro.coordinates import spherical as coord

from drift.core.telescope import TransitTelescope
from drift.telescope.cylbeam import polpattern

# Types
POLTYPE = Literal["X", "Y"]
FloatArrayLike = Union[np.ndarray, float]

# Constants
FWHM2SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def rot_mats_thetaphi(theta: float, phi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation matrices for rotating cartesian vectors by angles corresponding
    to the HEALPix :math:`\\theta, \phi` directions. Convention is that a positive 
    :math:`\\theta` rotation rotates the :math:`\hat{z}` direction south along the 
    :math:`\phi=0` meridian and a positive :math:`\phi` rotation rotates the
    :math:`\hat{x}` direction eastwards.

    Parameters
    ----------
    theta
        Theta in radians.
    phi
        Phi in radians.

    Returns
    -------
        Tuple of (3, 3) theta_rotation_matrix and phi_rotation_matrix
    """

    rot_mat_theta = np.array(
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]
    )

    rot_mat_phi = np.array(
        [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )

    return rot_mat_theta, rot_mat_phi


def pointing_rot_mat(zenith: np.ndarray, altaz_pointing: np.ndarray) -> np.ndarray:
    """Rotation matrix that rotates a cartesian vector from HEALPix theta, phi coordinates,
    to zenith-local theta, phi coordinates and then finally to theta, phi coordinates
    relative to an altitude-azimuth pointing from that zenith.

    Parameters
    ----------
    zenith
        (2,) array of theta, phi zenith coordinates in radians.
    altaz_pointing
        (2,) array of altitude, azimuth coordinates in radians.

    Returns
    -------
        (3, 3) rotation matrix.
    """
    rth, rphi = rot_mats_thetaphi(zenith[0], zenith[1])
    rzenith = rth @ rphi
    hp_alt = np.pi / 2 - altaz_pointing[0]
    hp_az = np.pi - altaz_pointing[1]
    ralt, raz = rot_mats_thetaphi(hp_alt, hp_az)
    rpoint = ralt @ raz

    return rpoint @ rzenith


def rotate_thetaphi_beam(
    beam: np.ndarray, rot_theta: np.ndarray, angpos: np.ndarray
) -> np.ndarray:
    """Rotate a HEALPix vector beam pattern packed in Etheta, Ephi
    by an angle in the theta direction.

    Parameters
    ----------
    beam
        (N, 2) vector beam pattern in Etheta, Ephi, assumed packed in RING 
        ordering. May be complex.
    rot_theta
        Angle to rotate by in the theta direction in radians. Positive theta
        rotates South.
    angpos
        (N, 2) array of healpix theta, phi coordinates.
    Returns
    -------
        (N, 2) array of the rotated beam pattern in Etheta, Ephi.
    """

    # Rotate amplitude of Eth and Eph beam terms
    hp_rot = hp.Rotator(rot=(0, rot_theta, 0), deg=False)

    # If a null rotation, do nothing and return the original beam
    if not hp_rot.do_rot(0):
        return beam
        
    thph_amp_rot = np.empty_like(beam)

    # Healpy spams logs when on log level INFO (as in during a makeproducts run)
    # This sets the healpix log level to warn for the rotation
    global_log_level = logging.getLogger().level
    hp_logger = logging.getLogger("healpy")
    hp_logger.setLevel(level=logging.WARN)

    thph_amp_rot[:, 0] = hp_rot.rotate_map_alms(beam[:, 0].real)
    thph_amp_rot[:, 1] = hp_rot.rotate_map_alms(beam[:, 1].real)

    if np.iscomplexobj(beam):
        thph_amp_rot[:, 0] += 1j * hp_rot.rotate_map_alms(beam[:, 0].imag)
        thph_amp_rot[:, 1] += 1j * hp_rot.rotate_map_alms(beam[:, 1].imag)

    hp_logger.setLevel(level=global_log_level)

    # Need to correct for change of polarisation coordinate system
    # Work out projection of theta_hat and phi_hat onto theta_hat and
    # phi_hat from inverse rotation
    cart_sky = coord.sph_to_cart(angpos)
    inv_rot_cart_sky = cart_sky @ hp_rot.mat  # Equivalent to (hp_rot.mat.T @ cart_sky)
    inv_rot_angpos = coord.cart_to_sph(inv_rot_cart_sky)[..., 1:]

    thetaphi_hat_sky = np.stack(coord.thetaphi_plane_cart(angpos), axis=1)
    thetaphi_hat_beam = np.stack(coord.thetaphi_plane_cart(inv_rot_angpos), axis=-1)
    beam_to_sky = thetaphi_hat_sky @ thetaphi_hat_beam

    return (beam_to_sky @ thph_amp_rot[..., None]).squeeze()


def pointing_offset_thetaphi_coords(
    angpos: np.ndarray, zenith: np.ndarray, altaz_pointing: np.ndarray
) -> np.ndarray:
    """Offset radial (:math:`\\theta`) and azimuthal (:math:`\phi`) coordinates 
    for an input pointing relative to an observer with an input zenith.

    Useful for interpolating beams as functions of pointing offset coordinates
    onto sky coordinates for a given pointing. Phi convention is such that
    it increases anti-clockwise from the local vertical for pointing's with 
    an altitude <= pi/2. For altitude > pi/2 this is extended in an "over-the-top"
    sense.

    Parameters
    ----------
    angpos
        (N, 2) array of healpix theta, phi coordinates.
    zenith
        (2,) array of theta, phi zenith coordinates in radians.
    altaz_pointing
        (2,) array of altitude, azimuth coordinates in radians.

    Returns
    -------
        (N, 2) array of pointing offset coordinates 
    """
    cart_sky = coord.sph_to_cart(angpos)

    full_rot = pointing_rot_mat(zenith, altaz_pointing)

    off_cart = cart_sky @ full_rot.T
    off_thetaphi = coord.cart_to_sph(off_cart)[:, 1:]

    # Set phi convention as anti-clockwise from local vertical for
    # alt < 90. For alt >= 90, use an "over-the-top" extension of this
    # convention.
    off_thetaphi[:, 1] = np.pi - off_thetaphi[:, 1]

    return off_thetaphi


def cocr_to_thetaphi(
    cocr_beam: np.ndarray,
    pol_type: POLTYPE,
    angpos: np.ndarray,
    zenith: np.ndarray,
    altaz_pointing: np.ndarray,
) -> np.ndarray:
    """Convert a co-pol, cross-pol beam pattern for an antenna with polarisation
    axis aligned with the pointing-local vertical ("Y") or horizontal ("X") to 
    an Etheta, Ephi pattern relative to the sky coordinates.

    Parameters
    ----------
    cocr_beam
        (N, 2) co-pol, cross-pol beam pattern.
    pol_type : str
        "X" or "Y". If "X" ("Y") do the conversion assuming the polarisation
        axis is aligned with the pointing-local vertical (horizontal).
    angpos
        (N, 2) array of healpix theta, phi coordinates.
    zenith
        (2,) array of theta, phi zenith coordinates in radians.
    altaz_pointing
        (2,) array of altitude, azimuth coordinates in radians.
    Returns
    -------
        (N, 2) array of the converted beam pattern in Etheta, Ephi.
    """

    prot = pointing_rot_mat(zenith, altaz_pointing)
    # Local vertical is in local -xhat direction, local horizontal (phi=90 in pointing
    # offset convention) in local yhat direction. Convert these to the sky coordinates
    # using the inverse of the pointing transform.
    vertical = prot.T @ np.array([-1, 0, 0])
    horizontal = prot.T @ np.array([0, 1, 0])

    if pol_type == "X":
        co_dir, cr_dir = horizontal, vertical
    elif pol_type == "Y":
        co_dir, cr_dir = vertical, horizontal

    EthEph = (
        polpattern(angpos, co_dir) * cocr_beam[:, 0, None]
        + polpattern(angpos, cr_dir) * cocr_beam[:, 1, None]
    )

    return EthEph


def pointing_offset_separation(
    angpos: np.ndarray,
    zenith: np.ndarray,
    altaz_pointing: np.ndarray,
    degrees: Optional[bool] = False,
) -> np.ndarray:
    """Return the magnitude of the angular separation from a point in
    the sky defined by an altitude-azimuth pointing from a location
    with the given zenith.

    Parameters
    ----------
    angpos
        (N, 2) array of healpix theta, phi coordinates.
    zenith
        (2,) array of theta, phi zenith coordinates in radians.
    altaz_pointing
        (2,) array of altitude, azimuth coordinates in radians.
    degrees
        If True, return the offset separation angle in degrees. 
        Default False

    Returns
    -------
        (N,) array of angular separation angle from the pointing.
    """

    off_theta = pointing_offset_thetaphi_coords(angpos, zenith, altaz_pointing)[:, 0]

    if degrees:
        return np.degrees(off_theta)
    else:
        return off_theta


def pointing_offset_angles(
    angpos: np.ndarray,
    zenith: np.ndarray,
    altaz_pointing: np.ndarray,
    degrees: Optional[bool] = False,
) -> np.ndarray:
    """Offset latitudinal and longitudinal coordinates for an input
    pointing relative to an observer with an input zenith.

    Useful for interpolating beams as functions of pointing offset coordinates
    onto sky coordinates for a given pointing.

    .. math::

        \mathrm{latitude} = \\theta  \cos(\phi)

    .. math::

        \mathrm{longitude} = \\theta  \sin(\phi),

    For the :math:`\\theta` and :math:`\phi` convention used in 
    :py:func:`pointing_offset_thetaphi`.


    Parameters
    ----------
    angpos
        (N, 2) array of healpix theta, phi coordinates.
    zenith
        (2,) array of theta, phi zenith coordinates in radians.
    altaz_pointing
        (2,) array of altitude, azimuth coordinates in radians.
    degrees
        If True, return the offset great circle angles in degrees. 
        Default False

    Returns
    -------
        (N, 2) latitude, longitude offset angles relative to the pointing.
    """

    off_thetaphi = pointing_offset_thetaphi_coords(angpos, zenith, altaz_pointing)

    off_lat = off_thetaphi[:, 0] * np.cos(off_thetaphi[:, 1])
    off_lon = off_thetaphi[:, 0] * np.sin(off_thetaphi[:, 1])

    latlon = np.stack([off_lat, off_lon], axis=-1)
    if degrees:
        return np.degrees(latlon)
    else:
        return latlon


def airy_beam(
    separations: FloatArrayLike,
    wavelength: FloatArrayLike,
    diameter: FloatArrayLike,
    voltage: Optional[bool] = True,
    zero_over_horizon: Optional[bool] = True,
) -> FloatArrayLike:
    """An azimuthally symmetric Airy beam pattern with
    corresponding to a uniformly illuminated aperture with
    input effective dish diameter at a given wavelength.

    .. math::
        A(\\theta) = 2 \\frac{J_1(\pi D \sin\\theta/\lambda)}{\pi D \sin\\theta / \lambda}

    Parameters
    ----------
    separations
        Angular separations (:math:`\\theta`) to calculate the 
        pattern at, units of radians.
    wavelength
        Wavelength (:math:`\lambda`) in metres to use.
    diameter
        Effective dish diameter (:math:`D`) in metres to use.
    voltage
        If True, return the above expression for a voltage beam,
        otherwise return the square of this pattern for an power
        beam. Default: True.
    zero_over_horizon
        If True, explicitly set the pattern at :math:`\\theta > \pi/2` to
        zero. Note that with the above expression, the beam pattern is
        symmetric around :math:`\\theta = \pi/2`. Default: True.

    Returns
    -------
        The voltage or power beam pattern with shape following the
        broadcasting rules of the parameters.
    """

    x = np.pi * diameter / wavelength * np.sin(separations)
    out_shape = np.shape(x)
    x = np.atleast_1d(x)

    with np.errstate(divide="ignore"):
        out = 2 * bessel_j1(x) / x  # (Voltage Beam)

    out[x == 0] = 1

    if zero_over_horizon:
        horizon_mask = (separations <= np.pi / 2).astype(float)
        out = out * horizon_mask

    out = out.reshape(out_shape)

    if voltage:
        return out
    else:
        return out ** 2


def gaussian(
    separations: FloatArrayLike,
    wavelength: FloatArrayLike,
    diameter: FloatArrayLike,
    fwhm_factor: FloatArrayLike,
    voltage: Optional[bool] = True,
) -> FloatArrayLike:
    """An azimuthally symmetric gaussian beam pattern with
    FWHM of the power beam derived from an effective dish 
    diameter and wavelength.

    .. math::
        A(\\theta) = \exp\left[-\\theta^2/4/\sigma^2 \\right],

    where

    .. math::
        \sigma =  \\frac{f \lambda / D}{2\sqrt{2\log{2}}}

    Note that by default this returns the voltage beam pattern.

    Parameters
    ----------
    separations
        Angular separations (:math:`\\theta`) to calculate the 
        pattern at, units of radians.
    wavelength
        Wavelength (:math:`\lambda`) in metres to use for FWHM calculation.
    diameter
        Effective dish diameter (:math:`D`) in metres to use for FWHM calculation.
    fwhm_factor
        Scaling factor (:math:`f`) for relationship between FWHM and 
        wavelength/diameter. A value of one approximates the
        main lobe of a uniformally illuminated aperture.
    voltage
        If True, return the above expression for a voltage beam,
        otherwise return the square of this pattern for an power
        beam. Default: True.

    Returns
    -------
        The voltage or power beam pattern with shape following the
        broadcasting rules of the parameters.
    """

    fwhm = fwhm_factor * wavelength / diameter

    sigma = FWHM2SIGMA * fwhm
    arg = -(separations ** 2) / 2 / sigma ** 2

    if voltage:
        return np.exp(arg / 2)
    else:
        return np.exp(arg)


class AnalyticCoPolBeam(config.Reader, metaclass=abc.ABCMeta):
    """Base class for beams derived from analytic co-pol beam patterns 
    with options to add a scaled cross-pol term.

    Attributes
    ----------
    crosspol_type : :py:class:`caput.config.enum(["pure", "scaled"])`
        How to model the cross-pol component of the beam pattern. If
        "pure", cross-pol component is zero and we have a pure co-pol
        beam. If "scaled", the cross-pol voltage beam is a scaled version
        of the co-pol beam.
        Default: "pure".
    crosspol_scale_dB: :py:class:`caput.config.enum(["pure", "scaled"])`
        Amplitude of the scaled voltage cross-pol beam relative to the 
        co-pol voltage beam in dB. Not used if crosspol_type is "pure".
        Default: -40
    """

    crosspol_type = config.enum(["pure", "scaled"], default="pure")
    crosspol_scale_dB = config.Property(proptype=float, default=-40)

    supports_pointing = True

    @abc.abstractmethod
    def beam_func(
        self, tel_obj: TransitTelescope, feed_ind: int, freq_ind: int, altaz_pointing
    ):
        pass

    def __call__(
        self,
        tel_obj: TransitTelescope,
        feed_ind: int,
        freq_ind: int,
        altaz_pointing: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if altaz_pointing is None:
            altaz_pointing = np.radians([90, 180])

        copol_beam = self.beam_func(tel_obj, feed_ind, freq_ind, altaz_pointing)

        if tel_obj.num_pol_sky == 1:
            return copol_beam
        else:

            if self.crosspol_type == "pure":
                cocr_beam = np.stack([copol_beam, np.zeros_like(copol_beam)], axis=-1)
            elif self.crosspol_type == "scaled":
                cocr_beam = np.stack(
                    [copol_beam, copol_beam * (10 ** (self.crosspol_scale_dB / 10))],
                    axis=-1,
                )

            return cocr_to_thetaphi(
                cocr_beam,
                pol_type=tel_obj.polarisation[feed_ind],
                angpos=tel_obj._angpos,
                zenith=tel_obj.zenith,
                altaz_pointing=altaz_pointing,
            )


class AiryBeam(AnalyticCoPolBeam):
    """A Co-pol Airy beam pattern.

    Attributes
    ----------
    diameter : :py:class:`caput.config.Property(proptype=float)`
        The effective dish diameter in metres to use for the Airy beam pattern.
        See :py:func:`airy_beam` for details.
        Default: 6 m
    """

    diameter = config.Property(proptype=float, default=6.0)

    def beam_func(
        self, tel_obj: TransitTelescope, feed_ind: int, freq_ind: int, altaz_pointing
    ) -> np.ndarray:

        seps = pointing_offset_separation(
            tel_obj._angpos, tel_obj.zenith, altaz_pointing=altaz_pointing
        )

        return np.array(airy_beam(seps, tel_obj.wavelengths[freq_ind], self.diameter))


class GaussianBeam(AnalyticCoPolBeam):
    """A Co-pol Gaussian beam pattern.

    Attributes
    ----------
    diameter : :py:class:`caput.config.Property(proptype=float)`
        The effective dish diameter in metres to use for the Gaussian beam pattern.
        See :py:func:`gaussian` for details.
        Default: 6 m
    fwhm_factor: :py:class:`caput.config.Property(proptype=float)`
        The scaling factor to multiply the :math:`\lambda/D` term in
        the FWHM calculation. See :py:func:`gaussian` for details.
        Default: 1
    """

    dish_diameter = config.Property(proptype=float, default=6)
    fwhm_factor = config.Property(proptype=float, default=1.0)

    def beam_func(
        self, tel_obj: TransitTelescope, feed_ind: int, freq_ind: int, altaz_pointing
    ) -> np.ndarray:

        seps = pointing_offset_separation(
            tel_obj._angpos, tel_obj.zenith, altaz_pointing=altaz_pointing
        )

        return np.array(
            gaussian(
                seps,
                tel_obj.wavelengths[freq_ind],
                self.dish_diameter,
                self.fwhm_factor,
            )
        )


class HEALPixBeamFile(config.Reader):
    """Beam file read in from a HEALPix map. This uses the 
    `draco.core.containers.HEALPixBeam` container. Note that
    `draco` is therefore a requirement to use beams stored in 
    this way. However, it is not included in the `requirements.txt`
    and must be installed separately to avoid a circular dependency.

    Attributes
    ----------
    filename : :py:class:`caput.config.Property(proptype=str)`
        Filename of the HEALPix beam file.
    single_input : :py:class:`caput.config.Property(proptype=bool)`
        If True, every the 0th `input` index of the HEALPixBeam container
        will be used for all `feed_ind` values of the telescope. If False,
        the `feed_ind` will be used to index the `input`.
        Default True.
    freq_index_type : :py:class:`caput.config.enum(["matched", "nearest"])`
        If "matched", the HEALPixBeam container's frequency index will be
        indexed by the telescope's `freq_ind`. If "nearest", frequency index
        of the HEALPixBeam nearest to the frequency requested from the 
        telescope object will be used.
        Default "matched".
    """

    filename = config.Property(proptype=str)
    single_input = config.Property(proptype=bool, default=True)
    freq_index_type = config.enum(["matched", "nearest"], default="matched")

    supports_pointing = False

    def __call__(
        self, tel_obj: TransitTelescope, feed_ind: int, freq_ind: int
    ) -> np.ndarray:

        from draco.core.containers import HEALPixBeam

        # This can't be pickled and therefore can't be an
        # attribute of the class.
        beam_file = HEALPixBeam.from_file(
            self.filename, mode="r", distributed=False, ondisk=True
        )

        pol_type = tel_obj.polarisation[feed_ind]
        ipol = beam_file.pol.astype(str).tolist().index(pol_type)

        if self.single_input:
            ifeed = 0
        else:
            ifeed = feed_ind

        if self.freq_index_type == "matched":
            ifreq = freq_ind
            if beam_file.freq[ifreq] != tel_obj.frequencies[ifreq]:
                raise Exception(
                    f'freq_index_type is set to "matched" but frequency metadata '
                    f"in beam file {self.filename} does not match the telescope "
                    f"object for frequency index {ifreq}."
                )
        elif self.freq_index_type == "nearest":
            abs_diffs = np.abs(tel_obj.frequencies[freq_ind] - beam_file.freq)
            ifreq = int(np.argmin(abs_diffs))
            # TODO raise warning if abs_diffs[ifreq] passes some threshold.

        beam = beam_file.beam[ifreq, ipol, ifeed, :]

        Eth = hp.ud_grade(beam["Et"], tel_obj._nside)
        Eph = hp.ud_grade(beam["Ep"], tel_obj._nside)

        return np.stack([Eth, Eph], axis=-1).astype(np.complex128)


AVAILABLE_BEAMS = {
    "airy": AiryBeam,
    "gaussian": GaussianBeam,
    "healpix": HEALPixBeamFile,
}
