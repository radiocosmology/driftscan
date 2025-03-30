"""Telescope classes and beam transfer routines related to external beam models."""

import abc
import logging

import numpy as np
import healpy
from scipy.interpolate import RectBivariateSpline

from caput import cache
from caput import config

from cora.util import coord, hputil, units

from drift.core import telescope
from drift.telescope import cylinder, cylbeam

from draco.core.containers import ContainerBase, GridBeam, HEALPixBeam


# Create logger object
logger = logging.getLogger(__name__)


class PolarisedTelescopeExternalBeam(telescope.PolarisedTelescope):
    """Base class for polarised telescope with external beam model.

    Beam model is read in from a file.

    Attributes
    ----------
    primary_beam_filename : str
        Path to the file containing the primary beam. Can either be a Healpix beam or a
        GridBeam.
    freq_interp_beam : bool, optional
        Interpolate between neighbouring frequencies if we don't have a beam for every
        frequency channel. Default: False.
    force_real_beam : bool, optional
        Ensure the output beam is real, regardless of what the datatype of the beam file
        is. This can help save memory if the saved beam is complex but you know the
        imaginary part is zero. Default: False.
    """

    primary_beam_filename = config.Property(proptype=str)
    freq_interp_beam = config.Property(proptype=bool, default=False)
    force_real_beam = config.Property(proptype=bool, default=False)

    def _finalise_config(self):
        """Get the beam file object."""
        (
            self._primary_beam,
            self._is_grid_beam,
            self._beam_freq,
            self._beam_nside,
            self._beam_pol_map,
            self._output_dtype,
        ) = self._load_external_beam(self.primary_beam_filename)

        self._x_grid = None
        self._y_grid = None
        self._x_tel = None
        self._pix_mask = None

        self._pvec_x = None
        self._pvec_y = None

    def _load_external_beam(self, filename):
        """Load beam from file, and return container and metadata."""
        logger.debug("Reading beam model from {}...".format(filename))
        beam = ContainerBase.from_file(
            filename, mode="r", distributed=False, ondisk=True
        )

        is_grid_beam = isinstance(beam, GridBeam)
        is_healpix_beam = isinstance(beam, HEALPixBeam)

        # cache axes
        beam_freq = beam.freq[:]
        beam_nside = None if is_grid_beam else beam.nside

        # TODO must use bytestring here because conversion doesn't work with ondisk=True
        if is_grid_beam:
            beam_pol_map = {
                "X": list(beam.pol[:]).index(b"XX"),
                "Y": list(beam.pol[:]).index(b"YY"),
            }
        else:
            beam_pol_map = {
                "X": list(beam.pol[:]).index(b"X"),
                "Y": list(beam.pol[:]).index(b"Y"),
            }

        if len(beam.input) > 1:
            raise ValueError("Per-feed beam model not supported for now.")

        # If a HEALPixBeam, must check types of theta and phi fields
        if is_healpix_beam:
            hpb_types = [v[0].type for v in beam.beam.dtype.fields.values()]

            complex_beam = np.all(
                [np.issubclass_(hpbt, np.complexfloating) for hpbt in hpb_types]
            )
        else:
            complex_beam = np.issubclass_(beam.beam.dtype.type, np.complexfloating)

        output_dtype = (
            np.complex128 if complex_beam and not self.force_real_beam else np.float64
        )

        return beam, is_grid_beam, beam_freq, beam_nside, beam_pol_map, output_dtype

    def beam(self, feed, freq_id):
        """Compute the beam pattern.

        Parameters
        ----------
        feed : int
            Feed index.
        freq_id : int
            Frequency ID.

        Returns
        -------
        beam : np.ndarray[pixel, pol]
            Return the vector beam response at each point in the Healpix grid. This
            array is of type `np.float64` if the input beam pattern is real, or if
            `force_real_beam` is set; otherwise it is of type `np.complex128`.
        """
        if self._is_grid_beam:
            # Either we haven't set up interpolation coords yet, or the nside has
            # changed
            if (self._beam_nside is None) or (self._beam_nside != self._nside):
                self._beam_nside = self._nside
                (
                    self._x_grid,
                    self._y_grid,
                    self._x_tel,
                    self._pix_mask,
                ) = self._setup_gridbeam_interpolation(self._primary_beam)
                self._setup_polpattern()

        map_out = self._evaluate_external_beam(
            feed,
            freq_id,
            self._primary_beam,
            self._beam_pol_map,
            self._beam_freq,
            self._is_grid_beam,
            self._beam_nside,
            self._output_dtype,
            self._x_grid,
            self._y_grid,
            self._x_tel,
            self._pix_mask,
        )

        return map_out

    def _evaluate_external_beam(
        self,
        feed,
        freq_id,
        primary_beam,
        beam_pol_map,
        beam_freq,
        is_grid_beam,
        beam_nside,
        output_dtype,
        x_grid=None,
        y_grid=None,
        x_tel=None,
        pix_mask=None,
    ):
        tel_freq = self.frequencies
        nside = self._nside
        npix = healpy.nside2npix(nside)

        # Get beam model polarization index corresponding to pol of requested feed
        if self.polarisation[feed] == "X":
            pol_ind = beam_pol_map["X"]
        elif self.polarisation[feed] == "Y":
            pol_ind = beam_pol_map["Y"]
        else:
            raise ValueError(
                f"Unexpected polarisation ({self.polarisation[feed]} for feed {feed}!)"
            )

        # Find nearest frequency
        freq_sel = _nearest_freq(
            tel_freq, beam_freq, freq_id, single=(not self.freq_interp_beam)
        )
        # Raise an error if we can't find any suitable frequency
        if len(freq_sel) == 0:
            raise ValueError(f"No beam model spans frequency {tel_freq[freq_id]}.")

        if is_grid_beam:
            # Interpolate gridbeam onto Healpix
            beam_map = self._interpolate_gridbeam(
                freq_sel,
                pol_ind,
                primary_beam,
                beam_pol_map,
                x_grid,
                y_grid,
                x_tel,
                pix_mask,
            )

        else:  # Healpix input beam - just need to change to the required resolution
            beam_map = primary_beam.beam[freq_sel, pol_ind, 0, :]

            # Check resolution and resample to a better resolution if needed
            if nside != beam_nside:
                if nside > beam_nside:
                    logger.warning(
                        f"Requested nside={nside} higher than that of "
                        f"beam {beam_nside}"
                    )

                logger.debug(
                    "Resampling external beam from nside {:d} to {:d}".format(
                        beam_nside,
                        nside,
                    )
                )
                beam_map_new = np.zeros((len(freq_sel), npix), dtype=beam_map.dtype)
                beam_map_new["Et"] = healpy.ud_grade(beam_map["Et"], nside)
                beam_map_new["Ep"] = healpy.ud_grade(beam_map["Ep"], nside)
                beam_map = beam_map_new

        map_out = np.empty((npix, 2), dtype=output_dtype)

        # Pull out the real part of the beam if we are forcing a conversion. This should
        # do nothing if the array is already real
        def _conv_real(x):
            if self.force_real_beam:
                x = x.real
            return x

        if len(freq_sel) == 1:
            # Exact match
            map_out[:, 0] = _conv_real(beam_map["Et"][0])
            map_out[:, 1] = _conv_real(beam_map["Ep"][0])
        else:
            # Interpolate between pair of frequencies
            freq_high = beam_freq[freq_sel[1]]
            freq_low = beam_freq[freq_sel[0]]
            freq_int = tel_freq[freq_id]

            alpha = (freq_high - freq_int) / (freq_high - freq_low)
            beta = (freq_int - freq_low) / (freq_high - freq_low)

            map_out[:, 0] = _conv_real(
                beam_map["Et"][0] * alpha + beam_map["Et"][1] * beta
            )
            map_out[:, 1] = _conv_real(
                beam_map["Ep"][0] * alpha + beam_map["Ep"][1] * beta
            )

        return map_out

    def _setup_gridbeam_interpolation(self, primary_beam):
        # Grid beam coordinates
        x_grid = primary_beam.phi[:]
        y_grid = primary_beam.theta[:]

        # Celestial coordinates
        angpos = hputil.ang_positions(self._nside)
        x_cel = coord.sph_to_cart(angpos).T

        # Rotate to telescope coords:
        # first align y with N, then polar axis with NCP
        x_tel = cylbeam.rotate_ypr(
            (1.5 * np.pi, np.radians(90.0 - self.latitude), 0), *x_cel
        )

        # Mask any pixels outside grid
        x_t, y_t, z_t = x_tel
        pix_mask = (
            (z_t > 0)
            & (np.abs(x_t) < np.abs(x_grid.max()))
            & (np.abs(y_t) < np.abs(y_grid.max()))
        )

        return x_grid, y_grid, x_tel, pix_mask

    def _setup_polpattern(self):
        # Pre-compute polarisation pattern.
        # Taken from driftscan
        zenith = np.array([np.pi / 2.0 - np.radians(self.latitude), 0.0])
        that, phat = coord.thetaphi_plane_cart(zenith)
        xhat, yhat, zhat = cylbeam.rotate_ypr(
            [0.0, 0.0, 0.0], phat, -that, coord.sph_to_cart(zenith)
        )

        angpos = hputil.ang_positions(self._nside)
        self._pvec_x = cylbeam.polpattern(angpos, xhat)
        self._pvec_y = cylbeam.polpattern(angpos, yhat)

    def _interpolate_gridbeam(
        self, f_sel, p_ind, primary_beam, beam_pol_map, x_grid, y_grid, x_tel, pix_mask
    ):
        x, y = x_grid, y_grid
        x_t, y_t, z_t = x_tel
        mask = pix_mask

        # Interpolation routine requires increasing axes
        reverse_x = (np.diff(x) < 0).any()
        if reverse_x:
            x = x[::-1]

        npix = healpy.nside2npix(self._nside)
        beam_out = np.zeros(
            (len(f_sel), npix), dtype=HEALPixBeam._dataset_spec["beam"]["dtype"]
        )
        for i, fi in enumerate(f_sel):
            # For now we just use the magnitude. Assumes input is power beam
            beam = primary_beam.beam[fi, p_ind, 0]
            if reverse_x:
                beam = beam[:, ::-1]
            beam_spline = RectBivariateSpline(y, x, np.sqrt(np.abs(beam)))

            # Beam amplitude
            amp = np.zeros(npix, dtype=beam.real.dtype)
            amp[mask] = beam_spline(y_t[mask], x_t[mask], grid=False)

            # Polarisation projection
            pvec = self._pvec_x if beam_pol_map["X"] == p_ind else self._pvec_y

            beam_out[i]["Et"] = amp * pvec[:, 0]
            beam_out[i]["Ep"] = amp * pvec[:, 1]

        return beam_out


def _nearest_freq(tel_freq, map_freq, freq_id, single=False):
    """Find nearest neighbor frequencies. Assumes map frequencies
    are uniformly spaced.

    Parameters
    ----------
    tel_freq : float
        frequencies from telescope object.
    map_freq : float
        frequencies from beam map file.
    freq_id : int
        frequency selection.
    single : bool
        Only return the single nearest neighbour.

    Returns
    -------
    freq_ind : list of neighboring map frequencies matched to tel_freq.

    """

    diff_freq = abs(map_freq - tel_freq[freq_id])
    if single:
        return np.array([np.argmin(diff_freq)])

    map_freq_width = abs(map_freq[1] - map_freq[0])
    match_mask = diff_freq < map_freq_width

    freq_ind = np.nonzero(match_mask)[0]

    return freq_ind


class PolarisedCylinderTelescopeExternalBeam(
    cylinder.CylinderTelescope, PolarisedTelescopeExternalBeam
):
    """Class for polarised cylinder telescope with external beam model.

    Repeats some code from SimplePolarisedTelescope.
    """

    @property
    def polarisation(self):
        """
        Polarisation map.

        Returns
        -------
        pol : np.ndarray
            One-dimensional array with the polarization for each feed ('X' or 'Y').
        """
        return np.asarray(
            ["X" if feed % 2 == 0 else "Y" for feed in self.beamclass], dtype=np.str
        )

    @property
    def beamclass(self):
        """Simple beam mode of dual polarisation feeds."""
        nsfeed = self._single_feedpositions.shape[0]
        return np.concatenate((np.zeros(nsfeed), np.ones(nsfeed))).astype(np.int64)

    @property
    def feedpositions(self):
        return np.concatenate((self._single_feedpositions, self._single_feedpositions))
