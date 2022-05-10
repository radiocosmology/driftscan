"""Telescope classes and beam transfer routines related to external beam models."""

import os
import abc
import logging
from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import healpy
import h5py
from scipy.interpolate import RectBivariateSpline

from caput import cache, config, misc, mpiutil

from cora.util import coord, hputil, units

from drift.core import beamtransfer, kltransform, telescope
from drift.telescope import cylinder, cylbeam
from drift.util import util
from drift.core.kltransform import eigh_gen
from drift.core.beamtransfer import matrix_image, matrix_nullspace
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

        complex_beam = np.issubclass_(beam.beam.dtype.type, np.complexfloating)
        output_dtype = (
            np.complex128 if complex_beam and not self.force_real_beam else np.float64
        )

        return beam, is_grid_beam, beam_freq, beam_nside, beam_pol_map, output_dtype

    def beam(self, feed, freq_id, angpos=None):
        """Compute the beam pattern.

        Parameters
        ----------
        feed : int
            Feed index.
        freq_id : int
            Frequency ID.
        angpos : np.ndarray[nposition, 2], optional
            Angular position on the sky (in radians). If not provided, default to the
            _angpos class attribute. Currently not implemented.

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


class CylinderPerturbedTemplates(PolarisedCylinderTelescopeExternalBeam):
    """Polarised cylinder telescope, expanded to include beam perturbation templates.

    Templates are internally tracked by beamclass: the beamclass for template i is 2*i
    (for X pol) or 2*i+1 (for Y pol).

    Attributes
    ----------
    base_beam_filename : str, optional
        Path to the file containing the zeroth-order ("base") beam. Can either be a
        Healpix beam or a GridBeam. If not specified, we use the default driftscan
        beam model for the base beam. Default: None.
    beam_template_filenames : list
        List of filenames corresponding to beam perturbation templates. Must have at
        least one item.
    """

    base_beam_filename = config.Property(proptype=str, default=None)
    beam_template_filenames = config.Property(proptype=list)

    def _finalise_config(self):
        """Load base beam and beam templates from files."""

        # Get number of templates
        self.n_pert = len(self.beam_template_filenames)
        if self.n_pert == 0:
            raise InputError("Need to specify at least one beam perturbation template!")

        # If file for base beam is specified, load it in
        if self.base_beam_filename is not None:
            (
                self._primary_beam,
                self._is_grid_beam,
                self._beam_freq,
                self._beam_nside,
                self._beam_pol_map,
                self._output_dtype,
            ) = self._load_external_beam(self.base_beam_filename)
            self.ext_base_beam = True

            self._x_grid = None
            self._y_grid = None
            self._x_tel = None
            self._pix_mask = None
        else:
            self.ext_base_beam = False

        self._pvec_x = None
        self._pvec_y = None

        # Load information for each template into separate dict, indexed by template
        # number in self._templates dict.
        # Note that the indexing in self._templates begins with 1, with index 0
        # reserved for the base beam.
        self._templates = {}
        for i in range(self.n_pert):
            t = {}
            (
                t["primary_beam"],
                t["is_grid_beam"],
                t["beam_freq"],
                t["beam_nside"],
                t["beam_pol_map"],
                t["output_dtype"],
            ) = self._load_external_beam(self.beam_template_filenames[i])
            self._templates[i + 1] = t

    @property
    def beamclass(self):
        """Simple beam mode of dual polarisation feeds."""
        nsfeed = self._single_feedpositions.shape[0]

        # Evens are X-pol feed for each perturbation, odds are Y-pol feed
        beamclass = [bc * np.ones(nsfeed) for bc in range(2 * (self.n_pert + 1))]

        return np.concatenate(beamclass).astype(np.int64)

    @property
    def feedpositions(self):
        beampos = [self._single_feedpositions for bc in range(2 * (self.n_pert + 1))]
        return np.concatenate(beampos)

    def beam(self, feed, freq_id, angpos=None):
        """Get the beam pattern.

        Examines the beamclass associated with the requested feed to determine whether
        to use the base beam or one of the templates.

        Parameters
        ----------
        feed : int
            Feed index.
        freq_id : int
            Frequency ID.
        angpos : np.ndarray[nposition, 2], optional
            Angular position on the sky (in radians). If not provided, default to the
            _angpos class attribute. Currently not implemented.

        Returns
        -------
        beam : np.ndarray[pixel, pol]
            Return the vector beam response at each point in the Healpix grid. This
            array is of type `np.float64` if the input beam pattern is real, or if
            `force_real_beam` is set, otherwise it is of type `np.complex128`.
        """
        # Get perturbation number from beamclass (remember, evens are X-pol and odds are
        # Y-pol)
        pert_num = int(self.beamclass[feed] // 2)

        if pert_num == 0:
            # Base beam
            if self.ext_base_beam:

                if self._is_grid_beam:
                    # Either we haven't set up interpolation coords yet, or the nside
                    # has changed
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

            else:
                # Use driftscan beam model if no external base beam has been specified
                if self.polarisation[feed] == "X":
                    map_out = self.beamx(feed, freq_id)
                else:
                    map_out = self.beamy(feed, freq_id)

        else:
            # pert_num > 0, so we're computing for a beam perturbation template
            if self._templates[pert_num]["is_grid_beam"]:
                # Either we haven't set up interpolation coords yet, or the nside
                # has changed
                if (self._templates[pert_num]["beam_nside"] is None) or (
                    self._templates[pert_num]["beam_nside"] != self._nside
                ):
                    self._templates[pert_num]["beam_nside"] = self._nside
                    (
                        self._templates[pert_num]["x_grid"],
                        self._templates[pert_num]["y_grid"],
                        self._templates[pert_num]["x_tel"],
                        self._templates[pert_num]["pix_mask"],
                    ) = self._setup_gridbeam_interpolation(
                        self._templates[pert_num]["primary_beam"]
                    )

                    self._setup_polpattern()

            map_out = self._evaluate_external_beam(
                feed, freq_id, **self._templates[pert_num]
            )

        return map_out

    def beamx(self, feed, freq):
        """Driftscan beam model for X pol."""
        return cylbeam.beam_x(
            self._angpos,
            self.zenith,
            self.cylinder_width / self.wavelengths[freq],
            self.fwhm_e,
            self.fwhm_h,
        )

    def beamy(self, feed, freq):
        """Driftscan beam model for Y pol."""
        return cylbeam.beam_y(
            self._angpos,
            self.zenith,
            self.cylinder_width / self.wavelengths[freq],
            self.fwhm_e,
            self.fwhm_h,
        )


class BeamTransferTemplates(beamtransfer.BeamTransfer):
    def _svdfile(self, mi: int, t: Optional[int] = None):
        # Pattern to form the `m` ordered file.

        if t is None:
            f = "/svd.hdf5"
        else:
            f = "/svd_template%d.hdf5" % t

        pat = self.directory + "/beam_m/" + util.natpattern(self.telescope.mmax) + f
        return pat % mi

    def _generate_svdfile_m_in_basebeam_basis(
        self,
        mi,
        filename,
        skip_svd_inv=False,
        bl_mask=None,
        bl_mask2=None,
    ):

        if bl_mask is None:
            bl_mask = [True for i in range(self.telescope.npairs)]

        if bl_mask2 is not None and np.sum(bl_mask2) != np.sum(bl_mask):
            raise ValueError("bl_mask2 must select same number of elements as bl_mask!")

        npairs = np.sum(bl_mask)
        ntel = 2 * npairs

        # Open file to write SVD results into, using caput.misc.lock_file()
        # to guard against crashes while the file is open. With preserve=True,
        # the temp file will be saved with a period in front of its name
        # if a crash occurs.
        with misc.lock_file(filename, preserve=True) as fs_lock:
            with h5py.File(fs_lock, "w") as fs:

                # Create a chunked dataset for writing the SVD beam matrix into.
                dsize_bsvd = (
                    self.telescope.nfreq,
                    self.svd_len(ntel),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    1,
                    min(10, self.svd_len(ntel)),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                if not skip_svd_inv:
                    # Create a chunked dataset for writing the inverse SVD beam matrix
                    # into.
                    dsize_ibsvd = (
                        self.telescope.nfreq,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        self.svd_len(ntel),
                    )
                    csize_ibsvd = (
                        1,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        min(10, self.svd_len(ntel)),
                    )
                    dset_ibsvd = fs.create_dataset(
                        "invbeam_svd",
                        dsize_ibsvd,
                        chunks=csize_ibsvd,
                        compression="lzf",
                        dtype=np.complex128,
                    )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (self.telescope.nfreq, self.svd_len(ntel), ntel)
                csize_ut = (1, min(10, self.svd_len(ntel)), ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (self.telescope.nfreq, self.svd_len(ntel))
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ## For each frequency in the m-files read in the block, SVD it,
                ## and construct the new beam matrix, and save.
                for fi in np.arange(self.telescope.nfreq):

                    # Get U^T for base beam, packed as [svd_len, ntel]
                    ut = self.beam_ut(mi, fi)

                    # Read the positive and negative m beams, and combine into one.
                    # No need to prewhiten, because the factor of N^{-1/2} is already
                    # contained in ut
                    bf = self.beam_m(mi, fi)[:, bl_mask, :, :].reshape(
                        ntel,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )
                    if bl_mask2 is not None:
                        bf += self.beam_m(mi, fi)[:, bl_mask2, :, :].reshape(
                            ntel,
                            self.telescope.num_pol_sky,
                            self.telescope.lmax + 1,
                        )

                    # Reshape beam to 2D matrix and apply U^T to template BTM
                    bf = bf.reshape(ntel, -1)
                    bf2 = np.dot(ut, bf)

                    # Perform SVD and find U and Sigma for projected template BTM.
                    # Don't cut any modes here - just want to get the SVD decomposition
                    u_template, s_template = beamtransfer.matrix_nullspace(
                        bf2, rtol=10, errmsg=("Template SVD m=%i f=%i" % (mi, fi))
                    )
                    u_template_t = u_template.T.conj()
                    nmodes = u_template_t.shape[0]
                    sig = s_template[:nmodes]
                    beam = np.dot(u_template_t, bf2)

                    # We flip the order of the SVs to be in ascending instead of
                    # descending order, so that cutting high SVs corresponds to cutting
                    # elements from the end of the list.
                    u_template_t = u_template_t[::-1]
                    beam = beam[::-1]
                    sig = sig[::-1]

                    # Save out the evecs (for transforming from the telescope frame
                    # into the SVD basis)
                    dset_ut[fi, :nmodes, :nmodes] = u_template_t

                    # Save out the modified beam matrix (for mapping from the sky
                    # into the SVD basis)
                    dset_bsvd[fi, :nmodes] = beam.reshape(
                        nmodes, self.telescope.num_pol_sky, self.telescope.lmax + 1
                    )

                    if not skip_svd_inv and beam.shape[0] > 0:
                        # Find the pseudo-inverse of the beam matrix and save to
                        # disk. First try la.pinv, which uses a least-squares
                        # solver.
                        try:
                            ibeam = la.pinv(beam)
                        except la.LinAlgError as e:
                            # If la.pinv fails, try la.pinv2, which is SVD-based and
                            # more likely to succeed. If successful, add file
                            # attribute
                            # indicating pinv2 was used for this frequency.
                            logger.info(
                                "***Beam-SVD pesudoinverse (scipy.linalg.pinv) "
                                f"failure: m = {mi}, fi = {fi}. Trying pinv2..."
                            )
                            try:
                                ibeam = la.pinv2(beam)
                                if "inv_bsvd_from_pinv2" not in fs.attrs.keys():
                                    fs.attrs["inv_bsvd_from_pinv2"] = [fi]
                                else:
                                    bad_freqs = fs.attrs["inv_bsvd_from_pinv2"]
                                    fs.attrs["inv_bsvd_from_pinv2"] = bad_freqs.append(
                                        fi
                                    )
                            except:
                                # If pinv2 fails, print error message
                                raise Exception(
                                    "Beam-SVD pseudoinverse (scipy.linalg.pinv2) "
                                    "failure: m = %d, fi = %d" % (mi, fi)
                                )

                        dset_ibsvd[fi, :, :, :nmodes] = ibeam.reshape(
                            self.telescope.num_pol_sky,
                            self.telescope.lmax + 1,
                            nmodes,
                        )

                    # Save out the singular values for each block
                    dset_sig[fi, :nmodes] = sig

                # Write a few useful attributes.
                # fs.attrs["baselines"] = self.telescope.baselines[]
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies

    def _generate_svdfiles(self, regen=False, skip_svd_inv=False):

        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        def _file_list(mi):
            return [self._svdfile(mi)] + [
                self._svdfile(mi, t) for t in range(1, self.telescope.n_pert + 1)
            ]

        m_list = np.arange(self.telescope.mmax + 1)
        if mpiutil.rank0:
            # For each m, check whether the files exist, if so, whether we
            # can open them. If these tests all pass, we can skip this m.
            # Otherwise, we need to generate a SVD files for that m.
            for mi in m_list:

                file_list = _file_list(mi)

                run_mi = False
                for fname in file_list:
                    if run_mi:
                        continue

                    if os.path.exists(fname) and not regen:
                        # File may exist but be un-openable, so we catch such an
                        # exception. This shouldn't happen if we use caput.misc.lock_file(),
                        # but we catch it just in case.
                        try:
                            fs = h5py.File(fname, "r")
                            fs.close()

                            logger.info(
                                f"m index {mi}. Complete file: {fname} exists."
                                "Skipping..."
                            )

                        except Exception:
                            logger.info(
                                f"m index {mi}. ***INCOMPLETE file: {fname} "
                                "exists. Will regenerate..."
                            )
                            run_mi = True

                    else:
                        run_mi = True

                if not run_mi:
                    m_list[mi] = -1

            # Reduce m_list to the m's that we need to compute
            m_list = m_list[m_list != -1]

        # Broadcast reduced list to all tasks
        m_list = mpiutil.bcast(m_list)

        # Print m list
        if mpiutil.rank0:
            logger.info(f"m's remaining in beam SVD computation: {m_list}")
        mpiutil.barrier()

        # Distribute m list over tasks, and do computations
        for mi in mpiutil.partition_list_mpi(m_list):
            file_list = _file_list(mi)
            for t, fname in enumerate(file_list):
                logger.info(f"m index {mi}. Creating SVD file: {fname}")
                if t == 0:
                    bl_mask = [
                        (x[0] in [0, 1] and x[1] in [0, 1])
                        for x in self.telescope.beamclass[self.telescope.uniquepairs]
                    ]
                    self._generate_svdfile_m(
                        mi,
                        skip_svd_inv=skip_svd_inv,
                        bl_mask=bl_mask,
                        filename=fname,
                    )
                else:
                    bl_mask = [
                        (x[0] in [0, 1] and x[1] in [2 * t, 2 * t + 1])
                        for x in self.telescope.beamclass[self.telescope.uniquepairs]
                    ]
                    bl_mask2 = [
                        (x[0] in [2 * t, 2 * t + 1] and x[1] in [0, 1])
                        for x in self.telescope.beamclass[self.telescope.uniquepairs]
                    ]
                    self._generate_svdfile_m_in_basebeam_basis(
                        mi, fname, skip_svd_inv=True, bl_mask=bl_mask, bl_mask2=bl_mask2
                    )

        # If we're part of an MPI run, synchronise here.self._svdfile(mi)
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()

    @util.cache_last
    def beam_singularvalues(self, mi: int, t: Optional[int] = None) -> np.ndarray:
        """Fetch the vector of beam singular values for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len)
        """
        if t is None:
            return beamtransfer._load_beam_f(self._svdfile(mi), "singularvalues")
        else:
            return beamtransfer._load_beam_f(self._svdfile(mi, t), "singularvalues")

    def _collect_svd_spectrum(self):
        """Gather the SVD spectra into a single files for base beam and each template."""

        svd_func = lambda mi: self.beam_singularvalues(mi)

        ntel = 2 * np.sum(
            [
                (x[0] in [0, 1] and x[1] in [0, 1])
                for x in self.telescope.beamclass[self.telescope.uniquepairs]
            ]
        )

        svdspectrum = kltransform.collect_m_array(
            list(range(self.telescope.mmax + 1)),
            svd_func,
            (self.nfreq, self.svd_len(ntel)),
            np.float64,
        )

        if mpiutil.rank0:
            with h5py.File(self.directory + "/svdspectrum.hdf5", "w") as f:
                f.create_dataset("singularvalues", data=svdspectrum)

        for t in range(1, self.telescope.n_pert + 1):

            svd_func = lambda mi: self.beam_singularvalues(mi, t)

            svdspectrum = kltransform.collect_m_array(
                list(range(self.telescope.mmax + 1)),
                svd_func,
                (self.nfreq, self.svd_len(ntel)),
                np.float64,
            )

            if mpiutil.rank0:
                with h5py.File(
                    self.directory + "/svdspectrum_template%d.hdf5" % t, "w"
                ) as f:
                    f.create_dataset("singularvalues", data=svdspectrum)

        mpiutil.barrier()


class BeamTransferSingleStepFilterTemplate(beamtransfer.BeamTransfer):
    def __init__(self, directory, **kwargs):

        super(BeamTransferSingleStepFilterTemplate, self).__init__(directory, **kwargs)

        if self.telescope.n_pert != 1:
            raise NotImplementedError(
                "Can only use BeamTransferSingleStepFilterTemplate for a single "
                "template!"
            )

    def _generate_svdfile_m(self, mi, skip_svd_inv=False):

        bl_mask = [
            (x[0] in [0, 1] and x[1] in [2, 3])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask2 = [
            (x[0] in [2, 3] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]

        npairs = np.sum(bl_mask)
        ntel = 2 * npairs

        # Open file to write SVD results into, using caput.misc.lock_file()
        # to guard against crashes while the file is open. With preserve=True,
        # the temp file will be saved with a period in front of its name
        # if a crash occurs.
        with misc.lock_file(self._svdfile(mi), preserve=True) as fs_lock:
            with h5py.File(fs_lock, "w") as fs:

                # Create a chunked dataset for writing the SVD beam matrix into.
                dsize_bsvd = (
                    self.telescope.nfreq,
                    self.svd_len(ntel),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    1,
                    min(10, self.svd_len(ntel)),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                if not skip_svd_inv:
                    # Create a chunked dataset for writing the inverse SVD beam matrix
                    # into
                    dsize_ibsvd = (
                        self.telescope.nfreq,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        self.svd_len(ntel),
                    )
                    csize_ibsvd = (
                        1,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        min(10, self.svd_len(ntel)),
                    )
                    dset_ibsvd = fs.create_dataset(
                        "invbeam_svd",
                        dsize_ibsvd,
                        chunks=csize_ibsvd,
                        compression="lzf",
                        dtype=np.complex128,
                    )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (self.telescope.nfreq, self.svd_len(ntel), ntel)
                csize_ut = (1, min(10, self.svd_len(ntel)), ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (self.telescope.nfreq, self.svd_len(ntel))
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ## For each frequency in the m-files read in the block, SVD it,
                ## and construct the new beam matrix, and save.
                for fi in np.arange(self.telescope.nfreq):

                    # Read the positive and negative m beams, and combine into one.
                    bf = self.beam_m(mi, fi)[:, bl_mask, :, :].reshape(
                        ntel,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )
                    bf += self.beam_m(mi, fi)[:, bl_mask2, :, :].reshape(
                        ntel,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )

                    # Prewhiten beam transfer matrix by multiplying by N^-1/2 matrix
                    noisew = self.telescope.noisepower(
                        np.arange(self.telescope.npairs)[bl_mask], fi
                    ).flatten() ** (-0.5)
                    noisew = np.concatenate([noisew, noisew])
                    bf = bf * noisew[:, np.newaxis, np.newaxis]

                    # Reshape total beam to a 2D matrix
                    bfr = bf.reshape(ntel, -1)

                    # SVD 1 - coarse projection onto sky-modes.
                    # This is the only SVD we do for the template BTM
                    u1, s1 = beamtransfer.matrix_image(
                        bfr, rtol=0.0, errmsg=("SVD1 m=%i f=%i" % (mi, fi))
                    )

                    ut1 = u1.T.conj()
                    bf1 = np.dot(ut1, bfr)
                    nmodes = ut1.shape[0]

                    sig = s1[:nmodes]
                    beam = np.dot(ut1, bfr)

                    # We flip the order of the SVs to be in ascending instead of
                    # descending order, so that cutting high SVs corresponds to cutting
                    # elements from the end of the list.
                    ut1 = ut1[::-1]
                    beam = beam[::-1]
                    sig = sig[::-1]

                    # Save out the evecs (for transforming from the telescope frame
                    # into the SVD basis). We multiply ut by N^{-1/2} because ut
                    # must act on N^{-1/2} v, not v alone (where v are the
                    # visibilities), so we include that factor of N^{-1/2} in
                    # dset_ut so that we can apply it directly to v in the future.
                    dset_ut[fi, :nmodes] = ut1 * noisew[np.newaxis, :]

                    # Save out the modified beam matrix (for mapping from the sky
                    # into the SVD basis)
                    dset_bsvd[fi, :nmodes] = beam.reshape(
                        nmodes, self.telescope.num_pol_sky, self.telescope.lmax + 1
                    )

                    if not skip_svd_inv and beam.shape[0] > 0:
                        # Find the pseudo-inverse of the beam matrix and save to
                        # disk. First try la.pinv, which uses a least-squares
                        # solver.
                        try:
                            ibeam = la.pinv(beam)
                        except la.LinAlgError as e:
                            # If la.pinv fails, try la.pinv2, which is SVD-based and
                            # more likely to succeed. If successful, add file
                            # attribute
                            # indicating pinv2 was used for this frequency.
                            logger.info(
                                "***Beam-SVD pesudoinverse (scipy.linalg.pinv) "
                                f"failure: m = {mi}, fi = {fi}. Trying pinv2..."
                            )
                            try:
                                ibeam = la.pinv2(beam)
                                if "inv_bsvd_from_pinv2" not in fs.attrs.keys():
                                    fs.attrs["inv_bsvd_from_pinv2"] = [fi]
                                else:
                                    bad_freqs = fs.attrs["inv_bsvd_from_pinv2"]
                                    fs.attrs["inv_bsvd_from_pinv2"] = bad_freqs.append(
                                        fi
                                    )
                            except:
                                # If pinv2 fails, print error message
                                raise Exception(
                                    "Beam-SVD pseudoinverse (scipy.linalg.pinv2) "
                                    "failure: m = %d, fi = %d" % (mi, fi)
                                )

                        dset_ibsvd[fi, :, :, :nmodes] = ibeam.reshape(
                            self.telescope.num_pol_sky,
                            self.telescope.lmax + 1,
                            nmodes,
                        )

                    # Save out the singular values for each block
                    dset_sig[fi, :nmodes] = sig

                # Write a few useful attributes.
                fs.attrs["baselines"] = self.telescope.baselines[bl_mask]
                fs.attrs["baselines2"] = self.telescope.baselines[bl_mask2]
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies

    def _svd_num(self, mi, svcut=None):
        ## Calculate the number of SVD modes meeting the cut for each
        ## frequency, return the number and the array bounds.
        ## In contrast to the method in the base BeamTransfer class,
        ## here we cut modes with *high* SVs.

        if svcut is None:
            svcut = self.svcut

        # Get the array of singular values for each mode
        sv = self.beam_singularvalues(mi)

        # Number of significant SV modes at each frequency
        svnum = (sv < sv.max() * svcut).sum(axis=1)

        # Calculate the block bounds within the full matrix
        svbounds = np.cumsum(np.insert(svnum, 0, 0))

        return svnum, svbounds

    def svd_len(self, ntel=None):
        """The size of the SVD output matrices."""
        if ntel is None:
            ntel = self.ntel
        return min(4 * (self.telescope.lmax + 1), ntel)

    def _collect_svd_spectrum(self):
        """Gather the SVD spectrum into a single file."""

        svd_func = lambda mi: self.beam_singularvalues(mi)

        ntel = 2 * np.sum(
            [
                (x[0] in [0, 1] and x[1] in [0, 1])
                for x in self.telescope.beamclass[self.telescope.uniquepairs]
            ]
        )

        svdspectrum = kltransform.collect_m_array(
            list(range(self.telescope.mmax + 1)),
            svd_func,
            (self.nfreq, self.svd_len(ntel)),
            np.float64,
        )

        if mpiutil.rank0:

            with h5py.File(self.directory + "/svdspectrum.hdf5", "w") as f:

                f.create_dataset("singularvalues", data=svdspectrum)

        mpiutil.barrier()


class BeamTransferSingleStepKLFilterTemplate(beamtransfer.BeamTransfer):
    def __init__(self, directory, **kwargs):

        super(BeamTransferSingleStepKLFilterTemplate, self).__init__(
            directory, **kwargs
        )

        if self.telescope.n_pert != 1:
            raise NotImplementedError(
                "Can only use BeamTransferSingleStepKLFilterTemplate for a single "
                "template!"
            )

    def _generate_svdfile_m(self, mi, skip_svd_inv=False):

        # Define baseline masks for unpert-unpert, unpert-pert, and pert-unpert
        # visibilities
        bl_mask_00 = [
            (x[0] in [0, 1] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask_01 = [
            (x[0] in [0, 1] and x[1] in [2, 3])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask_10 = [
            (x[0] in [2, 3] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]

        npairs = np.sum(bl_mask_00)
        ntel = 2 * npairs

        # Open file to write SVD results into, using caput.misc.lock_file()
        # to guard against crashes while the file is open. With preserve=True,
        # the temp file will be saved with a period in front of its name
        # if a crash occurs.
        with misc.lock_file(self._svdfile(mi), preserve=True) as fs_lock:
            with h5py.File(fs_lock, "w") as fs:

                # Create a chunked dataset for writing the SVD beam matrix into.
                dsize_bsvd = (
                    self.telescope.nfreq,
                    self.svd_len(ntel),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    1,
                    min(10, self.svd_len(ntel)),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                if not skip_svd_inv:
                    # Create a chunked dataset for writing the inverse SVD beam matrix
                    # into
                    dsize_ibsvd = (
                        self.telescope.nfreq,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        self.svd_len(ntel),
                    )
                    csize_ibsvd = (
                        1,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        min(10, self.svd_len(ntel)),
                    )
                    dset_ibsvd = fs.create_dataset(
                        "invbeam_svd",
                        dsize_ibsvd,
                        chunks=csize_ibsvd,
                        compression="lzf",
                        dtype=np.complex128,
                    )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (self.telescope.nfreq, self.svd_len(ntel), ntel)
                csize_ut = (1, min(10, self.svd_len(ntel)), ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (self.telescope.nfreq, self.svd_len(ntel))
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ac = np.zeros(self.telescope.nfreq, dtype=np.float64)

                ## For each frequency in the m file: read in B^0 and B^1 and compute
                ## KL transform
                for fi in np.arange(self.telescope.nfreq):

                    # Read in B^0 and B^1, combining positive and negative m beams
                    bf_shape = (
                        ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1
                    )
                    bf0 = self.beam_m(mi, fi)[:, bl_mask_00, :, :].reshape(bf_shape)
                    bf1 = self.beam_m(mi, fi)[:, bl_mask_01, :, :].reshape(bf_shape)
                    bf1 += self.beam_m(mi, fi)[:, bl_mask_10, :, :].reshape(bf_shape)

                    # Prewhiten B^0 and B^1 by multiplying by N^-1/2 matrix
                    noisew = self.telescope.noisepower(
                        np.arange(self.telescope.npairs)[bl_mask_00], fi
                    ).flatten() ** (-0.5)
                    noisew = np.concatenate([noisew, noisew])
                    bf0 = bf0 * noisew[:, np.newaxis, np.newaxis]
                    bf1 = bf1 * noisew[:, np.newaxis, np.newaxis]

                    # Reshape to 2D matrices
                    bfr0 = bf0.reshape(ntel, -1)
                    bfr1 = bf1.reshape(ntel, -1)

                    # Form B^0 (B^0)^dagger and B^1 (B^1)^dagger
                    bfr0_sq = np.dot(bfr0, bfr0.T.conj())
                    bfr1_sq = np.dot(bfr1, bfr1.T.conj())

                    # Perform KL transform of B^0 (B^0)^dagger and B^1 (B^1)^dagger.
                    # Output evecs are packed as [evec component, evec number], so
                    # a projection that cuts the first n evecs is u1.T.conj()[n:].
                    # (Note that the evals are in ascending order.)
                    sig, u1, ac[fi] = eigh_gen(bfr0_sq, bfr1_sq)

                    ut1 = u1.T.conj()
                    nmodes = ut1.shape[0]

                    sig = sig[:nmodes]
                    beam = np.dot(ut1, bfr0)

                    # We flip the order of the evals to be in descending instead of
                    # ascending order, so that cutting low evals corresponds to cutting
                    # elements from the end of the list.
                    ut1 = ut1[::-1]
                    beam = beam[::-1]
                    sig = sig[::-1]

                    # Save out the evecs (for transforming from the telescope frame
                    # into the KL basis). We multiply ut by N^{-1/2} because ut
                    # must act on N^{-1/2} v, not v alone (where v are the
                    # visibilities), so we include that factor of N^{-1/2} in
                    # dset_ut so that we can apply it directly to v in the future.
                    dset_ut[fi, :nmodes] = ut1 * noisew[np.newaxis, :]

                    # Save out the modified beam matrix (for mapping from the sky
                    # into the KL basis)
                    dset_bsvd[fi, :nmodes] = beam.reshape(
                        nmodes, self.telescope.num_pol_sky, self.telescope.lmax + 1
                    )

                    if not skip_svd_inv and beam.shape[0] > 0:
                        # Find the pseudo-inverse of the beam matrix and save to
                        # disk. First try la.pinv, which uses a least-squares
                        # solver.
                        try:
                            ibeam = la.pinv(beam)
                        except la.LinAlgError as e:
                            # If la.pinv fails, try la.pinv2, which is SVD-based and
                            # more likely to succeed. If successful, add file
                            # attribute
                            # indicating pinv2 was used for this frequency.
                            logger.info(
                                "***Beam-SVD pesudoinverse (scipy.linalg.pinv) "
                                f"failure: m = {mi}, fi = {fi}. Trying pinv2..."
                            )
                            try:
                                ibeam = la.pinv2(beam)
                                if "inv_bsvd_from_pinv2" not in fs.attrs.keys():
                                    fs.attrs["inv_bsvd_from_pinv2"] = [fi]
                                else:
                                    bad_freqs = fs.attrs["inv_bsvd_from_pinv2"]
                                    fs.attrs["inv_bsvd_from_pinv2"] = bad_freqs.append(
                                        fi
                                    )
                            except:
                                # If pinv2 fails, print error message
                                raise Exception(
                                    "Beam-SVD pseudoinverse (scipy.linalg.pinv2) "
                                    "failure: m = %d, fi = %d" % (mi, fi)
                                )

                        dset_ibsvd[fi, :, :, :nmodes] = ibeam.reshape(
                            self.telescope.num_pol_sky,
                            self.telescope.lmax + 1,
                            nmodes,
                        )

                    # Save out the singular values for each block
                    dset_sig[fi, :nmodes] = sig

                # Write a few useful attributes.
                fs.attrs["baselines_00"] = self.telescope.baselines[bl_mask_00]
                fs.attrs["baselines"] = self.telescope.baselines[bl_mask_01]
                fs.attrs["baselines2"] = self.telescope.baselines[bl_mask_10]
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies
                fs.attrs["eigh_gen_add_const"] = ac

    def _svd_num(self, mi, svcut=None):
        ## Calculate the number of KL modes meeting the cut for each
        ## frequency, return the number and the array bounds.
        ## In contrast to the method in the base BeamTransfer class,
        ## we keep modes with sv > svcut ** 2 instead of sv > sv.max() * svcut,
        ## because we care about the absolute (not relative) values of the KL
        ## eigenvalues, and the stored eigenvalues are actually the square of
        ## what we want.

        if svcut is None:
            svcut = self.svcut

        # Get the array of singular values for each mode
        sv = self.beam_singularvalues(mi)

        # Number of significant SV modes at each frequency
        svnum = (sv > svcut ** 2).sum(axis=1)

        # Calculate the block bounds within the full matrix
        svbounds = np.cumsum(np.insert(svnum, 0, 0))

        return svnum, svbounds

    def svd_len(self, ntel=None):
        """The size of the SVD output matrices."""
        if ntel is None:
            ntel = self.ntel
        return min(4 * (self.telescope.lmax + 1), ntel)

    def _collect_svd_spectrum(self):
        """Gather the SVD spectrum into a single file."""

        svd_func = lambda mi: self.beam_singularvalues(mi)

        ntel = 2 * np.sum(
            [
                (x[0] in [0, 1] and x[1] in [0, 1])
                for x in self.telescope.beamclass[self.telescope.uniquepairs]
            ]
        )

        svdspectrum = kltransform.collect_m_array(
            list(range(self.telescope.mmax + 1)),
            svd_func,
            (self.nfreq, self.svd_len(ntel)),
            np.float64,
        )

        if mpiutil.rank0:

            with h5py.File(self.directory + "/svdspectrum.hdf5", "w") as f:

                f.create_dataset("singularvalues", data=svdspectrum)

        mpiutil.barrier()


class BeamTransferDoubleStepKLPolFilterTemplate(beamtransfer.BeamTransfer):
    def __init__(self, directory, **kwargs):

        super(BeamTransferDoubleStepKLPolFilterTemplate, self).__init__(
            directory, **kwargs
        )

        if self.telescope.n_pert != 1:
            raise NotImplementedError(
                "Can only use BeamTransferDoubleStepKLFilterTemplate for a single "
                "template!"
            )

    def _generate_svdfile_m(self, mi, skip_svd_inv=False):

        # Define baseline masks for unpert-unpert, unpert-pert, and pert-unpert
        # visibilities
        bl_mask_00 = [
            (x[0] in [0, 1] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask_01 = [
            (x[0] in [0, 1] and x[1] in [2, 3])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask_10 = [
            (x[0] in [2, 3] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]

        npairs = np.sum(bl_mask_00)
        ntel = 2 * npairs

        # Open file to write SVD results into, using caput.misc.lock_file()
        # to guard against crashes while the file is open. With preserve=True,
        # the temp file will be saved with a period in front of its name
        # if a crash occurs.
        with misc.lock_file(self._svdfile(mi), preserve=True) as fs_lock:
            with h5py.File(fs_lock, "w") as fs:

                # Create a chunked dataset for writing the SVD beam matrix into.
                dsize_bsvd = (
                    self.telescope.nfreq,
                    self.svd_len(ntel),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    1,
                    min(10, self.svd_len(ntel)),
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                if not skip_svd_inv:
                    # Create a chunked dataset for writing the inverse SVD beam matrix
                    # into
                    dsize_ibsvd = (
                        self.telescope.nfreq,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        self.svd_len(ntel),
                    )
                    csize_ibsvd = (
                        1,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        min(10, self.svd_len(ntel)),
                    )
                    dset_ibsvd = fs.create_dataset(
                        "invbeam_svd",
                        dsize_ibsvd,
                        chunks=csize_ibsvd,
                        compression="lzf",
                        dtype=np.complex128,
                    )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (self.telescope.nfreq, self.svd_len(ntel), ntel)
                csize_ut = (1, min(10, self.svd_len(ntel)), ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (self.telescope.nfreq, self.svd_len(ntel))
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ac = np.zeros(self.telescope.nfreq, dtype=np.float64)

                for fi in np.arange(self.telescope.nfreq):

                    ## Transform 1: KL-based B^1 filtering

                    # Read in B^0 and B^1, combining positive and negative m beams
                    bf_shape = (
                        ntel,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1
                    )
                    bf0 = self.beam_m(mi, fi)[:, bl_mask_00, :, :].reshape(bf_shape)
                    bf1 = self.beam_m(mi, fi)[:, bl_mask_01, :, :].reshape(bf_shape)
                    bf1 += self.beam_m(mi, fi)[:, bl_mask_10, :, :].reshape(bf_shape)

                    # Prewhiten B^0 and B^1 by multiplying by N^-1/2 matrix
                    noisew = self.telescope.noisepower(
                        np.arange(self.telescope.npairs)[bl_mask_00], fi
                    ).flatten() ** (-0.5)
                    noisew = np.concatenate([noisew, noisew])
                    bf0 = bf0 * noisew[:, np.newaxis, np.newaxis]
                    bf1 = bf1 * noisew[:, np.newaxis, np.newaxis]

                    # Reshape to 2D matrices
                    bfr0 = bf0.reshape(ntel, -1)
                    bfr1 = bf1.reshape(ntel, -1)

                    # Form B^0 (B^0)^dagger and B^1 (B^1)^dagger
                    bfr0_sq = np.dot(bfr0, bfr0.T.conj())
                    bfr1_sq = np.dot(bfr1, bfr1.T.conj())

                    # Perform KL transform of B^0 (B^0)^dagger and B^1 (B^1)^dagger.
                    # Output evecs are packed as [evec component, evec number], so
                    # a projection that cuts the first n evecs is u1.T.conj()[n:].
                    # (Note that the evals are in ascending order.)
                    sig1, u1, ac[fi] = eigh_gen(bfr0_sq, bfr1_sq)

                    # We flip the order of the evals to be in descending instead of
                    # ascending order, so that cutting low evals corresponds to cutting
                    # elements from the end of the list.
                    ut1 = u1.T.conj()[::-1]
                    sig1 = sig1[::-1]

                    # At this point, we cut some KL modes
                    nmodes = np.sum(sig1 > self.svcut**2)
                    ut1 = ut1[:nmodes]
                    sig1 = sig1[:nmodes]
                    beam1 = np.dot(ut1, bfr0)

                    ## Transform 2: pol-nullspace projection via pol SVD

                    beam1p = beam1.reshape(
                        beam1.shape[0],
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                    )[:, 1:]
                    beam1p = beam1p.reshape(
                        beam1.shape[0],
                        (self.telescope.num_pol_sky - 1) * (self.telescope.lmax + 1),
                    )
                    u2, sig2 = matrix_nullspace(
                        beam1p,
                        rtol=self.polsvcut,
                        errmsg=("Pol SVD m=%i f=%i" % (mi, fi)),
                    )

                    # # We now flip the order of the SVs to be in ascending instead of
                    # # descending order, so that cutting high evals corresponds to cutting
                    # # elements from the end of the list.
                    # ut2 = u2.T.conj()
                    # ut2 = ut2[::-1]
                    # sig2 = sig2[::-1]
                    #
                    # ut2 = np.dot(ut2, ut1)
                    # beam2 = np.dot(ut2, bfr0)

                    ut2 = np.dot(u2.T.conj(), ut1)
                    beam2 = np.dot(ut2, bfr0)

                    # Check to ensure polcut hasn't thrown away all modes. If it
                    # has, just leave datasets blank.
                    if beam2.shape[0] > 0 and (
                        self.telescope.num_pol_sky == 1 or (sig1 > 0.0).any()
                    ):

                        ## SVD 3 - decompose polarisation null space
                        beamt = beam2.reshape(
                            -1, self.telescope.num_pol_sky, self.telescope.lmax + 1
                        )[:, 0]

                        u3, sig3 = matrix_image(
                            beamt, rtol=0.0, errmsg=("SVD3 m=%i f=%i" % (mi, fi))
                        )
                        ut3 = np.dot(u3.T.conj(), ut2)

                        nmodes = ut3.shape[0]

                        # Skip if nmodes is zero for some reason.
                        if nmodes == 0:
                            continue

                        # Final products
                        ut = ut3
                        sig = sig3[:nmodes]
                        beam = np.dot(ut3, bfr0)

                        # Save out the evecs (for transforming from the telescope frame
                        # into the SVD basis). We multiply ut by N^{-1/2} because ut
                        # must act on N^{-1/2} v, not v alone (where v are the
                        # visibilities), so we include that factor of N^{-1/2} in
                        # dset_ut so that we can apply it directly to v in the future.
                        dset_ut[fi, :nmodes] = ut * noisew[np.newaxis, :]

                        # Save out the modified beam matrix (for mapping from the sky
                        # into the SVD basis)
                        dset_bsvd[fi, :nmodes] = beam.reshape(
                            nmodes, self.telescope.num_pol_sky, self.telescope.lmax + 1
                        )

                        if not skip_svd_inv:
                            # Find the pseudo-inverse of the beam matrix and save to
                            # disk. First try la.pinv, which uses a least-squares
                            # solver.
                            try:
                                ibeam = la.pinv(beam)
                            except la.LinAlgError as e:
                                # If la.pinv fails, try la.pinv2, which is SVD-based and
                                # more likely to succeed. If successful, add file
                                # attribute
                                # indicating pinv2 was used for this frequency.
                                logger.info(
                                    "***Beam-SVD pesudoinverse (scipy.linalg.pinv) "
                                    f"failure: m = {mi}, fi = {fi}. Trying pinv2..."
                                )
                                try:
                                    ibeam = la.pinv2(beam)
                                    if "inv_bsvd_from_pinv2" not in fs.attrs.keys():
                                        fs.attrs["inv_bsvd_from_pinv2"] = [fi]
                                    else:
                                        bad_freqs = fs.attrs["inv_bsvd_from_pinv2"]
                                        fs.attrs[
                                            "inv_bsvd_from_pinv2"
                                        ] = bad_freqs.append(fi)
                                except:
                                    # If pinv2 fails, print error message
                                    raise Exception(
                                        "Beam-SVD pseudoinverse (scipy.linalg.pinv2) "
                                        "failure: m = %d, fi = %d" % (mi, fi)
                                    )

                            dset_ibsvd[fi, :, :, :nmodes] = ibeam.reshape(
                                self.telescope.num_pol_sky,
                                self.telescope.lmax + 1,
                                nmodes,
                            )

                        # Save out the singular values for each block
                        dset_sig[fi, :nmodes] = sig

                # Write a few useful attributes.
                fs.attrs["baselines_00"] = self.telescope.baselines[bl_mask_00]
                fs.attrs["baselines"] = self.telescope.baselines[bl_mask_01]
                fs.attrs["baselines2"] = self.telescope.baselines[bl_mask_10]
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies
                fs.attrs["eigh_gen_add_const"] = ac

    def _collect_svd_spectrum(self):
        """Gather the SVD spectrum into a single file."""

        svd_func = lambda mi: self.beam_singularvalues(mi)

        ntel = 2 * np.sum(
            [
                (x[0] in [0, 1] and x[1] in [0, 1])
                for x in self.telescope.beamclass[self.telescope.uniquepairs]
            ]
        )

        svdspectrum = kltransform.collect_m_array(
            list(range(self.telescope.mmax + 1)),
            svd_func,
            (self.nfreq, self.svd_len(ntel)),
            np.float64,
        )

        if mpiutil.rank0:

            with h5py.File(self.directory + "/svdspectrum.hdf5", "w") as f:

                f.create_dataset("singularvalues", data=svdspectrum)

        mpiutil.barrier()


class BeamTransferKLBeamSkyPolFilterTemplate(beamtransfer.BeamTransferFullFreq):

    sky_foreground = config.Property(proptype=bool, default=True)
    sky_signal = config.Property(proptype=bool, default=True)
    matrix_regularizer = config.Property(proptype=float, default=None)

    def __init__(self, directory, **kwargs):

        super(BeamTransferKLBeamSkyPolFilterTemplate, self).__init__(
            directory, **kwargs
        )

        if self.telescope.n_pert != 1:
            raise NotImplementedError(
                "Can only use BeamTransferKLBeamSkyFilterTemplate for a single "
                "template!"
            )

    def _generate_svdfile_m(self, mi, skip_svd_inv=False):

        # Define baseline masks for unpert-unpert, unpert-pert, and pert-unpert
        # visibilities
        bl_mask_00 = [
            (x[0] in [0, 1] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask_01 = [
            (x[0] in [0, 1] and x[1] in [2, 3])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask_10 = [
            (x[0] in [2, 3] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]

        npairs = np.sum(bl_mask_00)
        ntel = 2 * npairs

        # Fetch other telescope information
        nfreq = self.telescope.nfreq
        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1

        # Open file to write SVD results into, using caput.misc.lock_file()
        # to guard against crashes while the file is open. With preserve=True,
        # the temp file will be saved with a period in front of its name
        # if a crash occurs.
        logger.debug(f"m = {mi}: Creating output file")
        with misc.lock_file(self._svdfile(mi), preserve=True) as fs_lock:
            with h5py.File(fs_lock, "w") as fs:

                # Create a chunked dataset for the SVD beam matrix
                dsize_bsvd = (
                    nfreq * self.svd_len(ntel),
                    nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    min(10, self.svd_len(ntel)),
                    1,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                if not skip_svd_inv:
                    # Create a chunked dataset for the inverse SVD beam matrix
                    dsize_ibsvd = (
                        nfreq,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        nfreq * self.svd_len(ntel),
                    )
                    csize_ibsvd = (
                        1,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        min(10, self.svd_len(ntel)),
                    )
                    dset_ibsvd = fs.create_dataset(
                        "invbeam_svd",
                        dsize_ibsvd,
                        chunks=csize_ibsvd,
                        compression="lzf",
                        dtype=np.complex128,
                    )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (nfreq * self.svd_len(ntel), nfreq * ntel)
                csize_ut = (min(10, self.svd_len(ntel)), nfreq * ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (nfreq * self.svd_len(ntel),)
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ## Transform 1: KL-based B^1 filtering

                # Get foreground and signal covariances in sky basis
                logger.debug(f"m = {mi}: Computing sky covariance")
                klt = kltransform.KLTransform(self)
                klt.use_polarised = True
                c_sky = 0
                if self.sky_foreground:
                    c_sky += klt.foreground()
                if self.sky_signal:
                    c_sky += klt.signal()

                # Form B^0 C_sky (B^0)^dagger and B^1 C_sky (B^1)^dagger.
                # These are packed as [freq, tel, freq, tel]
                logger.debug(f"m = {mi}: Projecting covariance to B^0 and B^1 bases")
                c0 = self.project_matrix_sky_to_telescope(mi, c_sky, bl_mask=bl_mask_00)
                c1 = self.project_matrix_sky_to_telescope(
                    mi, c_sky, bl_mask=bl_mask_01, bl_mask2=bl_mask_10
                )

                # If desired, add regularization to matrix diagonals of B^1...
                # to ensure all eigenvalues are positive (required for generalized
                # eigenvalue solver)
                if self.matrix_regularizer is not None:
                    logger.debug(f"m = {mi}: Regularizing C^1 matrix")
                    c1.reshape(nfreq * ntel, nfreq * ntel)[
                        np.diag_indices_from(c1.reshape(nfreq * ntel, nfreq * ntel))
                    ] += (self.matrix_regularizer * c1.real.max())

                # Prewhiten B^0 and B^1 by multiplying by N^-1/2 matrix
                logger.debug(f"m = {mi}: Pre-whitening C^0 and C^1")
                for fi in range(nfreq):
                    noisew_i = self.telescope.noisepower(
                        np.arange(self.telescope.npairs)[bl_mask_00], fi
                    ).flatten() ** (-0.5)
                    noisew_i = np.concatenate([noisew_i, noisew_i])

                    for fj in range(fi, nfreq):
                        noisew_j = self.telescope.noisepower(
                            np.arange(self.telescope.npairs)[bl_mask_00], fj
                        ).flatten() ** (-0.5)
                        noisew_j = np.concatenate([noisew_j, noisew_j])

                        c0[fi, :, fj, :] *= (
                            noisew_i[:, np.newaxis] * noisew_j[np.newaxis, :]
                        )
                        c1[fi, :, fj, :] *= (
                            noisew_i[:, np.newaxis] * noisew_j[np.newaxis, :]
                        )

                # Perform KL transform of C^0 and C^1.
                # Output evecs are packed as [evec component, evec number], so
                # a projection that cuts the first n evecs is u1.T.conj()[n:].
                # (Note that the evals are in ascending order.)
                logger.debug(f"m = {mi}: Doing C^0 / C^1 KL transform")
                sig1, u1, ac = eigh_gen(
                    c0.reshape(nfreq * ntel, nfreq * ntel),
                    c1.reshape(nfreq * ntel, nfreq * ntel),
                )
                del c0, c1

                # We flip the order of the evals to be in descending instead of
                # ascending order, so that cutting low evals corresponds to cutting
                # elements from the end of the list.
                ut1 = u1.T.conj()[::-1]
                sig1 = sig1[::-1]

                # At this point, we cut some KL modes
                nmodes = np.sum(sig1 > self.svcut)
                ut1 = ut1[:nmodes]
                sig1 = sig1[:nmodes]

                # Now, load B^0, packed as [freq,msign,base,pol,ell], and reshape to
                # [freq, msign*base, pol, ell]
                logger.debug(f"m = {mi}: Loading and prewhitening B^0")
                b0_diag = self.beam_m(mi, fi=None)[:, :, bl_mask_00, :, :].reshape(
                    nfreq, ntel, npol, lside
                )
                b_diag_shape = b0_diag.shape

                # Prewhiten B^0 by multiplying by N^-1/2 matrix
                for fi in range(nfreq):
                    noisew_i = self.telescope.noisepower(
                        np.arange(self.telescope.npairs)[bl_mask_00], fi
                    ).flatten() ** (-0.5)
                    noisew_i = np.concatenate([noisew_i, noisew_i])
                    b0_diag[fi] *= noisew_i[:, np.newaxis, np.newaxis]

                # Expand B^0 to 2 freq axes, packed as [freq,msign*base,freq,pol,ell]
                logger.debug(f"m = {mi}. : Expanding B^0 to 2 freq axes")
                b_full_shape = b_diag_shape[:2] + (
                    b_diag_shape[0],
                    b_diag_shape[2],
                    b_diag_shape[3],
                )
                b0_full = np.zeros(b_full_shape, dtype=np.complex128)
                for i in range(nfreq):
                    b0_full[i, :, i, :, :] = b0_diag[i]
                del b0_diag

                # Reshape B^0 to [freq*tel,freq*pol*ell], project into
                # high-B^0/B^1 basis, and reshape to [mode, freq, pol, ell]
                logger.debug(f"m = {mi}: Projecting B^0 into KL basis")
                b0_full = b0_full.reshape(nfreq * ntel, nfreq * npol * lside)
                beam1 = np.dot(ut1, b0_full)
                beam1 = beam1.reshape(beam1.shape[0], nfreq, npol, lside)

                ## Transform 2: pol-nullspace projection via pol SVD

                beam1p = beam1[:, :, 1:].reshape(
                    beam1.shape[0], nfreq * (npol - 1) * lside
                )
                logger.debug(
                    f"m = {mi}. : Performing SVD 2 - matrix shape = {beam1p.shape}"
                )
                u2, sig2 = matrix_nullspace(
                    beam1p,
                    rtol=self.polsvcut,
                    errmsg=f"Pol SVD m={mi}",
                )
                # del beam1, beam1p

                # print('pre', ut1.shape, u2.shape, sig2)

                ut2 = np.dot(u2.T.conj(), ut1)
                beam2 = np.dot(ut2, b0_full)

                # Check to ensure polcut hasn't thrown away all modes. If it
                # has, just leave datasets blank.
                if beam2.shape[0] > 0 and (npol == 1 or (sig1 > 0.0).any()):

                    ## SVD 3 - decompose polarisation null space
                    beamt = beam2.reshape(-1, nfreq, npol, lside)[:, :, 0].reshape(
                        -1, nfreq * lside
                    )
                    del beam2

                    logger.debug(
                        f"m = {mi} : Performing SVD 3 - matrix shape = {beamt.shape}"
                    )
                    u3, sig3 = matrix_image(beamt, rtol=0.0, errmsg=f"SVD3 m={mi}")
                    del beamt
                    ut3 = np.dot(u3.T.conj(), ut2)
                    del ut2

                    nmodes = ut3.shape[0]

                    # Final products
                    ut = ut3
                    sig = sig3[:nmodes]
                    beam = np.dot(ut3, b0_full)

                    # Save out the evecs (for transforming from the telescope frame
                    # into the SVD basis). We multiply ut by N^{-1/2} because ut
                    # must act on N^{-1/2} v, not v alone (where v are the
                    # visibilities), so we include that factor of N^{-1/2} in
                    # dset_ut so that we can apply it directly to v in the future.
                    logger.debug(
                        f"m = {mi}: Applying prewhitening to U^dagger and saving to"
                        " disk"
                    )
                    dset_ut[:nmodes] = self._apply_prewhitening_to_beam_ut(
                        ut, npairs, ntel
                    )

                    # Save out the modified beam matrix (for mapping from the sky
                    # into the SVD basis)
                    logger.debug(
                        f"m = {mi}: Saving projected BTM and SV spectrum to disk"
                    )
                    dset_bsvd[:nmodes] = beam.reshape(nmodes, nfreq, npol, lside)

                    # Save the singular values
                    dset_sig[:nmodes] = sig

                    if not skip_svd_inv:
                        # Find the pseudo-inverse of the beam matrix and save to
                        # disk. First try la.pinv, which uses a least-squares
                        # solver.
                        logger.debug(
                            f"m = {mi}: Finding pseudoinverse of projected "
                            f"beam - projected shape = {beam.shape}"
                        )
                        try:
                            ibeam = la.pinv(beam)
                        except la.LinAlgError as e:
                            # If la.pinv fails, try la.pinv2, which is SVD-based and
                            # more likely to succeed. If successful, add file attribute
                            # indicating pinv2 was used.
                            logger.info(
                                "***Beam-SVD pesudoinverse (scipy.linalg.pinv) "
                                f"failure: m = {mi}. Trying pinv2..."
                            )
                            try:
                                ibeam = la.pinv2(beam)
                                fs.attrs["inv_bsvd_from_pinv2"] = True
                            except:
                                # If pinv2 fails, print error message
                                raise Exception(
                                    "Beam-SVD pseudoinverse (scipy.linalg.pinv2) "
                                    f"failure: m = {mi}"
                                )

                        # Save pseudo-inverse of the beam matrix to disk.
                        dset_ibsvd[:, :, :, :nmodes] = ibeam.reshape(
                            nfreq, npol, lside, nmodes
                        )

                # Write a few useful attributes.
                fs.attrs["baselines_00"] = self.telescope.baselines[bl_mask_00]
                fs.attrs["baselines"] = self.telescope.baselines[bl_mask_01]
                fs.attrs["baselines2"] = self.telescope.baselines[bl_mask_10]
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies
                fs.attrs["eigh_gen_add_const"] = ac

    def _collect_svd_spectrum(self):
        """Gather the SVD spectrum into a single file."""

        svd_func = lambda mi: self.beam_singularvalues(mi)

        ntel = 2 * np.sum(
            [
                (x[0] in [0, 1] and x[1] in [0, 1])
                for x in self.telescope.beamclass[self.telescope.uniquepairs]
            ]
        )

        svdspectrum = kltransform.collect_m_array(
            list(range(self.telescope.mmax + 1)),
            svd_func,
            (self.nfreq * self.svd_len(ntel),),
            np.float64,
        )

        if mpiutil.rank0:

            with h5py.File(self.directory + "/svdspectrum.hdf5", "w") as f:

                f.create_dataset("singularvalues", data=svdspectrum)

        mpiutil.barrier()


class BeamTransferSingleStepKLBeamSkyFilterTemplate(beamtransfer.BeamTransferFullFreq):

    sky_foreground = config.Property(proptype=bool, default=True)
    sky_signal = config.Property(proptype=bool, default=True)
    matrix_regularizer = config.Property(proptype=float, default=None)

    def __init__(self, directory, **kwargs):

        super(BeamTransferSingleStepKLBeamSkyFilterTemplate, self).__init__(
            directory, **kwargs
        )

        if self.telescope.n_pert != 1:
            raise NotImplementedError(
                "Can only use BeamTransferKLBeamSkyFilterTemplate for a single "
                "template!"
            )

    def _generate_svdfile_m(self, mi, skip_svd_inv=False):

        # Define baseline masks for unpert-unpert, unpert-pert, and pert-unpert
        # visibilities
        bl_mask_00 = [
            (x[0] in [0, 1] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask_01 = [
            (x[0] in [0, 1] and x[1] in [2, 3])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]
        bl_mask_10 = [
            (x[0] in [2, 3] and x[1] in [0, 1])
            for x in self.telescope.beamclass[self.telescope.uniquepairs]
        ]

        npairs = np.sum(bl_mask_00)
        ntel = 2 * npairs

        # Fetch other telescope information
        nfreq = self.telescope.nfreq
        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1

        # Open file to write SVD results into, using caput.misc.lock_file()
        # to guard against crashes while the file is open. With preserve=True,
        # the temp file will be saved with a period in front of its name
        # if a crash occurs.
        logger.debug(f"m = {mi}: Creating output file")
        with misc.lock_file(self._svdfile(mi), preserve=True) as fs_lock:
            with h5py.File(fs_lock, "w") as fs:

                # Create a chunked dataset for the SVD beam matrix
                dsize_bsvd = (
                    nfreq * self.svd_len(ntel),
                    nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                csize_bsvd = (
                    min(10, self.svd_len(ntel)),
                    1,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                dset_bsvd = fs.create_dataset(
                    "beam_svd",
                    dsize_bsvd,
                    chunks=csize_bsvd,
                    compression="lzf",
                    dtype=np.complex128,
                )

                if not skip_svd_inv:
                    # Create a chunked dataset for the inverse SVD beam matrix
                    dsize_ibsvd = (
                        nfreq,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        nfreq * self.svd_len(ntel),
                    )
                    csize_ibsvd = (
                        1,
                        self.telescope.num_pol_sky,
                        self.telescope.lmax + 1,
                        min(10, self.svd_len(ntel)),
                    )
                    dset_ibsvd = fs.create_dataset(
                        "invbeam_svd",
                        dsize_ibsvd,
                        chunks=csize_ibsvd,
                        compression="lzf",
                        dtype=np.complex128,
                    )

                # Create a chunked dataset for the stokes T U-matrix (left evecs)
                dsize_ut = (nfreq * self.svd_len(ntel), nfreq * ntel)
                csize_ut = (min(10, self.svd_len(ntel)), nfreq * ntel)
                dset_ut = fs.create_dataset(
                    "beam_ut",
                    dsize_ut,
                    chunks=csize_ut,
                    compression="lzf",
                    dtype=np.complex128,
                )

                # Create a dataset for the singular values.
                dsize_sig = (nfreq * self.svd_len(ntel),)
                dset_sig = fs.create_dataset(
                    "singularvalues", dsize_sig, dtype=np.float64
                )

                ## Transform 1: KL-based B^1 filtering

                # Get foreground and signal covariances in sky basis
                logger.debug(f"m = {mi}: Computing sky covariance")
                klt = kltransform.KLTransform(self)
                klt.use_polarised = True
                c_sky = 0
                if self.sky_foreground:
                    c_sky += klt.foreground()
                if self.sky_signal:
                    c_sky += klt.signal()

                # Form B^0 C_sky (B^0)^dagger and B^1 C_sky (B^1)^dagger.
                # These are packed as [freq, tel, freq, tel]
                logger.debug(f"m = {mi}: Projecting covariance to B^0 and B^1 bases")
                c0 = self.project_matrix_sky_to_telescope(mi, c_sky, bl_mask=bl_mask_00)
                c1 = self.project_matrix_sky_to_telescope(
                    mi, c_sky, bl_mask=bl_mask_01, bl_mask2=bl_mask_10
                )

                # If desired, add regularization to matrix diagonals of B^1...
                # to ensure all eigenvalues are positive (required for generalized
                # eigenvalue solver)
                if self.matrix_regularizer is not None:
                    c1.reshape(nfreq * ntel, nfreq * ntel)[
                        np.diag_indices_from(c1.reshape(nfreq * ntel, nfreq * ntel))
                    ] += (self.matrix_regularizer * c1.real.max())

                # Prewhiten B^0 and B^1 by multiplying by N^-1/2 matrix
                for fi in range(nfreq):
                    noisew_i = self.telescope.noisepower(
                        np.arange(self.telescope.npairs)[bl_mask_00], fi
                    ).flatten() ** (-0.5)
                    noisew_i = np.concatenate([noisew_i, noisew_i])

                    for fj in range(fi, nfreq):
                        noisew_j = self.telescope.noisepower(
                            np.arange(self.telescope.npairs)[bl_mask_00], fj
                        ).flatten() ** (-0.5)
                        noisew_j = np.concatenate([noisew_j, noisew_j])

                        c0[fi, :, fj, :] *= (
                            noisew_i[:, np.newaxis] * noisew_j[np.newaxis, :]
                        )
                        c1[fi, :, fj, :] *= (
                            noisew_i[:, np.newaxis] * noisew_j[np.newaxis, :]
                        )

                # Perform KL transform of C^0 and C^1.
                # Output evecs are packed as [evec component, evec number], so
                # a projection that cuts the first n evecs is u1.T.conj()[n:].
                # (Note that the evals are in ascending order.)
                logger.debug(f"m = {mi}: Doing C^0 / C^1 KL transform")
                sig1, u1, ac = eigh_gen(
                    c0.reshape(nfreq * ntel, nfreq * ntel),
                    c1.reshape(nfreq * ntel, nfreq * ntel),
                )
                del c0, c1

                # We flip the order of the evals to be in descending instead of
                # ascending order, so that cutting low evals corresponds to cutting
                # elements from the end of the list.
                ut1 = u1.T.conj()[::-1]
                sig1 = sig1[::-1]
                nmodes = ut1.shape[0]

                # Now, load B^0, packed as [freq,msign,base,pol,ell], and reshape to
                # [freq, msign*base, pol, ell]
                logger.debug(f"m = {mi}: Loading and prewhitening B^0")
                b0_diag = self.beam_m(mi, fi=None)[:, :, bl_mask_00, :, :].reshape(
                    nfreq, ntel, npol, lside
                )
                b_diag_shape = b0_diag.shape

                # Prewhiten B^0 by multiplying by N^-1/2 matrix
                for fi in range(nfreq):
                    noisew_i = self.telescope.noisepower(
                        np.arange(self.telescope.npairs)[bl_mask_00], fi
                    ).flatten() ** (-0.5)
                    noisew_i = np.concatenate([noisew_i, noisew_i])
                    b0_diag[fi] *= noisew_i[:, np.newaxis, np.newaxis]

                # Expand B^0 to 2 freq axes, packed as [freq,msign*base,freq,pol,ell]
                logger.debug(f"m = {mi}: Expanding B^0 to 2 freq axes")
                b_full_shape = b_diag_shape[:2] + (
                    b_diag_shape[0],
                    b_diag_shape[2],
                    b_diag_shape[3],
                )
                b0_full = np.zeros(b_full_shape, dtype=np.complex128)
                for i in range(nfreq):
                    b0_full[i, :, i, :, :] = b0_diag[i]
                del b0_diag

                # Reshape B^0 to [freq*tel,freq*pol*ell], project into
                # high-B^0/B^1 basis, and reshape to [mode, freq, pol, ell]
                logger.debug(f"m = {mi}: Projecting B^0 into KL basis")
                b0_full = b0_full.reshape(nfreq * ntel, nfreq * npol * lside)
                beam1 = np.dot(ut1, b0_full)
                beam1 = beam1.reshape(beam1.shape[0], nfreq, npol, lside)

                # Save out the evecs (for transforming from the telescope frame
                # into the SVD basis). We multiply ut by N^{-1/2} because ut
                # must act on N^{-1/2} v, not v alone (where v are the
                # visibilities), so we include that factor of N^{-1/2} in
                # dset_ut so that we can apply it directly to v in the future.
                logger.debug(
                    f"m = {mi}: Applying prewhitening to U^dagger and saving to disk"
                )
                dset_ut[:nmodes] = self._apply_prewhitening_to_beam_ut(
                    ut1, npairs, ntel
                )

                # Save out the modified beam matrix (for mapping from the sky
                # into the SVD basis)
                logger.debug(
                    f"m = {mi}: Saving projected BTM and KL spectrum to disk"
                )
                dset_bsvd[:nmodes] = beam1.reshape(nmodes, nfreq, npol, lside)

                # Save the singular values
                dset_sig[:nmodes] = sig1

                if not skip_svd_inv:
                    # Find the pseudo-inverse of the beam matrix and save to
                    # disk. First try la.pinv, which uses a least-squares
                    # solver.
                    logger.debug(
                        f"m = {mi}: Finding pseudoinverse of projected "
                        f"beam - projected shape = {beam1.shape}"
                    )
                    try:
                        ibeam = la.pinv(beam1.reshape(beam1.shape[0], -1))
                    except la.LinAlgError as e:
                        # If la.pinv fails, try la.pinv2, which is SVD-based and
                        # more likely to succeed. If successful, add file attribute
                        # indicating pinv2 was used.
                        logger.info(
                            "***Beam-SVD pesudoinverse (scipy.linalg.pinv) "
                            f"failure: m = {mi}. Trying pinv2..."
                        )
                        try:
                            ibeam = la.pinv2(beam1)
                            fs.attrs["inv_bsvd_from_pinv2"] = True
                        except:
                            # If pinv2 fails, print error message
                            raise Exception(
                                "Beam-SVD pseudoinverse (scipy.linalg.pinv2) "
                                f"failure: m = {mi}"
                            )

                    # Save pseudo-inverse of the beam matrix to disk.
                    dset_ibsvd[:, :, :, :nmodes] = ibeam.reshape(
                        nfreq, npol, lside, nmodes
                    )

                # Write a few useful attributes.
                fs.attrs["baselines_00"] = self.telescope.baselines[bl_mask_00]
                fs.attrs["baselines"] = self.telescope.baselines[bl_mask_01]
                fs.attrs["baselines2"] = self.telescope.baselines[bl_mask_10]
                fs.attrs["m"] = mi
                fs.attrs["frequencies"] = self.telescope.frequencies
                fs.attrs["eigh_gen_add_const"] = ac

    def _collect_svd_spectrum(self):
        """Gather the SVD spectrum into a single file."""

        svd_func = lambda mi: self.beam_singularvalues(mi)

        ntel = 2 * np.sum(
            [
                (x[0] in [0, 1] and x[1] in [0, 1])
                for x in self.telescope.beamclass[self.telescope.uniquepairs]
            ]
        )

        svdspectrum = kltransform.collect_m_array(
            list(range(self.telescope.mmax + 1)),
            svd_func,
            (self.nfreq * self.svd_len(ntel),),
            np.float64,
        )

        if mpiutil.rank0:

            with h5py.File(self.directory + "/svdspectrum.hdf5", "w") as f:

                f.create_dataset("singularvalues", data=svdspectrum)

        mpiutil.barrier()


class BeamTransferSeparateSVD(beamtransfer.BeamTransfer):

    svd_products_dir = config.Property(proptype=str)

    def _svdfile(self, mi):
        # Pattern to form the `m` ordered file.
        pat = (
            self.svd_products_dir
            + "/bt/beam_m/"
            + util.natpattern(self.telescope.mmax)
            + "/svd.hdf5"
        )

        return pat % mi

    def generate(self, regen=False, skip_svd=True, skip_svd_inv=True):
        super(BeamTransferSeparateSVD, self).generate(
            regen=regen, skip_svd=True, skip_svd_inv=True
        )

    def _generate_svdfiles(self, regen=False, skip_svd_inv=False):
        return

    def svd_all(self):
        f = h5py.File(self.svd_products_dir + "/bt/svdspectrum.hdf5", "r")
        svd = f["singularvalues"][:]
        f.close()

        return svd


class BeamTransferFullFreqSeparateSVD(
    BeamTransferSeparateSVD, beamtransfer.BeamTransferFullFreq
):
    """Use separate telescope-SVD computed with full-freq SVD."""
