# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import time
import os

import numpy as np
import h5py

from caput import mpiutil, config

from drift.core import kltransform


class DoubleKL(kltransform.KLTransform):
    """Modified KL technique that performs a first transformation to remove
    foreground modes, and a subsequent transformation to diagonalise the full
    noise (remaining foregrounds+instrumental space).

    Attributes
    ----------
    foreground_threshold : scalar
        Ratio of S/F power below which we throw away modes as being foreground
        contaminated.
    foreground_mode_cut : int, optional
        If specified, overrides foreground_threshold and simply cuts the N modes
        with highest F/S ratio, where N=foreground_mode_cut. Default: None
    foreground_ev_compute_fraction: float, optional
        If specified, we only compute the highest N S/F or lowest N F/S
        modes in step 1 of the KL transform, where
            N = foreground_ev_compute_fraction * evs_total ,
        for computational efficiency. We make sure that these N modes cover
        the desired foreground threshold range, and if not, we solve the full
        eigenvalue problem. Default: None
    """

    foreground_threshold = config.Property(proptype=float, default=100.0)
    foreground_mode_cut = config.Property(proptype=int, default=None)
    foreground_ev_compute_fraction = config.Property(proptype=float, default=None)

    def _transform_m(self, mi):

        # print("m = %d: Solving for double-KL eigenvalues...." % mi)

        # Start timer
        st = time.time()
        nside = self.beamtransfer.ndof(mi)

        inv = None

        # Ensure that number of SVD degrees of freedom is non-zero before proceeding
        if nside == 0:
            return (
                np.array([]),
                np.array([[]]),
                np.array([[]]),
                {"ac": 0.0, "f_evals": np.array([])},
            )

        # Construct S and F matrices and regularise foregrounds
        self.use_thermal = False
        cs, cn = self.sn_covariance(mi)
        if self.external_svd_basis_dir is None:
            cs = cs.reshape(nside, nside)
            cn = cn.reshape(nside, nside)
        et = time.time()
        print("m = %d: Time to generate S,F covariances =\t\t" % mi, (et - st))

        # If desired, compute traces of S and N covariances, so we can
        # save them later
        if self.save_cov_traces:
            s_trace = np.trace(cs)
            n_trace = np.trace(cn)

        # If we want to restrict our computation to the highest/lowest eigenvectors,
        # translate the desired fraction into indices
        compute_indices = None
        if self.foreground_ev_compute_fraction is not None:
            max_evs = cs.shape[0]
            ev_frac = int(self.foreground_ev_compute_fraction * max_evs)
            if not self.do_NoverS:
                # If doing S/N, we want the highest values
                compute_indices = (max_evs-1-ev_frac, max_evs-1)
            else:
                # If doing N/S, we want the lowest values
                compute_indices = (0, ev_frac)

        # Find joint eigenbasis and transformation matrix
        st = time.time()
        if not self.do_NoverS:
            evals, evecs2, ac = kltransform.eigh_gen(
                cs, cn, message="m = %d; KL step 1" % mi, eigvals=compute_indices
            )
            sf_str = "S/F"

            if compute_indices is not None:
                print("m = %d: Step 1 threshold = %g, min eval computed = %g" \
                        % (mi, self.foreground_threshold, evals[0]))

                if evals[0] >= self.foreground_threshold:
                    print("**** m = %d: Attempted to cut too many KL modes in S/F computation!" % mi)
                    print("**** m = %d: Re-running full S/F transform..." % mi)
                    evals, evecs2, ac = kltransform.eigh_gen(
                        cs, cn, message="m = %d; KL step 1" % mi, eigvals=None
                    )

        else:
            evals, evecs, ac = kltransform.eigh_gen(
                cn, cs, message="m = %d; KL step 1" % mi, eigvals=compute_indices
            )
            evals = 1./evals[::-1]
            evecs2 = evecs[:, ::-1]
            sf_str = "F/S"

            if compute_indices is not None:
                print("**** m = %d: Threshold = %g, min equivalent S/F eval computed = %g" \
                        % (mi, self.foreground_threshold, 1/evals[-1]))

                if 1/evals[-1] >= self.foreground_threshold:
                    print("**** m = %d: Attempted to cut too many KL modes in F/S computation!" % mi)
                    print("**** m = %d: Re-running full F/S transform..." % mi)
                    evals, evecs, ac = kltransform.eigh_gen(
                        cn, cs, message="m = %d; KL step 1" % mi, eigvals=None
                    )
                    evals = 1./evals[::-1]
                    evecs2 = evecs[:, ::-1]

        evecs = evecs2.T.conj()
        et = time.time()
        print("m = %d: Time to solve generalized %s EV problem =\t" % (mi, sf_str), (et - st))

        # Get the indices that extract the high-S/F ratio modes.
        # This is subtle in the case where the F/S transform has been done
        # instead, since there can be large negative S/F values that actually
        # correspond to modes we want to keep. To deal with this, we detect
        # whether there are negative eigenvalues at the end of the array
        # (since evals should be in ascending order), and manually include
        # those elements if so.

        ind = self._eval_indices_retained(evals, self.foreground_threshold)
        if self.foreground_mode_cut is not None:
            ind = np.arange(-self.foreground_mode_cut, 1)

        # Construct dictionary of extra parameters to return.
        # Includes regularization constant if KL transform failed on first
        # attempt, and flag indicating that N/S transform was performed.
        # Also includes traces of covariance matrices if desired.
        evextra = {"ac": ac, "sf_evals": evals.copy()}
        if self.do_NoverS:
            evextra["NoverS"] = True
        if self.save_cov_traces:
            evextra["Strace"] = s_trace
            evextra["Ntrace"] = n_trace

        # Construct inverse transformation if required
        if self.inverse:
            inv = kltransform.inv_gen(evecs).T
            # TODO: for external SVD filtering, also need to add elements
            # to inverse evecs matrix to account for tel-SVD modes that were
            # omitted

        # If we've used an external SVD basis defined on tel-SVD modes,
        #  the number of tel-SVD modes
        # assumed by the KL eigenvectors will not match the true number of tel-SVD
        # modes. To make the KL eigenvectors compatible with the original tel-SVD
        # basis, we need to add zero elements to those tel-SVD modes into the
        # KL vectors.
        if self.external_svd_basis_dir is not None and not self.external_sv_from_m_modes:
            evecs = self._reshape_evecs_for_ext_svd(mi, evecs)

        # Construct the foreground-removed subset of the space
        evals = evals[ind]
        evecs = evecs[ind]
        inv = inv[ind] if self.inverse else None

        if evals.size > 0:
            # Generate the full S and F+N covariances in the truncated basis
            # st = time.time()
            # self.use_thermal = True
            # cs, cn = self.sn_covariance(mi)
            # if self.external_svd_basis_dir is None:
            #     cs = cs.reshape(nside, nside)
            #     cn = cn.reshape(nside, nside)
            # et = time.time()
            # print("m = %d: Time to generate S,F+N covariances =\t\t" % mi, (et - st))
            st = time.time()
            cn_thermal = self.thermalnoise_covariance(mi)
            if self.external_svd_basis_dir is None:
                cn_thermal = cn_thermal.reshape(nside, nside)
            et = time.time()
            print("m = %d: Time to generate thermal noise covariance =\t" % mi, (et - st))

            cs = np.dot(evecs, np.dot(cs, evecs.T.conj()))
            cn = np.dot(evecs, np.dot(cn + cn_thermal, evecs.T.conj()))

            # Find the eigenbasis and the transformation into it.
            st = time.time()
            evals, evecs2, ac = kltransform.eigh_gen(cs, cn, message="m = %d; KL step 2" % mi)
            evecs = np.dot(evecs2.T.conj(), evecs)
            et = time.time()
            print("m = %d: Time to solve generalized S/(F+N) EV problem =\t" % mi, (et - st))

            # Construct the inverse if required.
            if self.inverse:
                inv2 = kltransform.inv_gen(evecs2)
                inv = np.dot(inv2, inv)
                # TODO: for external SVD filtering, also need to add elements
                # to inverse evecs matrix to account for tel-SVD modes that were
                # omitted

            # If we've used an external SVD basis, the number of tel-SVD modes
            # assumed by the KL eigenvectors will not match the true number of tel-SVD
            # modes. To make the KL eigenvectors compatible with the original tel-SVD
            # basis, we need to add zero elements to those tel-SVD modes into the
            # KL vectors.
            if self.external_svd_basis_dir is not None and not self.external_sv_from_m_modes:
                evecs = self._reshape_evecs_for_ext_svd(mi, evecs)

        return evals, evecs, inv, evextra

    def _ev_save_hook(self, f, evextra):

        kltransform.KLTransform._ev_save_hook(self, f, evextra)

        # Save out S/F ratios
        f.create_dataset("sf_evals", data=evextra["sf_evals"])

        # If N/S flag exists, write it to file
        if "NoverS" in evextra.keys():
            f.attrs["NoverS"] = "True"

        # If desired, save traces of S and F covariances
        if "Strace" in evextra.keys():
            f.attrs["Strace"] = evextra["Strace"]
        if "Ntrace" in evextra.keys():
            f.attrs["Ntrace"] = evextra["Ntrace"]

    def _collect(self):
        def evfunc(mi):

            ta = np.zeros(shape, dtype=np.float64)

            f = h5py.File(self._evfile % mi, "r")

            if f["evals_full"].shape[0] > 0:
                ev = f["evals_full"][:]
                fev = f["sf_evals"][:]
                ta[0, -ev.size :] = ev
                ta[1, -fev.size :] = fev

            f.close()

            return ta

        if mpiutil.rank0:
            print("Creating eigenvalues file (process 0 only).")

        mlist = list(range(self.telescope.mmax + 1))
        shape = (2, self.beamtransfer.ndofmax)

        evarray = kltransform.collect_m_array(mlist, evfunc, shape, np.float64)

        if mpiutil.rank0:
            if os.path.exists(self.evdir + "/evals.hdf5"):
                print("File: %s exists. Skipping..." % (self.evdir + "/evals.hdf5"))
                return

            f = h5py.File(self.evdir + "/evals.hdf5", "w")
            f.create_dataset("evals", data=evarray[:, 0])
            f.create_dataset("sf_evals", data=evarray[:, 1])
            f.close()
