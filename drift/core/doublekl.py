import logging
import os

import numpy as np
import h5py

from caput import mpiutil, config

from drift.core import kltransform


# Get logger for module
logger = logging.getLogger(__name__)


class DoubleKL(kltransform.KLTransform):
    """Modified KL technique that performs a first transformation to remove
    foreground modes, and a subsequent transformation to diagonalise the full
    noise (remaining foregrounds+instrumental space).

    Attributes
    ----------
    foreground_threshold : scalar
        Ratio of S/F power below which we throw away modes as being foreground
        contaminated.
    """

    foreground_threshold = config.Property(proptype=float, default=100.0)

    def _transform_m(self, mi):
        inv = None

        nside = self.beamtransfer.ndof(mi)

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
        cs, cn = [cv.reshape(nside, nside) for cv in self.sn_covariance(mi)]

        # Find joint eigenbasis and transformation matrix
        evals, evecs2, ac = kltransform.eigh_gen(
            cs, cn, message="m = %d; KL step 1" % mi
        )
        evecs = evecs2.T.conj()

        # Get the indices that extract the high S/F ratio modes
        ind = np.where(evals > self.foreground_threshold)

        # Construct evextra dictionary (holding foreground ratio)
        evextra = {"ac": ac, "f_evals": evals.copy()}

        # Construct inverse transformation if required
        if self.inverse:
            inv = kltransform.inv_gen(evecs).T

        # Construct the foreground removed subset of the space
        evals = evals[ind]
        evecs = evecs[ind]
        inv = inv[ind] if self.inverse else None

        if evals.size > 0:
            # Generate the full S and N covariances in the truncated basis
            self.use_thermal = True
            cs, cn = [cv.reshape(nside, nside) for cv in self.sn_covariance(mi)]
            cs = np.dot(evecs, np.dot(cs, evecs.T.conj()))
            cn = np.dot(evecs, np.dot(cn, evecs.T.conj()))

            # Find the eigenbasis and the transformation into it.
            evals, evecs2, ac = kltransform.eigh_gen(
                cs, cn, message="m = %d; KL step 2" % mi
            )
            evecs = np.dot(evecs2.T.conj(), evecs)

            # Construct the inverse if required.
            if self.inverse:
                inv2 = kltransform.inv_gen(evecs2)
                inv = np.dot(inv2, inv)

        return evals, evecs, inv, evextra

    def _ev_save_hook(self, f, evextra):
        kltransform.KLTransform._ev_save_hook(self, f, evextra)

        # Save out S/F ratios
        f.create_dataset("f_evals", data=evextra["f_evals"])

    def _collect(self):
        def evfunc(mi):
            ta = np.zeros(shape, dtype=np.float64)

            f = h5py.File(self._evfile % mi, "r")

            if f["evals_full"].shape[0] > 0:
                ev = f["evals_full"][:]
                fev = f["f_evals"][:]
                ta[0, -ev.size :] = ev
                ta[1, -fev.size :] = fev

            f.close()

            return ta

        if mpiutil.rank0:
            logger.info("Creating eigenvalues file (process 0 only).")

        mlist = list(range(self.telescope.mmax + 1))
        shape = (2, self.beamtransfer.ndofmax)

        evarray = kltransform.collect_m_array(mlist, evfunc, shape, np.float64)

        if mpiutil.rank0:
            fname = self.evdir + "/evals.hdf5"
            if os.path.exists(fname):
                logger.info("File: {fname} exists. Skipping...")
                return

            f = h5py.File(fname, "w")
            f.create_dataset("evals", data=evarray[:, 0])
            f.create_dataset("f_evals", data=evarray[:, 1])
            f.close()
