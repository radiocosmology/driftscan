import pickle
import os

import h5py
import numpy as np

from caput import mpiutil

from cora.util import hputil

from drift.core import kltransform
from drift.util import util


class Timestream(object):
    directory = None
    output_directory = None
    beamtransfer_dir = None

    no_m_zero = True

    # ============ Constructor etc. =====================

    def __init__(self, tsdir, prodmanager):
        """Create a new Timestream object.

        Parameters
        ----------
        tsdir : string
            Directory to create the Timestream in.
        prodmanager : drift.core.manager.ProductManager
            ProductManager object containing the analysis products.
        """
        self.directory = os.path.abspath(tsdir)
        self.output_directory = self.directory
        self.manager = prodmanager

    # ====================================================

    # ===== Accessing the BeamTransfer and Telescope =====

    _beamtransfer = None

    @property
    def beamtransfer(self):
        """The BeamTransfer object corresponding to this timestream."""
        # if self._beamtransfer is None:
        #     self._beamtransfer = beamtransfer.BeamTransfer(self.beamtransfer_dir)

        # return self._beamtransfer

        return self.manager.beamtransfer

    @property
    def telescope(self):
        """The telescope object corresponding to this timestream."""
        return self.beamtransfer.telescope

    # ====================================================

    # ======== Fetch and generate the f-stream ===========

    def _fdir(self, fi):
        # Pattern to form the `freq` ordered file.
        pat = self.directory + "/timestream_f/" + util.natpattern(self.telescope.nfreq)
        return pat % fi

    def _ffile(self, fi):
        # Pattern to form the `freq` ordered file.
        return self._fdir(fi) + "/timestream.hdf5"

    @property
    def ntime(self):
        """Get the number of timesamples."""

        with h5py.File(self._ffile(0), "r") as f:
            ntime = f.attrs["ntime"]

        return ntime

    def timestream_f(self, fi):
        """Fetch the timestream for a given frequency.

        Parameters
        ----------
        fi : integer
            Frequency to load.

        Returns
        -------
        timestream : np.ndarray[npairs, ntime]
            The visibility timestream.
        """

        with h5py.File(self._ffile(fi), "r") as f:
            ts = f["timestream"][:]
        return ts

    # ====================================================

    # ======== Fetch and generate the m-modes ============

    def _mdir(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self.output_directory + "/mmodes/" + util.natpattern(self.telescope.mmax)
        return pat % abs(mi)

    def _mfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + "/mode.hdf5"

    def mmode(self, mi):
        """Fetch the timestream m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        timestream : np.ndarray[nfreq, pm, npairs]
            The visibility m-modes.
        """

        with h5py.File(self._mfile(mi), "r") as f:
            return f["mmode"][:]

    def generate_mmodes(self):
        """Calculate the m-modes corresponding to the Timestream.

        Perform an MPI transpose for efficiency.
        """

        if os.path.exists(self.output_directory + "/mmodes/COMPLETED_M"):
            if mpiutil.rank0:
                print("******* m-files already generated ********")
            return

        tel = self.telescope
        mmax = tel.mmax
        nfreq = tel.nfreq

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
        lm, sm, em = mpiutil.split_local(mmax + 1)

        # Load in the local frequencies of the time stream
        tstream = np.zeros((lfreq, tel.npairs, self.ntime), dtype=np.complex128)
        for lfi, fi in enumerate(range(sfreq, efreq)):
            tstream[lfi] = self.timestream_f(fi)

        # FFT to calculate the m-modes for the timestream
        row_mmodes = np.fft.fft(tstream, axis=-1) / self.ntime

        ## Combine positive and negative m parts.
        row_mpairs = np.zeros((lfreq, 2, tel.npairs, mmax + 1), dtype=np.complex128)

        row_mpairs[:, 0, ..., 0] = row_mmodes[..., 0]
        for mi in range(1, mmax + 1):
            row_mpairs[:, 0, ..., mi] = row_mmodes[..., mi]
            row_mpairs[:, 1, ..., mi] = row_mmodes[..., -mi].conj()

        # Transpose to get the entirety of an m-mode on each process (i.e. all frequencies)
        col_mmodes = mpiutil.transpose_blocks(
            row_mpairs, (nfreq, 2, tel.npairs, mmax + 1)
        )

        # Transpose the local section to make the m's first
        col_mmodes = np.transpose(col_mmodes, (3, 0, 1, 2))

        for lmi, mi in enumerate(range(sm, em)):
            # Make directory for each m-mode
            if not os.path.exists(self._mdir(mi)):
                os.makedirs(self._mdir(mi))

            # Create the m-file and save the result.
            with h5py.File(self._mfile(mi), "w") as f:
                f.create_dataset("/mmode", data=col_mmodes[lmi])
                f.attrs["m"] = mi

        if mpiutil.rank0:
            # Make file marker that the m's have been correctly generated:
            open(self.output_directory + "/mmodes/COMPLETED_M", "a").close()

        mpiutil.barrier()

    # ====================================================

    # ======== Make and fetch SVD m-modes ================

    def _svdfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + "/svd.hdf5"

    def mmode_svd(self, mi):
        """Fetch the SVD m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        svd_mode : np.ndarray[nfreq, pm, npairs]
            The visibility m-modes.
        """

        with h5py.File(self._svdfile(mi), "r") as f:
            if f["mmode_svd"].shape[0] == 0:
                return np.zeros((0,), dtype=np.complex128)
            else:
                return f["mmode_svd"][:]

    def generate_mmodes_svd(self):
        """Generate the SVD modes for the Timestream."""

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):
            if os.path.exists(self._svdfile(mi)):
                print("File %s exists. Skipping..." % self._svdfile(mi))
                continue

            tm = self.mmode(mi).reshape(self.telescope.nfreq, 2 * self.telescope.npairs)
            svdm = self.beamtransfer.project_vector_telescope_to_svd(mi, tm)

            with h5py.File(self._svdfile(mi), "w") as f:
                f.create_dataset("mmode_svd", data=svdm)
                f.attrs["m"] = mi

        mpiutil.barrier()

    # ====================================================

    # ======== Make map from uncleaned stream ============

    def mapmake_full(self, nside, mapname):
        def _make_alm(mi):
            print("Making %i" % mi)

            mmode = self.mmode(mi)
            sphmode = self.beamtransfer.project_vector_telescope_to_sky(mi, mmode)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, list(range(self.telescope.mmax + 1)))

        if mpiutil.rank0:
            alm = np.zeros(
                (
                    self.telescope.nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    self.telescope.lmax + 1,
                ),
                dtype=np.complex128,
            )

            for mi in range(self.telescope.mmax + 1):
                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            with h5py.File(self.output_directory + "/" + mapname, "w") as f:
                f.create_dataset("/map", data=skymap)

        mpiutil.barrier()

    def mapmake_svd(self, nside, mapname):
        self.generate_mmodes_svd()

        def _make_alm(mi):
            svdmode = self.mmode_svd(mi)

            sphmode = self.beamtransfer.project_vector_svd_to_sky(mi, svdmode)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, list(range(self.telescope.mmax + 1)))

        if mpiutil.rank0:
            alm = np.zeros(
                (
                    self.telescope.nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    self.telescope.lmax + 1,
                ),
                dtype=np.complex128,
            )

            for mi in range(self.telescope.mmax + 1):
                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            with h5py.File(self.output_directory + "/" + mapname, "w") as f:
                f.create_dataset("/map", data=skymap)

        mpiutil.barrier()

    # ====================================================

    # ========== Project into KL-mode basis ==============

    def set_kltransform(self, klname, threshold=None):
        self.klname = klname

        if threshold is None:
            kl = self.manager.kltransforms[self.klname]
            threshold = kl.threshold

        self.klthreshold = threshold

    def _klfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + ("/klmode_%s_%f.hdf5" % (self.klname, self.klthreshold))

    def mmode_kl(self, mi):
        with h5py.File(self._klfile(mi), "r") as f:
            if f["mmode_kl"].shape[0] == 0:
                return np.zeros((0,), dtype=np.complex128)
            else:
                return f["mmode_kl"][:]

    def generate_mmodes_kl(self):
        """Generate the KL modes for the Timestream."""

        kl = self.manager.kltransforms[self.klname]

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):
            if os.path.exists(self._klfile(mi)):
                print("File %s exists. Skipping..." % self._klfile(mi))
                continue

            svdm = self.mmode_svd(
                mi
            )  # .reshape(self.telescope.nfreq, 2*self.telescope.npairs)
            # svdm = self.beamtransfer.project_vector_telescope_to_svd(mi, tm)

            klm = kl.project_vector_svd_to_kl(mi, svdm, threshold=self.klthreshold)

            with h5py.File(self._klfile(mi), "w") as f:
                f.create_dataset("mmode_kl", data=klm)
                f.attrs["m"] = mi

        mpiutil.barrier()

    def collect_mmodes_kl(self):
        def evfunc(mi):
            evf = np.zeros(self.beamtransfer.ndofmax, dtype=np.complex128)

            ev = self.mmode_kl(mi)
            if ev.size > 0:
                evf[-ev.size :] = ev

            return evf

        if mpiutil.rank0:
            print("Creating eigenvalues file (process 0 only).")

        mlist = list(range(self.telescope.mmax + 1))
        shape = (self.beamtransfer.ndofmax,)
        evarray = kltransform.collect_m_array(mlist, evfunc, shape, np.complex128)

        if mpiutil.rank0:
            fname = self.output_directory + (
                "/klmodes_%s_%f.hdf5" % (self.klname, self.klthreshold)
            )
            if os.path.exists(fname):
                print("File: %s exists. Skipping..." % (fname))
                return

            with h5py.File(fname, "w") as f:
                f.create_dataset("evals", data=evarray)

    def fake_kl_data(self):
        kl = self.manager.kltransforms[self.klname]

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):
            evals = kl.evals_m(mi)

            if evals is None:
                klmode = np.array([], dtype=np.complex128)
            else:
                modeamp = ((evals + 1.0) / 2.0) ** 0.5
                klmode = modeamp * (
                    np.array([1.0, 1.0j])
                    * np.random.standard_normal((modeamp.shape[0], 2))
                ).sum(axis=1)

            with h5py.File(self._klfile(mi), "w") as f:
                f.create_dataset("mmode_kl", data=klmode)
                f.attrs["m"] = mi

        mpiutil.barrier()

    def mapmake_kl(self, nside, mapname, wiener=False):
        mapfile = self.output_directory + "/" + mapname

        if os.path.exists(mapfile):
            if mpiutil.rank0:
                print("File %s exists. Skipping...")
            return

        kl = self.manager.kltransforms[self.klname]

        if not kl.inverse:
            raise Exception("Need the inverse to make a meaningful map.")

        def _make_alm(mi):
            print("Making %i" % mi)

            klmode = self.mmode_kl(mi)

            if wiener:
                evals = kl.evals_m(mi, self.klthreshold)

                if evals is not None:
                    klmode *= evals / (1.0 + evals)

            isvdmode = kl.project_vector_kl_to_svd(
                mi, klmode, threshold=self.klthreshold
            )

            sphmode = self.beamtransfer.project_vector_svd_to_sky(mi, isvdmode)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, list(range(self.telescope.mmax + 1)))

        if mpiutil.rank0:
            alm = np.zeros(
                (
                    self.telescope.nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                    self.telescope.lmax + 1,
                ),
                dtype=np.complex128,
            )

            # Determine whether to use m=0 or not
            mlist = list(range(1 if self.no_m_zero else 0, self.telescope.mmax + 1))

            for mi in mlist:
                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            with h5py.File(mapfile, "w") as f:
                f.create_dataset("/map", data=skymap)

        mpiutil.barrier()

    # ====================================================

    # ======= Estimate powerspectrum from data ===========

    @property
    def _psfile(self):
        # Pattern to form the `m` ordered file.
        return self.output_directory + ("/ps_%s.hdf5" % self.psname)

    def set_psestimator(self, psname):
        self.psname = psname

    def powerspectrum(self):
        import scipy.linalg as la

        if os.path.exists(self._psfile):
            print("File %s exists. Skipping..." % self._psfile)
            return

        ps = self.manager.psestimators[self.psname]
        ps.genbands()

        def _q_estimate(mi):
            return ps.q_estimator(mi, self.mmode_kl(mi))

        # Determine whether to use m=0 or not
        mlist = list(range(1 if self.no_m_zero else 0, self.telescope.mmax + 1))
        qvals = mpiutil.parallel_map(_q_estimate, mlist)

        qtotal = np.array(qvals).sum(axis=0)

        fisher, bias = ps.fisher_bias()

        powerspectrum = np.dot(la.inv(fisher), qtotal - bias)

        if mpiutil.rank0:
            with h5py.File(self._psfile, "w") as f:
                cv = la.inv(fisher)
                err = cv.diagonal() ** 0.5
                cr = cv / np.outer(err, err)

                f.create_dataset("fisher/", data=fisher)
                #                f.create_dataset('bias/', data=self.bias)
                f.create_dataset("covariance/", data=cv)
                f.create_dataset("error/", data=err)
                f.create_dataset("correlation/", data=cr)

                f.create_dataset("bandpower/", data=ps.band_power)
                # f.create_dataset('k_start/', data=ps.k_start)
                # f.create_dataset('k_end/', data=ps.k_end)
                # f.create_dataset('k_center/', data=ps.k_center)
                # f.create_dataset('psvalues/', data=ps.psvalues)

                f.create_dataset("powerspectrum", data=powerspectrum)

        # Delete cache of bands for memory reasons
        del ps.clarray
        ps.clarray = None

        mpiutil.barrier()

        return powerspectrum

    # ====================================================

    # ======== Load and save the Pickle files ============

    def __getstate__(self):
        ## Remove the attributes we don't want pickled.
        state = self.__dict__.copy()

        for key in self.__dict__:
            # if (key in delkeys) or (key[0] == "_"):
            if key[0] == "_":
                del state[key]

        return state

    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.output_directory + "/timestreamobject.pickle"

    def save(self):
        """Save out the Timestream object information."""

        # Save pickled telescope object
        if mpiutil.rank0:
            with open(self._picklefile, "wb") as f:
                print("=== Saving Timestream object. ===")
                pickle.dump(self, f)

    @classmethod
    def load(cls, tsdir):
        """Load the Timestream object from disk.

        Parameters
        ----------
        tsdir : string
            Name of the directory containing the Timestream object.
        """

        # Create temporary object to extract picklefile property
        tmp_obj = cls(tsdir, tsdir)

        with open(tmp_obj._picklefile, "rb") as f:
            print("=== Loading Timestream object. ===")
            return pickle.load(f)

    # ====================================================


def cross_powerspectrum(timestreams, psname, psfile):
    import scipy.linalg as la

    if os.path.exists(psfile):
        print("File %s exists. Skipping..." % psfile)
        return

    products = timestreams[0].manager

    ps = products.psestimators[psname]
    ps.genbands()

    nstream = len(timestreams)

    def _q_estimate(mi):
        qp = np.zeros((nstream, nstream, ps.nbands), dtype=np.float64)

        for ti in range(nstream):
            for tj in range(ti + 1, nstream):
                print("Making m=%i (%i, %i)" % (mi, ti, tj))

                si = timestreams[ti]
                sj = timestreams[tj]

                qp[ti, tj] = ps.q_estimator(mi, si.mmode_kl(mi), sj.mmode_kl(mi))
                qp[tj, ti] = qp[ti, tj]

        return qp

    # Determine whether to use m=0 or not
    mlist = list(
        range(1 if timestreams[0].no_m_zero else 0, products.telescope.mmax + 1)
    )
    qvals = mpiutil.parallel_map(_q_estimate, mlist)

    qtotal = np.array(qvals).sum(axis=0)

    fisher, bias = ps.fisher_bias()

    # Subtract bias and reshape into new array
    qtotal = (qtotal - bias).reshape(nstream**2, ps.nbands).T

    powerspectrum = np.dot(la.inv(fisher), qtotal)
    powerspectrum = powerspectrum.T.reshape(nstream, nstream, ps.nbands)

    if mpiutil.rank0:
        with h5py.File(psfile, "w") as f:
            cv = la.inv(fisher)
            err = cv.diagonal() ** 0.5
            cr = cv / np.outer(err, err)

            f.create_dataset("fisher", data=fisher)
            #                f.create_dataset('bias', data=self.bias)
            f.create_dataset("covariance", data=cv)
            f.create_dataset("error", data=err)
            f.create_dataset("correlation", data=cr)

            f.create_dataset("bandpower", data=ps.band_power)
            # f.create_dataset('k_start', data=ps.k_start)
            # f.create_dataset('k_end', data=ps.k_end)
            # f.create_dataset('k_center', data=ps.k_center)
            # f.create_dataset('psvalues', data=ps.psvalues)

            f.create_dataset("powerspectrum", data=powerspectrum)

    # Delete cache of bands for memory reasons
    del ps.clarray
    ps.clarray = None

    mpiutil.barrier()

    return powerspectrum


# kwargs is to absorb any extra params
def simulate(m, outdir, maps=[], ndays=None, resolution=0, seed=None, **kwargs):
    """Create a simulated timestream and save it to disk.

    Parameters
    ----------
    m : ProductManager object
        Products of telescope to simulate.
    outdir : directoryname
        Directory that we will save the timestream into.
    maps : list
        List of map filenames. The sum of these form the simulated sky.
    ndays : int, optional
        Number of days of observation. Setting `ndays = None` (default) uses
        the default stored in the telescope object; `ndays = 0`, assumes the
        observation time is infinite so that the noise is zero.
    resolution : scalar, optional
        Approximate time resolution in seconds. Setting `resolution = 0`
        (default) calculates the value from the mmax.

    Returns
    -------
    timestream : Timestream
    """

    ## Read in telescope system
    bt = m.beamtransfer
    tel = bt.telescope

    lmax = tel.lmax
    mmax = tel.mmax
    nfreq = tel.nfreq
    npol = tel.num_pol_sky

    projmaps = len(maps) > 0

    lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
    local_freq = list(range(sfreq, efreq))

    lm, sm, em = mpiutil.split_local(mmax + 1)

    # If ndays is not set use the default value.
    if ndays is None:
        ndays = tel.ndays

    # Calculate the number of timesamples from the resolution
    if resolution == 0:
        # Set the minimum resolution required for the sky.
        ntime = 2 * mmax + 1
    else:
        # Set the cl
        ntime = int(np.round(24 * 3600.0 / resolution))

    col_vis = np.zeros((tel.npairs, lfreq, ntime), dtype=np.complex128)

    ## If we want to add maps use the m-mode formalism to project a skymap
    ## into visibility space.

    if projmaps:
        # Load file to find out the map shapes.
        with h5py.File(maps[0], "r") as f:
            mapshape = f["map"].shape

        if lfreq > 0:
            # Allocate array to store the local frequencies
            row_map = np.zeros((lfreq,) + mapshape[1:], dtype=np.float64)

            # Read in and sum up the local frequencies of the supplied maps.
            for mapfile in maps:
                with h5py.File(mapfile, "r") as f:
                    row_map += f["map"][sfreq:efreq]

            # Calculate the alm's for the local sections
            row_alm = hputil.sphtrans_sky(row_map, lmax=lmax).reshape(
                (lfreq, npol * (lmax + 1), lmax + 1)
            )

        else:
            row_alm = np.zeros(
                (lfreq, npol * (lmax + 1), lmax + 1), dtype=np.complex128
            )

        # Perform the transposition to distribute different m's across processes. Neat
        # tip, putting a shorter value for the number of columns, trims the array at
        # the same time
        col_alm = mpiutil.transpose_blocks(
            row_alm, (nfreq, npol * (lmax + 1), mmax + 1)
        )

        # Transpose and reshape to shift m index first.
        col_alm = np.transpose(col_alm, (2, 0, 1)).reshape(lm, nfreq, npol, lmax + 1)

        # Create storage for visibility data
        vis_data = np.zeros((lm, nfreq, bt.ntel), dtype=np.complex128)

        # Iterate over m's local to this process and generate the corresponding
        # visibilities
        for mp, mi in enumerate(range(sm, em)):
            vis_data[mp] = bt.project_vector_sky_to_telescope(mi, col_alm[mp])

        # Rearrange axes such that frequency is last (as we want to divide
        # frequencies across processors)
        row_vis = vis_data.transpose((0, 2, 1))  # .reshape((lm * bt.ntel, nfreq))

        # Parallel transpose to get all m's back onto the same processor
        col_vis_tmp = mpiutil.transpose_blocks(row_vis, ((mmax + 1), bt.ntel, nfreq))
        col_vis_tmp = col_vis_tmp.reshape(mmax + 1, 2, tel.npairs, lfreq)

        # Transpose the local section to make the m's the last axis and unwrap the
        # positive and negative m at the same time.
        col_vis[..., 0] = col_vis_tmp[0, 0]
        for mi in range(1, mmax + 1):
            col_vis[..., mi] = col_vis_tmp[mi, 0]
            col_vis[..., -mi] = col_vis_tmp[
                mi, 1
            ].conj()  # Conjugate only (not (-1)**m - see paper)

        del col_vis_tmp

    ## If we're simulating noise, create a realisation and add it to col_vis
    if ndays > 0:
        # Fetch the noise powerspectrum
        noise_ps = tel.noisepower(
            np.arange(tel.npairs)[:, np.newaxis],
            np.array(local_freq)[np.newaxis, :],
            ndays=ndays,
        ).reshape(tel.npairs, lfreq)[:, :, np.newaxis]

        # Seed random number generator to give consistent noise
        if seed is not None:
            # Must include rank such that we don't have massive power deficit from correlated noise
            np.random.seed(seed + mpiutil.rank)

        # Create and weight complex noise coefficients
        noise_vis = (
            np.array([1.0, 1.0j]) * np.random.standard_normal(col_vis.shape + (2,))
        ).sum(axis=-1)
        noise_vis *= (noise_ps / 2.0) ** 0.5

        # Reset RNG
        if seed is not None:
            np.random.seed()

        # Add into main noise sims
        col_vis += noise_vis

        del noise_vis

    # Fourier transform m-modes back to get timestream.
    vis_stream = np.fft.ifft(col_vis, axis=-1) * ntime
    vis_stream = vis_stream.reshape(tel.npairs, lfreq, ntime)

    # The time samples the visibility is calculated at
    tphi = np.linspace(0, 2 * np.pi, ntime, endpoint=False)

    # Create timestream object
    tstream = Timestream(outdir, m)

    ## Iterate over the local frequencies and write them to disk.
    for lfi, fi in enumerate(local_freq):
        # Make directory if required
        if not os.path.exists(tstream._fdir(fi)):
            os.makedirs(tstream._fdir(fi))

        # Write file contents
        with h5py.File(tstream._ffile(fi), "w") as f:
            # Timestream data
            f.create_dataset("/timestream", data=vis_stream[:, lfi])
            f.create_dataset("/phi", data=tphi)

            # Telescope layout data
            f.create_dataset("/feedmap", data=tel.feedmap)
            f.create_dataset("/feedconj", data=tel.feedconj)
            f.create_dataset("/feedmask", data=tel.feedmask)
            f.create_dataset("/uniquepairs", data=tel.uniquepairs)
            f.create_dataset("/baselines", data=tel.baselines)

            # Write metadata
            f.attrs["beamtransfer_path"] = os.path.abspath(bt.directory)
            f.attrs["ntime"] = ntime

    tstream.save()

    mpiutil.barrier()

    return tstream
