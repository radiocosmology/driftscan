import sys

import numpy as np
import scipy.linalg as la

import h5py

from drift.core import beamtransfer, kltransform
from drift.util import mpiutil

path = sys.argv[1]

bt = beamtransfer.BeamTransfer(path)

# The size of the SVD output matrices
svd_len = min(bt.telescope.num_pol_sky * (bt.telescope.lmax + 1), bt.ntel)


def svd_func(mi):

    print(mi)
    # Open m beams for reading.

    sv = np.zeros((bt.nfreq, svd_len), dtype=np.float64)

    with h5py.File(bt._mfile(mi), "r") as fm:

        for fi in range(bt.nfreq):
            # print "fi", fi
            # Read the positive and negative m beams, and combine into one.
            bf = fm["beam_m"][fi][:].reshape(
                bt.ntel, bt.telescope.num_pol_sky, (bt.telescope.lmax + 1)
            )
            bf = bf[:, :3, :]
            bf = bf.reshape(bt.ntel, 3 * (bt.telescope.lmax + 1))

            # Weight by noise matrix
            noisew = bt.telescope.noisepower(
                np.arange(bt.telescope.npairs), fi
            ).flatten() ** (-0.5)
            noisew = np.concatenate([noisew, noisew])
            bf = bf * noisew[:, np.newaxis]

            # Regularise me.
            bf = bf + bf.max() * 1e-11 * np.eye(bf.shape[0], bf.shape[1])

            u, s, v = la.svd(bf)
            # s = np.linalg.svd(bf, compute_uv=False)
            sv[fi] = s

    return sv


svdspectrum = kltransform.collect_m_array(
    list(range(bt.telescope.mmax + 1)), svd_func, (bt.nfreq, svd_len), np.float64
)

if mpiutil.rank0:

    with h5py.File(bt.directory + "/allsvdspectrum.hdf5", "w") as f:

        f.create_dataset("singularvalues", data=svdspectrum)

mpiutil.barrier()
