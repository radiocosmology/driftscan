import sys

import numpy as np
import h5py

from cora.util import hputil
from drift.core import beamtransfer
from drift.util import mpiutil

btdir = sys.argv[1]
nside = int(sys.argv[2])
outfile = sys.argv[3]


bt = beamtransfer.BeamTransfer(btdir)

cyl = bt.telescope
mmax = cyl.mmax

print("Making map with T_sys = %f" % cyl.tsys_flat)

shape = (cyl.nfreq, cyl.nbase)
ind = np.indices(shape)

noise = np.squeeze(cyl.noisepower(ind[1], ind[0]) / 2.0) ** 0.5

if not cyl.positive_m_only:
    shape = (shape[0], 2, shape[1])
    noise2 = np.empty(shape, dtype=np.float64)
    noise2[:, 0] = noise
    noise2[:, 1] = noise
    noise = noise2


def noisem(mi):

    print("Noise vector %i" % mi)

    vis = (np.random.standard_normal(shape + (2,)) * np.array([1.0, 1.0j])).sum(
        axis=-1
    ) * noise
    sbinv = bt.project_vector_backward(mi, vis)

    return sbinv


# Project m-modes across different processes
mlist = list(range(mmax + 1))
mpart = mpiutil.partition_list_mpi(mlist)
mproj = [[mi, noisem(mi)] for mi in mpart]

if mpiutil.rank0:
    print("Gather results onto root process")
p_all = mpiutil.world.gather(mproj, root=0)


# Save out results
if mpiutil.rank0:

    nalm = np.zeros((cyl.nfreq, cyl.lmax + 1, cyl.lmax + 1), dtype=np.complex128)

    print("Combining results.")

    for p_process in p_all:

        for mi, proj in p_process:

            if proj is None:
                continue

            nalm[:, :, mi] = proj.reshape(nalm.shape[:-1])

    print("Transforming onto sky.")
    noise_map = hputil.sphtrans_inv_sky(nalm, nside)

    print("Saving file.")
    f = h5py.File(outfile, "w")

    f.create_dataset("/map", data=noise_map)

    f.close()
