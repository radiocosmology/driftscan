import argparse

import numpy as np

import h5py
import healpy

from cora.util import hputil

from drift.core import beamtransfer
from drift.util import mpiutil
from mpi4py import MPI

## Read arguments in.
parser = argparse.ArgumentParser(description="Filter a map using S/N eigenmodes.")
parser.add_argument("teldir", help="The telescope directory to use.")
parser.add_argument("mapfile", help="Input map.")
parser.add_argument("outfile", help="Output map.")
parser.add_argument("threshold", help="Threshold S/N value to cut at.", type=float)
parser.add_argument(
    "-e", "--evsubdir", help="The subdirectory containing the eigensystem files."
)
args = parser.parse_args()

## Read in cylinder system
bt = beamtransfer.BeamTransfer(args.teldir)
# klt = kltransform.KLTransform(bt, subdir=args.evsubdir)
cyl = bt.telescope
ntel = bt.ntel * bt.nfreq
mmax = cyl.mmax

cut = args.threshold


def blackman_harris(N):

    k = np.arange(N)

    # w = 0.3635819 - 0.4891775*np.cos(2*np.pi*k/(N-1)) + 0.1365995*np.cos(4*np.pi*k/(N-1)) -0.0106411*np.cos(6*np.pi*k/(N-1))
    w = (
        0.35875
        - 0.48829 * np.cos(2 * np.pi * k / (N - 1))
        + 0.14128 * np.cos(4 * np.pi * k / (N - 1))
        - 0.01168 * np.cos(6 * np.pi * k / (N - 1))
    )
    return w


nside = 0


alm = np.zeros((cyl.nfreq, cyl.lmax + 1, cyl.lmax + 1), dtype=np.complex128)

if mpiutil.rank0:
    ## Useful output
    print("==================================")
    print("Projecting file:\n    %s\ninto:\n    %s" % (args.mapfile, args.outfile))
    print("Using beamtransfer: %s" % args.teldir)
    print("Truncating to modes with S/N > %f" % cut)
    print("==================================")

    # Calculate alm's and broadcast
    print("Read in skymap.")
    f = h5py.File(args.mapfile)
    skymap = f["map"][:]
    f.close()
    nside = healpy.get_nside(skymap[0])

    alm = hputil.sphtrans_sky(skymap, lmax=cyl.lmax)
# else:
#    almr = None

mpiutil.world.Bcast([alm, MPI.COMPLEX16], root=0)

cb = cyl.baselines + np.array([[cyl.u_width, 0.0]])

if cyl.positive_m_only:
    taumax = (cb ** 2).sum(axis=-1) ** 0.5 / 3e8
else:
    taumax = (np.concatenate((cb, cb)) ** 2).sum(axis=-1) ** 0.5 / 3e8
tau = np.fft.fftfreq(cyl.nfreq, (cyl.frequencies[1] - cyl.frequencies[0]) * 1e6)

# blmask = (np.abs(tau)[:, np.newaxis] > cut * (taumax[np.newaxis, :] + 10.0 / 3e8))

# blmask = (np.abs(tau)[:, np.newaxis] / ((taumax[np.newaxis, :] + 3.0/3e8)))

# bl2 = 10 - np.abs(blmask)
# bl3 = np.exp(-np.where(bl2 > 0, bl2, 0)**2 / 2.0)

# blmask = bl3
blmask = np.exp(-((np.arange(50.0) - 25.0) ** 2) / (2 * (3.0 ** 2)))[:, np.newaxis]


def projm(mi):
    ## Worker function for mapping over list and projecting onto signal modes.
    print("Projecting %i" % mi)

    sky_vec = bt.project_vector_forward(mi, alm[:, :, mi])

    tau_vec = np.fft.fft(blackman_harris(cyl.nfreq)[:, np.newaxis] * sky_vec, axis=0)

    tau_vec = np.fft.ifft(tau_vec * blmask, axis=0)

    alm2 = bt.project_vector_backward(mi, tau_vec)

    return alm2


# Project m-modes across different processes
mlist = list(range(mmax + 1))
mpart = mpiutil.partition_list_mpi(mlist)
mproj = [[mi, projm(mi)] for mi in mpart]

if mpiutil.rank0:
    print("Gather results onto root process")
p_all = mpiutil.world.gather(mproj, root=0)


# Save out results
if mpiutil.rank0:

    palm = np.zeros_like(alm)

    print("Combining results.")
    for p_process in p_all:

        for mi, proj in p_process:

            if proj is None:
                continue

            palm[:, :, mi] = proj.reshape(palm.shape[:-1])

    print("Transforming onto sky.")
    proj_map = hputil.sphtrans_inv_sky(palm, nside)

    print("Saving file.")
    f = h5py.File(args.outfile, "w")
    f.attrs["threshold"] = cut

    f.create_dataset("/klproj", data=proj_map)

    f.close()
