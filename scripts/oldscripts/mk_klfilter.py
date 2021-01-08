import argparse

import numpy as np

import h5py
import healpy

from cora.util import hputil

from drift.core import beamtransfer
from drift.core import kltransform
from drift.util import mpiutil

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
klt = kltransform.KLTransform(bt, subdir=args.evsubdir)
cyl = bt.telescope
ntel = bt.ntel * bt.nfreq
mmax = cyl.mmax

cut = args.threshold


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
    f = h5py.File(args.mapfile, "r")
    skymap = f["map"][:]
    f.close()
    nside = healpy.get_nside(skymap[0])

    alm = hputil.sphtrans_sky(skymap, lmax=cyl.lmax)
# else:
#    almr = None

# mpiutil.world.Bcast([alm, MPI.COMPLEX16], root=0)


def projm(mi):
    ## Worker function for mapping over list and projecting onto signal modes.
    print("Projecting %i" % mi)

    mvals, mvecs = klt.modes_m(mi, threshold=cut)

    if mvals is None:
        return None

    ev_vec = klt.project_sky_vector_forward(mi, alm[:, :, mi], threshold=cut)
    tel_vec = klt.project_tel_vector_backward(mi, ev_vec, threshold=cut)

    alm2 = bt.project_vector_backward(mi, tel_vec)

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
