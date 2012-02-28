from cylsim import cylinder
from cylsim import beamtransfer
from cylsim import kltransform

import os
import sys

import numpy as np

import h5py
import healpy

from cylsim import skysim, util, skymodel, mpiutil, hputil


teldir = sys.argv[1]
mapfile = sys.argv[2]
outfile = sys.argv[3]

if mpiutil.rank0:
    print "Read in beamtransfers and extract telescope object."
bt = beamtransfer.BeamTransfer(teldir)
cyl = bt.telescope
mmax = cyl.mmax


nside = 0

# Calculate alm's and broadcast
if mpiutil.rank0:
    print "Read in skymap."
    f = h5py.File(mapfile)
    skymap = f['map'][:]
    f.close()
    nside = healpy.npix2nside(skymap[0].size)

    alm = hputil.sphtrans_sky(skymap, lmax=cyl.lmax)
else:
    alm = None
    
alm = mpiutil.world.bcast(alm, root=0)




def projm(mi):

    print "Projecting %i" % mi

    bproj = bt.project_vector_forward(mi, alm[:, :, mi]).flatten()
    sbdirty = bt.project_vector_backward_dirty(mi, bproj)
    sbinv = bt.project_vector_backward(mi, bproj)

    return [sbdirty, sbinv]

# Project m-modes across different processes
mlist = range(-mmax, mmax+1)
mpart = mpiutil.partition_list_mpi(mlist)
mproj = [[mi, projm(mi)] for mi in mpart]

if mpiutil.rank0:
    print "Gather results onto root process"
p_all = mpiutil.world.gather(mproj, root=0)


# Save out results
if mpiutil.rank0:

    balm = np.zeros_like(alm)
    dalm = np.zeros_like(alm)

    print "Combining results."
    for p_process in p_all:

        for mi, proj in p_process:

            if proj == None:
                continue

            dalm[:, :, mi] = proj[0].reshape(dalm.shape[:-1])
            balm[:, :, mi] = proj[1].reshape(balm.shape[:-1])

    print "Transforming onto sky."
    beamp_map = hputil.sphtrans_inv_sky(balm, nside)
    dirty_map = hputil.sphtrans_inv_sky(dalm, nside)

    print "Saving file."
    f = h5py.File(outfile, 'w')

    f.create_dataset('/beamp', data=beamp_map)
    f.create_dataset('/dirty', data=dirty_map)

    f.close()

