import numpy as np

import h5py

import time

import scipy.linalg as la

from cylsim import beamtransfer, kltransform, skysim, util, skymodel, mpiutil, hputil

from simulations import foregroundsck


#bt = beamtransfer.BeamTransfer("/Users/richard/science/data/ueli/wide/")
bt = beamtransfer.BeamTransfer("/mnt/raid-cita/jrs65/ueli/wide/")
klt = kltransform.KLTransform(bt)

cyl = bt.telescope

cut = 3.0

stem = "uw_c_%.2f" % cut

mpiutil.barrier()

print "Opening files."

f = h5py.File("maps_uw_c1.hdf5", "r")
cs_alm = f["/map_cs"][:]
gs_alm = f["/map_gs"][:]
sg_alm = f["/map_sg"][:]

f.close()

print "Done."

mmax = cyl.mmax
ntel = cyl.nbase * cyl.nfreq * cyl.num_pol_telescope




#mlist = range(-100, -97)
#mlist = range(-mmax, mmax+1)
#mlist = range(-430, 210)

mlist = range(-72, -65)
#mlist = range(-430, -400)

mpart = mpiutil.partition_list_mpi(mlist)



    #for mi in range(-mmax, mmax+1):
def projm(mi, alm_list):

    print "Projecting %i" % mi

    def projalm(alm):

        bproj = bt.project_vector_forward(mi, alm[:, :, mi]).flatten()
        sbinv = bt.project_vector_backward(mi, bproj)

        eproj = klt.project_tel_vector_forward(mi, bproj, threshold=cut)
        bpinv = klt.project_tel_vector_backward(mi, eproj, threshold=cut)

        seinv = bt.project_vector_backward(mi, bpinv)

        return [sbinv, seinv]

    return [ projalm(alm) for alm in alm_list ]

    
alms = [cs_alm, gs_alm, sg_alm]    

mproj = [[mi, projm(mi, alms)] for mi in mpart]

p_all = mpiutil.world.gather(mproj, root=0)

if mpiutil.rank0:


    
    cs_balm = np.zeros_like(cs_alm)
    gs_balm = np.zeros_like(gs_alm)
    sg_balm = np.zeros_like(sg_alm)


    cs_palm = np.zeros_like(cs_alm)
    gs_palm = np.zeros_like(gs_alm)
    sg_palm = np.zeros_like(sg_alm)

        

    for p_process in p_all:

        for mi, proj in p_process:

            if proj == None:
                continue

            cs_balm[:, :, mi] = proj[0][0].reshape(cs_alm.shape[:-1])
            gs_balm[:, :, mi] = proj[1][0].reshape(gs_alm.shape[:-1])
            sg_balm[:, :, mi] = proj[2][0].reshape(sg_alm.shape[:-1])

            cs_palm[:, :, mi] = proj[0][1].reshape(cs_alm.shape[:-1])
            gs_palm[:, :, mi] = proj[1][1].reshape(gs_alm.shape[:-1])
            sg_palm[:, :, mi] = proj[2][1].reshape(sg_alm.shape[:-1])


    
    f = h5py.File("mapbt_%s.hdf5" % stem, 'w')

    f.create_dataset('/map_cs', data=cs_balm)
    f.create_dataset('/map_gs', data=gs_balm)
    f.create_dataset('/map_sg', data=sg_balm)

    f.close()


    f = h5py.File("mapev_%s.hdf5" % stem, 'w')

    f.create_dataset('/map_cs', data=cs_palm)
    f.create_dataset('/map_gs', data=gs_palm)
    f.create_dataset('/map_sg', data=sg_palm)

    f.close()
