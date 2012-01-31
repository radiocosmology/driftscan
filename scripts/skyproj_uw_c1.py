import numpy as np

import h5py

import time

import scipy.linalg as la

from cylsim import beamtransfer, kltransform, skysim, util, skymodel, mpiutil

from simulations import foregroundsck


bt = beamtransfer.BeamTransfer("/mnt/scratch-3week/jrs65/ueli/wide/")
klt = kltransform.KLTransform(bt)

cyl = bt.telescope

cut = 0.0

stem = "uw_c1"

if mpiutil.rank0 and False:
    
    print "Generating foregrounds...."
    # Constrained realisation
    cs = skysim.c_syn(256, cyl.frequencies)
    cs_alm = skysim.sphtrans_sky(cs, lmax=cyl.lmax)

    # Gaussian realisation
    cla = skymodel.foreground_model(cyl.lmax, cyl.frequencies, 1)[0, 0]
    gs_alm = util.mkfullsky(cla, 256, alms=True)

    # Signal realisation
    cla = skymodel.im21cm_model(cyl.lmax, cyl.frequencies, 1)[0, 0]
    sg_alm = util.mkfullsky(cla, 256, alms=True)

    f = h5py.File("maps_%s.hdf5" % stem, "w")
    f.create_dataset("/map_cs", data=cs_alm)
    f.create_dataset("/map_gs", data=gs_alm)
    f.create_dataset("/map_sg", data=sg_alm)
    f.close()

mpiutil.barrier()

print "Opening files."

f = h5py.File("maps_%s.hdf5" % stem, "r")
cs_alm = f["/map_cs"][:]
gs_alm = f["/map_gs"][:]
sg_alm = f["/map_sg"][:]

f.close()

print "Done."

mmax = cyl.mmax
ntel = cyl.nbase * cyl.nfreq * cyl.num_pol_telescope




#mlist = range(-100, -97)
mlist = range(-mmax, mmax+1)
#mlist = range(-430, 210)
#mlist = range(-430, -400)

mpart = mpiutil.partition_list_mpi(mlist)



    #for mi in range(-mmax, mmax+1):
def projm(mi, alm_list):

    print "Projecting %i" % mi

    mvals, mvecs = klt.modes_m(mi, threshold=cut)
    
    if mvals is None:
        return None
    nmode = mvals.size
    
    #st = time.time()
    svecs = klt.skymodes_m(mi, threshold=cut).reshape((nmode, -1))
    #et = time.time()
    #print et-st, svecs.size*16.0 / 2**30.0
    
    st = time.time()
    mvh = mvecs.T
    #cinv = np.dot(mvh, la.inv(np.dot(mvecs.conj(), mvh)))
    et = time.time()
    #print "Inv: ", et-st, cinv.size*16.0 / 2**30.0
    
    def projalm(alm):
        pval = np.zeros(ntel, dtype=np.complex128)
        #tproj = bt.project_vector_forward(mi, alm[:, :, mi]).flatten()
        mproj = np.dot(svecs, alm[:, :, mi].flatten())
        pval[-nmode:] = mproj

        st = time.time()
        #almp = bt.project_vector_backward(mi, np.dot(cinv, mproj))
        et = time.time()
        print "Pback: ", et-st
        
        #return [pval, almp]
        return [pval]

    return [ projalm(alm) for alm in alm_list ]

    
alms = [cs_alm, gs_alm, sg_alm]    

mproj = [[mi, projm(mi, alms)] for mi in mpart]

p_all = mpiutil.world.gather(mproj, root=0)

if mpiutil.rank0:

    
    cs_pvals = np.zeros((2*mmax+1, ntel), dtype=np.complex128)
    gs_pvals = np.zeros((2*mmax+1, ntel), dtype=np.complex128)
    sg_pvals = np.zeros((2*mmax+1, ntel), dtype=np.complex128)

    cs_palm = np.zeros_like(cs_alm)
    gs_palm = np.zeros_like(gs_alm)
    sg_palm = np.zeros_like(sg_alm)

        

    for p_process in p_all:

        for mi, proj in p_process:

            if proj == None:
                continue

            cs_pvals[mi, :] = proj[0][0]
            gs_pvals[mi, :] = proj[1][0]
            sg_pvals[mi, :] = proj[2][0]

            #cs_palm[:, :, mi] = proj[0][1].reshape(cs_alm.shape[:-1])
            #gs_palm[:, :, mi] = proj[1][1].reshape(gs_alm.shape[:-1])
            #sg_palm[:, :, mi] = proj[2][1].reshape(sg_alm.shape[:-1])


    f = h5py.File("projtest_%s.hdf5" % stem, 'w')

    f.create_dataset('/proj_cs', data=cs_pvals)
    f.create_dataset('/proj_gs', data=gs_pvals)
    f.create_dataset('/proj_sg', data=sg_pvals)

    f.close()


    #f = h5py.File("maptest_%s.hdf5" % stem, 'w')

    #f.create_dataset('/map_cs', data=cs_palm)
    #f.create_dataset('/map_gs', data=gs_palm)
    #f.create_dataset('/map_sg', data=sg_palm)

    f.close()
