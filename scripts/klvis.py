from cylsim import beamtransfer, kltransform, hputil, util

import numpy as np

import healpy

mi = -10
vn = 0

bt = beamtransfer.BeamTransfer("/Users/richard/science/data/ueli/wide/")

klt = kltransform.KLTransform(bt)


cyl = bt.telescope

@util.cache_last
def inv_m(mi):
    evals, evecs = klt.modes_m(mi)
    inv_ev = kltransform.inv_gen(evecs)

    return inv_ev


def projmode(mi, vn, inv=True):

    mode = inv_m(mi)[:, vn] if inv else klt.modes_m(mi)[1][vn]
    
    btv = bt.project_vector_backward(mi, mode)

    sv = np.zeros((cyl.nfreq, cyl.lmax+1, 2*cyl.lmax+1), dtype=np.complex128)

    sv[:, :, mi] = btv.reshape((cyl.nfreq, cyl.lmax+1))

    klsky = hputil.sphtrans_inv_sky(sv, 128)

    return klsky, btv

klsky = projmode(mi, 0)
