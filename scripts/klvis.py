from cylsim import beamtransfer, kltransform, hputil

import numpy as np

import healpy

mi = -10
vn = 0

bt = beamtransfer.BeamTransfer("/Users/richard/science/data/ueli/wide/")

klt = kltransform.KLTransform(bt)

evals, evecs = klt.modes_m(mi)
inv_ev = kltransform.inv_gen(evecs)

cyl = bt.telescope

def projmode(mi, mode):
    btv = bt.project_vector_backward(mi, mode)

    sv = np.zeros((cyl.nfreq, cyl.lmax+1, 2*cyl.lmax+1), dtype=np.complex128)

    sv[:, :, mi] = btv.reshape((cyl.nfreq, cyl.lmax+1))

    klsky = hputil.sphtrans_inv_sky(sv, 128)

    return klsky

klsky = projmode(mi, inv_ev[:, vn])
