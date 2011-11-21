from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import skymodel

from simulations import corr21cm

from scipy.integrate import quad

import numpy as np

bt = beamtransfer.BeamTransfer("cmut/")

bm = bt.beam_m(100)

klt = kltransform.KLTransform(bt)

evals, evecs = klt.modes_m(-100)

sig = klt.signal()
fg = klt.foreground()

sigt = klt.project_sky_matrix_forward(-100, sig)
fgt = klt.project_sky_matrix_forward(-100, fg)

#fgc = klt.project_sky_matrix_forward_c(-100, fg)

#cs, cn = klt.signal_covariance(-100)

#sigt2 = klt.project_tel_matrix_forward(-100, cs.reshape((2400, 2400)))
#fgt2 = klt.project_tel_matrix_forward(-100, cn.reshape((2400, 2400)))


bands = list(np.logspace(-1.3, 0.7, 7))


def uniform_band(k, kstart, kend):
    return np.where(np.logical_and(k > kstart, k < kend), np.ones_like(k), np.zeros_like(k))

band_pk = [((lambda bs, be: (lambda k: uniform_band(k, bs, be)))(b_start, b_end), b_start, b_end) for b_start, b_end in zip(bands[:-1], bands[1:])]


cr = corr21cm.Corr21cm()

bpower = [quad(cr.ps_vv, bs, be, ) for pk, bs, be in band_pk]
bpow = np.array(zip(*bpower)[0])

def make_clzz(pk):
    print "Making C_l(z,z')"
    crt = corr21cm.Corr21cm(ps = pk)

    clzz = skymodel.im21cm_model(klt.telescope.lmax, klt.telescope.frequencies,
                                 klt.telescope.num_pol_sky, cr = crt)

    return clzz

cl_alpha = [make_clzz(pk) for pk, bs, be in band_pk]

def makeproj(clzz):
    print "Projecting to eigenbasis."
    return klt.project_sky_matrix_forward(-100, clzz)

c_alpha = [makeproj(clzz) for clzz in cl_alpha]

ci = np.diag(1.0 / (evals+1))

print "Making fisher."
fab = np.array([ [ np.trace(np.dot(np.dot(c_a, ci), np.dot(c_b, ci))) for c_b in c_alpha] for c_a in c_alpha ])

fcab = (np.linalg.inv(fab) / np.outer(bpow, bpow))

stdev = fcab.diagonal()**0.5
