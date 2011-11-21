from cylsim import beamtransfer
from cylsim import kltransform


bt = beamtransfer.BeamTransfer("cmut/")

bm = bt.beam_m(100)

klt = kltransform.KLTransform(bt)

evals, evecs = klt.modes_m(-100)

sig = klt.signal()
fg = klt.foreground()

sigt = klt.project_sky_matrix_forward(-100, sig)
fgt = klt.project_sky_matrix_forward(-100, fg)

#fgc = klt.project_sky_matrix_forward_c(-100, fg)

cs, cn = klt.signal_covariance(-100)

sigt2 = klt.project_tel_matrix_forward(-100, cs.reshape((2400, 2400)))
fgt2 = klt.project_tel_matrix_forward(-100, cn.reshape((2400, 2400)))

