from cylsim import beamtransfer
from cylsim import kltransform


bt = beamtransfer.BeamTransfer("cmut/")

bm = bt.beam_m(100)

klt = kltransform.KLTransform(bt)

fg1 = klt.foreground()
sg1 = klt.signal()

cv_sg, cv_fg = klt.signal_covariance(-100)
