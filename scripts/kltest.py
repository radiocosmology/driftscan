from cylsim import beamtransfer
from cylsim import kltransform


bt = beamtransfer.BeamTransfer("cmut/")

bm = bt.beam_m(100)

klt = kltransform.KLTransform(bt)

evals, evecs = klt.transform_save(-100)


