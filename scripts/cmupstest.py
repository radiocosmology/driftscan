
from cylsim import cmutelescope

from cylsim import visibility
from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import psestimation


bt = beamtransfer.BeamTransfer('cmut/')
klt = kltransform.KLTransform(bt)

pse = psestimation.PSEstimation(klt)

pse.genbands()
pse.fisher_mpi(range(-200, -190))
