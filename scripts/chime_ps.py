
from cylsim import cmutelescope

from cylsim import visibility
from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import psestimation


import os

teldir = ((os.environ['SCRATCH'] if 'SCRATCH' in os.environ else ".") + '/cylinder/fullbt_50/')


bt = beamtransfer.BeamTransfer(teldir)
#bt.generate_cache()

klt = kltransform.KLTransform(bt, evsubdir='ev2')

pse = psestimation.PSEstimation(klt)

pse.genbands()
pse.fisher_mpi()
