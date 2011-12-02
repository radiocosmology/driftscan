
from cylsim import cmutelescope

from cylsim import visibility
from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import psestimation

import numpy as np
import os

#teldir = '/mnt/scratch-3week/jrs65/chimebt/fullbt_50/'
teldir = ((os.environ['SCRATCH'] if 'SCRATCH' in os.environ else ".") + '/cylinder/fullbt_50/')

bt = beamtransfer.BeamTransfer(teldir)
#bt.generate_cache()

klt = kltransform.KLTransform(bt, evsubdir='ev2')

pse = psestimation.PSEstimation(klt, subdir='ps_f10/')

#pse.bands = np.concatenate((np.linspace(0.0, 0.2, 12, endpoint=False), np.logspace(np.log10(0.2), np.log10(3.0), 5)))
#pse.bands = np.linspace(0.1, 0.2, 3, endpoint=True)

pse.threshold = 10.0

pse.genbands()
#pse.fisher_mpi()
pse.fisher_mpi()
