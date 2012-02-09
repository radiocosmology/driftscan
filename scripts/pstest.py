from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import skymodel
from cylsim import psestimation

from simulations import corr21cm

from scipy.integrate import quad

import numpy as np

import os
import time

mi = -200

#bt = beamtransfer.BeamTransfer("/mnt/scratch-3week/jrs65/chimebt/fullbt_50")
teldir = ((os.environ['SCRATCH'] if 'SCRATCH' in os.environ else ".") + ('/cylinder/chimebt50/band_4/'))
bt = beamtransfer.BeamTransfer(teldir)


klt = kltransform.KLTransform(bt)

pse = psestimation.PSEstimation(klt)
pse.bands = np.logspace(-3.0, 1.0, 4)
pse.genbands()

st = time.time()
pk1 = klt.project_sky_matrix_forward(mi, pse.clarray[1])
et = time.time()
print (et - st)


st = time.time()
pk2 = klt.project_sky_matrix_forward_old(mi, pse.clarray[1])
et = time.time()
print (et - st)


#fab = pse.fisher_m(-100)
