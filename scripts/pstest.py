from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import skymodel
from cylsim import psestimation

from simulations import corr21cm

from scipy.integrate import quad

import numpy as np

bt = beamtransfer.BeamTransfer("/mnt/scratch-3week/jrs65/chimebt/fullbt_50")

klt = kltransform.KLTransform(bt, evsubdir='ev2/')

pse = psestimation.PSEstimation(klt)
pse.bands = np.logspace(-3.0, 1.0, 4)
pse.genbands()

st = time.time()
pk1 = klt.project_sky_matrix_forward(-50, pse.clarray[1])
et = time.time()
print (et - st)


st = time.time()
pk2 = klt.project_sky_matrix_forward_old(-50, pse.clarray[1])
et = time.time()
print (et - st)


#fab = pse.fisher_m(-100)
