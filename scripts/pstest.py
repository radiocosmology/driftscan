from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import skymodel
from cylsim import psestimation

from simulations import corr21cm

from scipy.integrate import quad

import numpy as np

bt = beamtransfer.BeamTransfer("cmut/")

klt = kltransform.KLTransform(bt)

pse = psestimation.PSEstimation(klt)

pse.genbands()
fab = pse.fisher_m(-100)
