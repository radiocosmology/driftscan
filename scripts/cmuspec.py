from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import skymodel

from simulations import corr21cm

from scipy.integrate import quad

import numpy as np

bt = beamtransfer.BeamTransfer("cmut/")

klt = kltransform.KLTransform(bt)

klt.generate()
