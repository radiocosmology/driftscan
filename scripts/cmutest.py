
from cylsim import cmutelescope

from cylsim import visibility
from cylsim import beamtransfer

cmut = cmutelescope.UnpolarisedCMU()

bt = beamtransfer.BeamTransfer('cmut/', telescope=cmut)

bt.generate_cache()


