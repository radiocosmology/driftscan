
from cylsim import cylinder
from cylsim import beamtransfer

cyl = cylinder.UnpolarisedCylinderTelescope()
cyl.cylinder_width=10.0
cyl.num_freq = 4
cyl.num_feeds = 40

bt = beamtransfer.BeamTransfer('testdir2/', telescope=cyl)

bt.generate_cache()
