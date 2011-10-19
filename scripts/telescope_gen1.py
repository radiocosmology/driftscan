
from cylsim import cylinder
from cylsim import beamtransfer

cyl = cylinder.UnpolarisedCylinderTelescope()


cyl.num_freq = 100
cyl.freq_lower = 400.0
cyl.freq_upper = 600.0

cyl.cylinder_width = 20.0

cyl.num_feeds = 100
cyl.feed_spacing = 0.25

bt = beamtransfer.BeamTransfer('/scratch/jrs65/cylinder/telescope1/', telescope=cyl)

bt.generate_cache()


