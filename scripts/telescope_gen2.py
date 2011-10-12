
from cylsim import cylinder
from cylsim import beamtransfer

cyl = cylinder.UnpolarisedCylinderTelescope()


cyl.num_freq = 50
cyl.freq_lower = 400.0
cyl.freq_upper = 600.0

cyl.cylinder_width = 20.0

cyl.num_feeds = 50
cyl.feed_spacing = 0.25

bt = beamtransfer.BeamTransfer('/scratch/jrs65/cylinder/telescope2/', telescope=cyl)

bt.generate_cache()


