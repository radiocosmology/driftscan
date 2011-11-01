
from cylsim import cylinder
from cylsim import beamtransfer

cyl = cylinder.PolarisedCylinderTelescope()


cyl.num_freq = 25
cyl.freq_lower = 500.0
cyl.freq_upper = 600.0

cyl.cylinder_width = 20.0

cyl.num_feeds = 30
cyl.feed_spacing = 0.25

bt = beamtransfer.BeamTransfer('poltelescope2/', telescope=cyl)

bt.generate_cache()


