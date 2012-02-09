from cylsim import cylinder
from cylsim import beamtransfer
from cylsim import kltransform

import os

cyl = cylinder.UnpolarisedCylinderTelescope()

teldir = ((os.environ['SCRATCH'] if 'SCRATCH' in os.environ else ".") + '/cylinder/voltest/')

cyl.num_freq = 25
cyl.freq_lower = 700.0
cyl.freq_upper = 800.0

cyl.cylinder_width = 10.0
cyl.num_cylinders = 2

cyl.feed_spacing = 0.5
cyl.num_feeds = 20

cyl.tsys_flat = 5.0

bt = beamtransfer.BeamTransfer(teldir, telescope=cyl)
#bt.generate_cache()

klt = kltransform.KLTransform(bt)

klt.foreground()

# Set zero foregrounds.
klt._cvfg[:] = 0.0

klt.subset = False
klt.generate()

