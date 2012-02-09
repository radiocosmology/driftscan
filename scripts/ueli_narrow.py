from cylsim import cylinder
from cylsim import beamtransfer
from cylsim import kltransform

import os

cyl = cylinder.UnpolarisedCylinderTelescope()

teldir = ((os.environ['SCRATCH'] if 'SCRATCH' in os.environ else ".") + '/cylinder/ueli/narrow')

cyl.num_freq = 25
cyl.freq_lower = 400
cyl.freq_upper = 450

cyl.cylinder_width = 2.0
cyl.touching = False
cyl.cylspacing = 10.0
cyl.num_cylinders = 2

cyl.feed_spacing = 0.5
cyl.num_feeds = 20

cyl.tsys_flat = 1.0

bt = beamtransfer.BeamTransfer(teldir, telescope=cyl)
bt.generate_cache()

klt = kltransform.KLTransform(bt)
klt.subset = False
klt.generate()

