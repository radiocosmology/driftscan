from cylsim import cylinder
from cylsim import beamtransfer
from cylsim import kltransform

import os

cyl = cylinder.CylBT()

teldir = ((os.environ['SCRATCH'] if 'SCRATCH' in os.environ else ".") + '/cylinder/fullbt/')

cyl.num_freq = 80
cyl.freq_lower = 640
cyl.freq_upper = 720

cyl.cylinder_width = 20.0
cyl.num_cylinders = 5

cyl.feed_spacing = 0.4
cyl.num_feeds = int(100.0 / cyl.feed_spacing)
cyl.maxlength = 28.2

bt = beamtransfer.BeamTransfer(teldir, telescope=cyl)
#bt.generate_cache()

klt = kltransform.KLTransform(bt)

#klt.generate()

