from cylsim import cylinder
from cylsim import beamtransfer
from cylsim import kltransform

import os

import sys

bi = int(sys.argv[1])

### Directory to save into
teldir = ((os.environ['SCRATCH'] if 'SCRATCH' in os.environ else ".") + ('/cylinder/chimebt50/band_%i/ % bi'))

### Set up cylinder
cyl = cylinder.CylBT()

cyl.num_freq = 50
cyl.freq_lower = 400 + (50 * bi)
cyl.freq_upper = 400 + (50 * (bi+1) )

cyl.cylinder_width = 20.0
cyl.num_cylinders = 5

cyl.feed_spacing = 0.4
cyl.num_feeds = int(100.0 / cyl.feed_spacing)
cyl.maxlength = 28.2


### Set up beam transfer object and generate
bt = beamtransfer.BeamTransfer(teldir, telescope=cyl)
bt.generate_cache()


### Set up and perform KL-transfrom
klt = kltransform.KLTransform(bt, evsubdir='ev/')
klt.generate()


### Estimate the Fisher Matrix for the resulting power spectrum
pse = psestimation.PSEstimation(klt, subdir='ps/')
pse.threshold = 10.0
pse.genbands()

pse.fisher_mpi()

