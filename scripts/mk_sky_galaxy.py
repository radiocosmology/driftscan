import sys

import numpy as np

import h5py

from cylsim import skysim



nside = int(sys.argv[1])
freq_lower = float(sys.argv[2])
freq_upper = float(sys.argv[3])
nfreq = int(sys.argv[4])

mapname = sys.argv[5]



cs = skysim.c_syn(nside, np.linspace(freq_lower, freq_upper, nfreq))


f = h5py.File(mapname, 'w')

f.create_dataset("map", data=cs)

f.close()
