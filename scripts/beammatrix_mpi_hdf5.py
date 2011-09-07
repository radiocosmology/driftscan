

from mpi4py import MPI
import numpy as np

import h5py

from cylsim import cylinder

import pickle

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()



## Cylinder object

cyl = cylinder.UnpolarisedCylinderTelescope()

nfreq = cyl.frequencies.shape[0]
nbase = cyl.baselines.shape[0]

lmax, mmax = cyl.max_lm()


fbase = "data/beamtrans_freq_%0"+repr(int(np.ceil(np.log10(nfreq+1))))+"d.hdf5"
mfmt = "%0"+repr(int(np.ceil(np.log10(mmax+1))))+"d"

data = None
if rank == 0:
    print "Distributing across %i processes" % size
    
    freqind = np.random.permutation(nfreq)
    data = np.array_split(freqind, size)

    
    

local_fi = comm.scatter(data, root=0)

for fi in local_fi:

    fname = fbase % fi
    print 'Frequency index %i. Creating file: %s' % (fi, fname)
    
    f = h5py.File(fname, 'w')

    mgrp = f.create_group('m_section')
    btrans = cyl.transfer_for_frequency(fi)

    f.attrs['baselines'] = cyl.baselines
    f.attrs['frequency_index'] = fi
    f.attrs['frequency'] = cyl.frequencies[fi]

    for m in range(mmax):
        
        dset = mgrp.create_dataset((mfmt % m), data=btrans[m,:,:], compression='gzip')
        dset.attrs['m'] = m
        
    f.close()

