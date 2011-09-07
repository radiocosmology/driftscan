


import h5py

import numpy as np

from cylsim import cylinder

import sys
import os.path
import glob

root1 = sys.argv[1]

root1 = os.path.expanduser(root1)

freqfiles = glob.glob(root1 + "*freq_*.hdf5")

f = h5py.File(freqfiles[0], 'r')

lmax, mmax = cyl.max_lm()

fmax = cyl.frequencies.shape[0]

mfmt = "%0"+repr(int(np.ceil(np.log10(mmax+1))))+"d"
ffmt = "%0"+repr(int(np.ceil(np.log10(fmax+1))))+"d"

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()



data = None
if rank == 0:
    print "Distributing across %i processes" % size
    
    m_ind = np.random.permutation(mmax)
    data = np.array_split(m_ind, size)

    
    

local_mi = comm.scatter(data, root=0)

for mi in local_mi:

    fname = fbase % mi
    print 'm index %i. Creating file: %s' % (fi, fname)
    
    #f = h5py.File(fname, 'w')

    #fgrp = f.create_group('freq_section')

    for freqfile in freqfiles:
        ff = h5py.File(freqfile, 'r')
        
        fi = ff.attrs['frequency_index']
        
        #mstr = "m_section/" + (mfmt % mi)

        #dset = fgrp.create_dataset(ff[mstr], (ffmt % fi), compression='gzip')
        #dset.attrs['frequency_index'] = fi

        print "Creating section m %i, f %i" % (mi, fi)

        ff.close()

    #f.attrs['baselines'] = cyl.baselines
    #f.attrs['m'] = m
    #f.attrs['frequencies'] = cyl.frequencies
        
    #f.close()

