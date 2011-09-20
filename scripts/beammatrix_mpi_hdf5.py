

from mpi4py import MPI
import numpy as np

import h5py

from cylsim import cylinder

import pickle

import argparse
import os

parser = argparse.ArgumentParser(description='MPI program to generate beam matrices frequency by frequency.')
parser.add_argument('rootdir', help='Root directory to create files in.')
parser.add_argument('filestem', default='', help='Prefix to add created files.', nargs='?')

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Setup Cylinder object

cyl = cylinder.UnpolarisedCylinderTelescope()

## Cylinder must be finalised by here

  
root = args.rootdir + "/" + args.filestem + "beammatrix"



nfreq = cyl.frequencies.shape[0]
nbase = cyl.baselines.shape[0]

lmax, mmax = cyl.max_lm()

cylpickle = pickle.dumps(cyl)



fbase = root + "_freq_%0"+repr(int(np.ceil(np.log10(nfreq+1))))+"d.hdf5"
mfmt = "%+0"+repr(int(np.ceil(np.log10(mmax+1))) + 1)+"d"






data = None
if rank == 0:
    print "== Distributing across %i processes ==\n\n" % size

    print "Calculating %d frequencies each with mmax = %d" % (nfreq, mmax)
    
    freqind = np.random.permutation(nfreq)
    data = np.array_split(freqind, size)

    # Create directory if required
    if not os.path.exists(args.rootdir):
        os.makedirs(args.rootdir)

    ## Write out a copy of the telescope config
    with open(root+"_cylobj.pickle", 'w') as f:
        pickle.dump(cyl, f)

        
comm.Barrier()
    

local_fi = comm.scatter(data, root=0)

for fi in local_fi:

    fname = fbase % fi
    print 'Frequency index %i. Creating file: %s' % (fi, fname)
    
    f = h5py.File(fname, 'w')

    mgrp = f.create_group('m_section')
    btrans = cyl.transfer_for_frequency(fi)

    f.attrs['baselines'] = cyl.baselines
    f.attrs['baseline_indices'] = np.arange(cyl.baselines.shape[0])
    f.attrs['frequency_index'] = fi
    f.attrs['frequency'] = cyl.frequencies[fi]
    f.attrs['cylobj'] = cylpickle

    for m in range(-mmax, mmax+1):

        # Symmetrise beam such that it only responds to real skies.
        #bsym = 0.5*(btrans[m,:,:] + (-1)**m * btrans[-m,:,:])
        
        dset = mgrp.create_dataset((mfmt % m), data=btrans[m,:,:], compression='gzip')
        dset.attrs['m'] = m
        
    f.close()

