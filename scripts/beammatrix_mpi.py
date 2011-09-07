

from mpi4py import MPI
import numpy as np

import h5py

from cylsim import cylinder


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()



## Cylinder object

cyl = cylinder.UnpolarisedCylinderTelescope()

nbase = cyl.baselines.shape[0]



fbase = "beamtrans_alpha_%0"+repr(int(np.ceil(np.log10(nbase))))+"d.npy"


data = None
if rank == 0:
    print rank, size
    
    baseind = np.random.permutation(nbase)
    data = np.array_split(baseind, size)

    
    

local_bi = comm.scatter(data, root=0)

for bi in local_bi:
    btrans = cyl.transfer_for_baseline(bi)

    fname = (fbase % bi)

    np.save(fname, btrans)


