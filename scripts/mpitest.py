

from mpi4py import MPI

import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()



data = None
if rank == 0:
    print rank, size
    
    arr = np.random.permutation(100)
    data = np.array_split(arr, size)

    
    

data = comm.scatter(data, root=0)

for i in range(size):
    comm.Barrier()

    if i == rank:
        print ("Rank %i. Data: " % rank), data

