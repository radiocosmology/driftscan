
import numpy as np

_rank = 0
_size = 1
_comm = None

rank0 = True

## Try to setup MPI and get the comm, rank and size.
## If not they should end up as rank=0, size=1.
try:
    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    
    rank = _comm.Get_rank()
    size = _comm.Get_size()

    _rank = rank if rank else 0
    _size = size if size else 1

    rank0 = True if _rank == 0 else False
    
except ImportError:
    pass


def partition_list_alternate(full_list, i, n):
    """Partition a list into `n` pieces. Return the `i`th partition."""
    return full_list[i::n]


def partition_list_mpi(full_list):
    """Return the partition of a list specific to the current MPI process."""
    return partition_list_alternate(full_list, _rank, _size)


def mpirange(*args):
    """An MPI aware version of `range`, each process gets its own sub section.
    """
    full_list = range(*args)
    
    #if alternate:
    return partition_list_alternate(full_list, _rank, _size)
    #else:
    #    return np.array_split(full_list, _size)[rank_]


def barrier():
    if _size > 1:
        _comm.Barrier()
    
