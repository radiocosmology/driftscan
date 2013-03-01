import warnings

import numpy as np

_rank = 0
_size = 1
_comm = None
world = None

rank0 = True

## Try to setup MPI and get the comm, rank and size.
## If not they should end up as rank=0, size=1.
try:
    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    world = _comm
    
    rank = _comm.Get_rank()
    size = _comm.Get_size()

    _rank = rank if rank else 0
    _size = size if size else 1

    if rank:
        print "MPI process %i of %i." % (_rank, _size)

    rank0 = True if _rank == 0 else False
    
except ImportError:
    warnings.warn("Warning: mpi4py not installed.")


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
    
def parallel_map(func, glist):
    """Apply a parallel map using MPI.

    Should be called collectively on the same list. All ranks return the full
    set of results.

    Parameters
    ----------
    func : function
        Function to apply.

    glist : list
        List of map over. Must be globally defined.

    Returns
    -------
    results : list
        Global list of results.
    """

    # Synchronize
    barrier()

    # If we're only on a single node, then just perform without MPI
    if _size == 1 and _rank == 0:
        return [func(item) for item in glist]

    # Pair up each list item with its position.
    zlist = list(enumerate(glist))

    # Partition list based on MPI rank
    llist = partition_list_mpi(zlist)

    # Operate on sublist
    flist = [(ind, func(item)) for ind, item in llist]

    # Gather all results onto all ranks
    rlist = _comm.allgather(flist)

    # Flatten the list of results
    flatlist = [item for sublist in rlist for item in sublist]

    # Sort into original order
    sortlist = sorted(flatlist, key=(lambda item: item[0]))

    # Synchronize
    barrier()

    # Zip to remove indices and extract the return values into a list
    return list(zip(*sortlist)[1])

