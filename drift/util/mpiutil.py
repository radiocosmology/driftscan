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


def typemap(dtype):
    """Map a numpy dtype into an MPI_Datatype.

    Parameters
    ----------
    dtype : np.dtype
        The numpy datatype.

    Returns
    -------
    mpitype : MPI.Datatype
        The MPI.Datatype.
    """
    return MPI.__TypeDict__[np.dtype(dtype).char]


def split_m(n, m):
    """
    Split a range (0, n-1) into m sub-ranges of similar length

    Parameters
    ----------
    n : integer
        Length of range to split.
    m : integer
        Number of subranges to split into.

    Returns
    -------
    num : np.ndarray[m]
        Number in each sub-range
    start : np.ndarray[m]
        Starting of each sub-range.
    end : np.ndarray[m]
        End of each sub-range.

    See Also
    --------
    `split_all`, `split_local`
    """
    base = (n / m)
    rem = n % m

    part = base * np.ones(m, dtype=np.int) + (np.arange(m) < rem).astype(np.int)

    bound = np.cumsum(np.insert(part, 0, 0))

    return np.array([part, bound[:m], bound[1:(m + 1)]])


def split_all(n, comm=MPI.COMM_WORLD):
    """
    Split a range (0, n-1) into sub-ranges for each MPI Process.

    Parameters
    ----------
    n : integer
        Length of range to split.
    comm : MPI Communicator, optional
        MPI Communicator to use (default COMM_WORLD).

    Returns
    -------
    num : np.ndarray[m]
        Number for each rank.
    start : np.ndarray[m]
        Starting of each sub-range on a given rank.
    end : np.ndarray[m]
        End of each sub-range.

    See Also
    --------
    `split_all`, `split_local`
    """
    return split_m(n, comm.size)


def split_local(n, comm=MPI.COMM_WORLD):
    """
    Split a range (0, n-1) into sub-ranges for each MPI Process. This returns
    the parameters only for the current rank.

    Parameters
    ----------
    n : integer
        Length of range to split.
    comm : MPI Communicator, optional
        MPI Communicator to use (default COMM_WORLD).

    Returns
    -------
    num : integer
        Number on this rank.
    start : integer
        Starting of the sub-range for this rank.
    end : integer
        End of rank for this rank.

    See Also
    --------
    `split_all`, `split_local`
    """
    pse = split_all(n, comm=comm)

    return pse[:, comm.rank]


def transpose_blocks(row_array, shape, comm=MPI.COMM_WORLD):
    """
    Take a 2D matrix which is split between processes row-wise and split it
    column wise between processes.

    Parameters
    ----------
    row_array : np.ndarray
        The local section of the global array (split row wise).
    shape : 2-tuple
        The shape of the global array
    comm : MPI communicator
        MPI communicator that array is distributed over. Default is MPI.COMM_WORLD.

    Returns
    -------
    col_array : np.ndarray
        Local section of the global array (split column wise).
    """

    nr = shape[0]
    nc = shape[-1]
    nm = 1 if len(shape) <= 2 else np.prod(shape[1:-1])

    pr, sr, er = split_local(nr, comm=comm) * nm
    pc, sc, ec = split_local(nc, comm=comm)

    par, sar, ear = split_all(nr, comm=comm) * nm
    pac, sac, eac = split_all(nc, comm=comm)

    #print pr, nc, shape, row_array.shape

    row_array = row_array[:nr, ..., :nc].reshape(pr, nc)

    requests_send = []
    requests_recv = []

    recv_buffer = np.empty((nr * nm, pc), dtype=row_array.dtype)

    mpitype = typemap(row_array.dtype)

    # Iterate over all processes row wise
    for ir in range(comm.size):
                
        # Get the start and end of each set of rows
        sir, eir = sar[ir], ear[ir]

        # Iterate over all processes column wise
        for ic in range(comm.size):
                    
            # Get the start and end of each set of columns
            sic, eic = sac[ic], eac[ic]

            # Construct a unique tag
            tag = ir * comm.size + ic

            # Send and receive the messages as non-blocking passes
            if comm.rank == ir:

                # Construct the block to send by cutting out the correct
                # columns
                block = row_array[:, sic:eic].copy()
                #print ir, ic, comm.rank, block.shape

                # Send the message
                request = comm.Isend([block, mpitype], dest=ic, tag=tag)
                requests_send.append([ir, ic, request])

            if comm.rank == ic:

                # Receive the message into the correct set of rows of recv_buffer
                request = comm.Irecv([recv_buffer[sir:eir], mpitype], source=ir, tag=tag)
                requests_recv.append([ir, ic, request])
                #print ir, ic, comm.rank, recv_buffer[sir:eir].shape

    # Wait for all processes to have started their messages
    comm.Barrier()

    # For each node iterate over all sends and wait until completion
    for ir, ic, request in requests_send:

        stat = MPI.Status()

        #try:
        request.Wait(status=stat)
        #except MPI.Exception:
        #    print comm.rank, ir, ic, sar[ir], ear[ir], sac[ic], eac[ic], shape

        if stat.error != MPI.SUCCESS:
            print "**** ERROR in MPI SEND (r: %i c: %i rank: %i) *****" % (ir, ic, comm.rank)


    #print "rank %i: Done waiting on MPI SEND" % comm.rank

    comm.Barrier()

    # For each frequency iterate over all receives and wait until completion
    for ir, ic, request in requests_recv:

        stat = MPI.Status()

        #try:
        request.Wait(status=stat)
        #except MPI.Exception:
        #    print comm.rank, (ir, ic), (ear[ir]-sar[ir], eac[ic]-sac[ic]), shape, recv_buffer[sar[ir]:ear[ir]].shape, recv_buffer.dtype, row_array.dtype

        if stat.error != MPI.SUCCESS:
            print "**** ERROR in MPI RECV (r: %i c: %i rank: %i) *****" % (ir, ir, comm.rank)

    return recv_buffer.reshape(shape[:-1] + (pc,))


def allocate_hdf5_dataset(fname, dsetname, shape, dtype, comm=MPI.COMM_WORLD):
    """Create a hdf5 dataset and return its offset and size.

    The dataset will be created contiguously and immediately allocated,
    however it will not be filled.

    Parameters
    ----------
    fname : string
        Name of the file to write.
    dsetname : string
        Name of the dataset to write (must be at root level).
    shape : tuple
        Shape of the dataset.
    dtype : numpy datatype
        Type of the dataset.
    comm : MPI communicator
        Communicator over which to broadcast results.

    Returns
    -------
    offset : integer
        Offset into the file at which the dataset starts (in bytes).
    size : integer
        Size of the dataset in bytes.

    """

    import h5py

    state = None

    if comm.rank == 0:

        # Create/open file
        f = h5py.File(fname, 'a')

        # Create dataspace and HDF5 datatype
        sp = h5py.h5s.create_simple(shape, shape)
        tp = h5py.h5t.py_create(dtype)

        # Create a new plist and tell it to allocate the space for dataset
        # immediately, but don't fill the file with zeros.
        plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        plist.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)

        # Create the dataset
        dset = h5py.h5d.create(f.id, dsetname, tp, sp, plist)

        # Get the offset of the dataset into the file.
        state = dset.get_offset(), dset.get_storage_size()

        f.close()

    state = comm.bcast(state, root=0)

    return state


def lock_and_write_buffer(obj, fname, offset, size):
    """Write the contents of a buffer to disk at a given offset, and explicitly
    lock the region of the file whilst doing so.

    Parameters
    ----------
    obj : buffer
        Data to write to disk.
    fname : string
        Filename to write.
    offset : integer
        Offset into the file to start writing at.
    size : integer
        Size of the region to write to (and lock).
    """
    import os
    import os.fcntl as fcntl

    buf = buffer(obj)

    if len(buf) > size:
        raise Exception("Size doesn't match array length.")

    fd = os.open(fname, os.O_RDRW | os.O_CREAT)

    fcntl.lockf(fd, fcntl.LOCK_EX, size, offset, os.SEEK_SET)

    nb = os.write(fd, buf)

    if nb != len(buf):
        raise Exception("Something funny happened with the reading.")

    fcntl.lockf(fd, fcntl.LOCK_UN)

    os.close(fd)


def parallel_rows_write_hdf5(fname, dsetname, local_data, shape, comm=MPI.COMM_WORLD):
    """Write out array (distributed across processes row wise) into a HDF5 in parallel.

    """
    offset, size = allocate_hdf5_dataset(fname, dsetname, shape, local_data.dtype, comm=comm)

    lr, sr, er = split_local(shape[0], comm=comm)

    nc = np.prod(shape[1:])

    lock_and_write_buffer(local_data, fname, offset + sr * nc, lr * nc)


