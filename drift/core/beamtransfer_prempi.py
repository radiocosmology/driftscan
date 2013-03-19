import pickle
import os
import time

import numpy as np
import scipy.linalg as la
import h5py

import mpiutil
import util
import blockla


def svd_gen(A, *args, **kwargs):
    """Find the inverse of A.

    If a standard matrix inverse has issues try using the pseudo-inverse.

    Parameters
    ----------
    A : np.ndarray
        Matrix to invert.
        
    Returns
    -------
    inv : np.ndarray
    """
    try:
        res = la.svd(A, *args, **kwargs)
    except la.LinAlgError:
        sv = la.svdvals(A)
        At = A + sv[0] * 1e-11 * np.eye(A.shape[0], A.shape[1])
        res = la.svd(At, *args, **kwargs)
        print "Matrix SVD did not converge. Regularised."

    return res




class BeamTransfer(object):
    """A class for reading and writing Beam Transfer matrices from disk. In
    addition this provides methods for projecting vectors and matrices between
    the sky and the telescope basis.
    """



    _mem_switch = 3.0 # Rough chunks (in GB) to divide calculation into.

    svcut = 1e-6


    #====== Properties giving internal filenames =======

    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.directory + "/telescopeobject.pickle"

    def _mdir(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self.directory + "/beam_m/" + util.natpattern(self.telescope.mmax)
        return pat % abs(mi)

    def _mfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + "/" + ('pos' if mi >= 0 else 'neg') + '.hdf5'

    def _fdir(self, fi):
        # Pattern to form the `freq` ordered file.
        pat = self.directory + "/beam_f/" + util.natpattern(self.telescope.nfreq)
        return pat % fi

    def _ffile(self, fi):
        # Pattern to form the `freq` ordered file.
        return self._fdir(fi) + "/beam.hdf5"

    def _svdfile(self, mi):
        # Pattern to form the `m` ordered file.

        # Pattern to form the `m` ordered file.
        pat = self.directory + "/beam_m/" + util.natpattern(self.telescope.mmax) + "/svd.hdf5"

        return pat % mi

    #===================================================


    @property
    def _telescope_pickle(self):
        # The pickled telescope object
        return pickle.dumps(self.telescope)

    def __init__(self, directory, telescope=None):

        self.directory = directory
        self.telescope = telescope

        # Create directory if required
        if mpiutil.rank0 and not os.path.exists(directory):
            os.makedirs(directory)

        mpiutil.barrier()

        if self.telescope == None:
            print "Attempting to read telescope from disk..."

            try:
                f = open(self._picklefile, 'r')
                self.telescope = pickle.load(f)
            except IOError, UnpicklingError:
                raise Exception("Could not load Telescope object from disk.")


    #===================================================



    #====== Loading m-order beams ======================

    def _load_beam_m(self, mi, fi=None):
        ## Read in beam from disk
        mfile = h5py.File(self._mfile(mi), 'r')

        # If fi is None, return all frequency blocks. Otherwise just the one requested.
        if fi is None:
            beam = mfile['beam_m'][:]
        else:
            beam = mfile['beam_m'][fi][:]
        
        mfile.close()

        return beam


    @util.cache_last
    def beam_m(self, mi, fi=None, single=False):
        """Fetch the beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.
        single : boolean, optional
            When set, fetch only the uncombined beam transfer (that is only
            positive or negative m). Default is False.

        Returns
        -------
        beam : np.ndarray (nfreq, 2, npairs, npol_tel, npol_sky, lmax+1)
            If `single` is set, shape (nfreq, npairs, npol_tel, npol_sky, lmax+1)
        """
        if single or self.telescope.positive_m_only:
            return self._load_beam_m(mi)

        bp = self._load_beam_m(mi, fi=fi)
        bm = (-1)**mi * self._load_beam_m(-mi, fi=fi).conj()

        # Zero out m=0 (for negative) to avoid double counting
        if mi == 0:
            bm[:] = 0.0

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            bc = np.empty(bp.shape[:1] + (2,) + bp.shape[1:], dtype=bp.dtype)
            bc[:, 0] = bp
            bc[:, 1] = bm
        else:
            bc = np.empty((2,) + bp.shape, dtype=bp.dtype)
            bc[0] = bp
            bc[1] = bm

        return bc

    #===================================================


    #====== Loading freq-ordered beams =================

    @util.cache_last
    def _load_beam_freq(self, fi, fullm=False):
        
        tel = self.telescope
        mside = 2 * tel.lmax + 1 if fullm else 2 * tel.mmax + 1
        
        ffile = h5py.File(self._ffile(fi), 'r')
        beamf = ffile['beam_freq'][:]
        ffile.close()
        
        if fullm:
            beamt = np.zeros(beamf.shape[:-1] + (2*tel.lmax+1,), dtype=np.complex128)

            for mi in range(-tel.mmax, tel.mmax + 1):
                beamt[..., mi] = beamf[..., mi]
        
            beamf = beamt
        
        return beamf


    @util.cache_last
    def beam_freq(self, fi, fullm=False, single=False):
        """Fetch the beam transfer matrix for a given frequency.

        Parameters
        ----------
        fi : integer
            Frequency to fetch.
        fullm : boolean, optional
            Pad out m-modes such that we have :math:`mmax = 2*lmax-1`. Useful
            for projecting around a_lm's. Default is False.
        single : boolean, optional
            When set, fetch only the uncombined beam transfers (that is only
            positive or negative m). Default is False.

        Returns
        -------
        beam : np.ndarray
        """
        bf = self._load_beam_freq(fi, fullm)

        if single or self.telescope.positive_m_only:
            return bf

        mside = (bf.shape[-1] + 1) / 2

        bfc = np.zeros((mside, 2) + bf.shape[:-1], dtype=bf.dtype)

        bfc[0, 0] = bf[..., 0]

        for mi in range(1, mside):
            bfc[mi, 0] = bf[..., mi]
            bfc[mi, 1] = (-1)**mi * bf[..., -mi].conj()

        return bfc

    #===================================================



    #====== Pseudo-inverse beams =======================

    noise_weight = True

    @util.cache_last
    def invbeam_m(self, mi):
        """Pseudo-inverse of the beam (for a given m).

        Uses the Moore-Penrose Pseudo-inverse as the optimal inverse for
        reconstructing the data. No `single` option as this only makes sense
        when combined.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Returns
        -------
        invbeam : np.ndarray (nfreq, npol_sky, lmax+1, 2, npairs, npol_tel)
        """

        beam = self.beam_m(mi)

        if self.noise_weight:
            noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), 0).flatten()**(-0.5)
            beam = beam * noisew[:, np.newaxis, np.newaxis, np.newaxis]

        beam = beam.reshape((self.nfreq, self.ntel, self.nsky))

        ibeam = blockla.pinv_dm(beam, rcond=1e-6)

        if self.noise_weight:
            # Reshape to make it easy to multiply baselines by noise level
            ibeam = ibeam.reshape((-1, self.telescope.npairs, self.telescope.num_pol_telescope))
            ibeam = ibeam * noisew[:, np.newaxis]

        shape = (self.nfreq, self.telescope.num_pol_sky,
                 self.telescope.lmax + 1, self.ntel,
                 self.telescope.num_pol_telescope)
        
        return ibeam.reshape(shape)

    #===================================================




    #====== SVD Beam loading ===========================

    @util.cache_last
    def beam_svd(self, mi, fi=None):
        """Fetch the SVD beam transfer matrix (S V^H) for a given m. This SVD beam
        transfer projects from the sky into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len, npol_sky, lmax+1)
        """
        
        svdfile = h5py.File(self._svdfile(mi), 'r')

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            bs = svdfile['beam_svd'][:]
        else:
            bs = svdfile['beam_svd'][fi][:]
            
        svdfile.close()

        return bs


    @util.cache_last
    def beam_ut(self, mi, fi=None):
        """Fetch the SVD beam transfer matrix (U^H) for a given m. This SVD beam
        transfer projects from the telescope space into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len, ntel)
        """
        
        svdfile = h5py.File(self._svdfile(mi), 'r')

        # Required array shape depends on whether we are returning all frequency blocks or not.
        if fi is None:
            bs = svdfile['beam_ut'][:]
        else:
            bs = svdfile['beam_ut'][fi][:]
            
        svdfile.close()

        return bs


    @util.cache_last
    def beam_singularvalues(self, mi):
        """Fetch the beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.
        single : boolean, optional
            When set, fetch only the uncombined beam transfer (that is only
            positive or negative m). Default is False.

        Returns
        -------
        beam : np.ndarray (nfreq, 2, npairs, npol_tel, npol_sky, lmax+1)
            If `single` is set, shape (nfreq, npairs, npol_tel, npol_sky, lmax+1)
        """
        
        svdfile = h5py.File(self._svdfile(mi), 'r')
        sv = svdfile['singularvalues'][:]
        svdfile.close()

        return sv






    #===================================================




    #====== Generation of all the cache files ==========

    def generate(self, regen=False):
        """Save out all beam transfer matrices to disk.

        Parameters
        ----------
        regen : boolean, optional
            Force regeneration even if cache files exist (default: False).
        """

        st = time.time()

        self._generate_dirs()
        self._generate_ffiles(regen)
        self._generate_mfiles(regen)
        self._generate_svdfiles(regen)


        # Save pickled telescope object
        if mpiutil.rank0:
            with open(self._picklefile, 'w') as f:
                print "=== Saving Telescope object. ==="
                pickle.dump(self.telescope, f)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:
            print "***** Beam generation time: %f" % (et - st)


    generate_cache = generate # For compatibility with old code


    def _generate_dirs(self):


        if mpiutil.rank0:

            # Create main directory for beamtransfer
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            # Create directories for storing frequency ordered beams
            for fi in range(self.nfreq):
                dirname = self._fdir(fi)

                if not os.path.exists(dirname):
                    os.makedirs(dirname)

            # Create directories for m beams and svd files.
            for mi in range(self.telescope.mmax + 1):
                dirname = self._mdir(mi)

                # Create directory if required
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

        mpiutil.barrier()


    def _generate_ffiles(self, regen=False):

        # For each frequency, create the HDF5 file, and write in each `m` as a
        # seperate compressed dataset. Use MPI if available. 
        for fi in mpiutil.mpirange(self.nfreq):

            if os.path.exists(self._ffile(fi)) and not regen:
                print ("f index %i. File: %s exists. Skipping..." %
                       (fi, (self._ffile(fi))))
                continue
            else:
                print ('f index %i. Creating file: %s' %
                       (fi, (self._ffile(fi))))

            f = h5py.File(self._ffile(fi), 'w')

            # Set a few useful attributes.
            f.attrs['baselines'] = self.telescope.baselines
            f.attrs['baseline_indices'] = np.arange(self.telescope.npairs)
            f.attrs['frequency_index'] = fi
            f.attrs['frequency'] = self.telescope.frequencies[fi]
            f.attrs['cylobj'] = self._telescope_pickle

            dsize = (self.telescope.nbase, self.telescope.num_pol_telescope,
                     self.telescope.num_pol_sky, self.telescope.lmax+1, 2*self.telescope.mmax+1)

            csize = (10, self.telescope.num_pol_telescope,
                     self.telescope.num_pol_sky, self.telescope.lmax+1, 1)

            dset = f.create_dataset('beam_freq', dsize, chunks=csize, compression='lzf', dtype=np.complex128)

            # Divide into roughly 5 GB chunks
            nsections = np.ceil(np.prod(dsize) * 16.0 / 2**30.0 / self._mem_switch)

            print "Dividing calculation of %f GB array into %i sections." % (np.prod(dsize) * 16.0 / 2**30.0, nsections)

            b_sec = np.array_split(np.arange(self.telescope.npairs, dtype=np.int), nsections)
            f_sec = np.array_split(fi * np.ones(self.telescope.npairs, dtype=np.int), nsections)

            # Iterate over each section, generating transfers and save them.
            for b_ind, f_ind in zip(b_sec, f_sec):
                tarray = self.telescope.transfer_matrices(b_ind, f_ind)
                dset[(b_ind[0]):(b_ind[-1]+1), ..., :(self.telescope.mmax+1)] = tarray[..., :(self.telescope.mmax+1)]
                dset[(b_ind[0]):(b_ind[-1]+1), ..., (-self.telescope.mmax):]  = tarray[..., (-self.telescope.mmax):]
                del tarray

            f.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()


    def _generate_mfiles(self, regen=False):

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available. 
        for mi in mpiutil.mpirange(-self.telescope.mmax,
                                   self.telescope.mmax + 1):

            if os.path.exists(self._mfile(mi)) and not regen:
                print ("m index %i. File: %s exists. Skipping..." %
                       (mi, (self._mfile(mi))))
                continue
            else:
                print 'm index %i. Creating file: %s' % (mi, self._mfile(mi))

            ## Create hdf5 file for each m-mode
            f = h5py.File(self._mfile(mi), 'w')

            dsize = (self.telescope.nfreq, self.telescope.nbase,
                     self.telescope.num_pol_telescope, self.telescope.num_pol_sky, self.telescope.lmax+1)

            csize = (1, 10, self.telescope.num_pol_telescope,
                     self.telescope.num_pol_sky, self.telescope.lmax+1)

            dset = f.create_dataset('beam_m', dsize, chunks=csize, compression='lzf', dtype=np.complex128)


            ## For each frequency read in the current m-mode 
            ## and copy into file.
            for fi in np.arange(self.telescope.nfreq):

                ff = h5py.File(self._ffile(fi), 'r')

                # Check frequency is what we expect.
                if fi != ff.attrs['frequency_index']:
                    raise Exception("Bork.")
        
                dset[fi] = ff['beam_freq'][..., mi]

                ff.close()
                
            # Write a few useful attributes.
            f.attrs['baselines'] = self.telescope.baselines
            f.attrs['m'] = mi
            f.attrs['frequencies'] = self.telescope.frequencies
            f.attrs['cylobj'] = self._telescope_pickle
            
            f.close()



    def _generate_mfiles_mpi(self, regen=False):

        st = time.time()

        nproc = mpiutil.size
        nf = self.telescope.nfreq
        nm = self.telescope.mmax + 1
        ranks = np.arange(nproc)

        # Work out a mapping between frequency and rank
        freq_per_rank = (nf / nproc) + (ranks < (nf % nproc)).astype(np.int)
        freq_rank_map = np.repeat(ranks, freq_per_rank)

        # Mapping between m's and rank
        m_per_rank = (nm / nproc) + (ranks < (nm % nproc)).astype(np.int)
        m_rank_map = np.repeat(ranks, m_per_rank)

        # Calculate the number of baselines to deal with at any one time. Aim
        # to have a maximum of 4 GB in memory at any one time
        blsize = (freq_per_rank.max() * self.telescope.num_pol_telescope *
            self.telescope.num_pol_sky * (self.telescope.lmax+1) *
            (2*self.telescope.mmax+1) * 16.0)

        num_bl_per_chunk = int(3e9 / blsize) # Number of baselines to process in each chunk
        num_chunks = int(self.telescope.nbase / num_bl_per_chunk) + 1

        if mpiutil.rank0:
            print "====================================================="
            print "  Processing %i frequencies and %i m's" % (nf, nm)
            print "  Split into groups of %i and %i respectively" % (freq_per_rank[0], m_per_rank[0])
            print
            print "  %i groups of %i baselines (size %f GB)" % (num_chunks, num_bl_per_chunk, blsize * num_bl_per_chunk / 2**30.0)
            print "====================================================="




        # Iterate over all m's and create the hdf5 files we will write into.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._mfile(mi) + '.mpi') and not regen:
                print ("m index %i. File: %s exists. Skipping..." %
                       (mi, (self._mfile(mi) + '.mpi')))
                continue
            else:
                print 'm index %i. Creating file: %s' % (mi, self._mfile(mi) + '.mpi')

            ## Create hdf5 file for each m-mode
            f = h5py.File(self._mfile(mi) + '.mpi', 'w')

            dsize = (self.telescope.nfreq, 2, self.telescope.nbase,
                     self.telescope.num_pol_telescope, self.telescope.num_pol_sky, self.telescope.lmax+1)

            csize = (1, 2, 10, self.telescope.num_pol_telescope,
                     self.telescope.num_pol_sky, self.telescope.lmax+1)

            dset = f.create_dataset('beam_m', dsize, chunks=csize, compression='lzf', dtype=np.complex128)

            f.close()


        # Iterate over all chunks performing a cycle of: read frequency data, transpose to m-order data, write m-data.
        for ci in range(num_chunks):

            if mpiutil.rank0:
                print
                print "============================================="
                print "    Starting chunk %i of %i" % (ci, num_chunks)
                print "============================================="
                print


            blstart = ci * num_bl_per_chunk
            blend = min((ci+1)*num_bl_per_chunk, self.telescope.nbase)

            # Load the current chunk of baselines from a frequency file into
            # memory (only load those contained on this rank)
            def _load_fchunk(fi):

                if mpiutil.rank == freq_rank_map[fi]:
                    ff = h5py.File(self._ffile(fi), 'r')
                    fchunk = ff['beam_freq'][blstart:blend][:]
                    ff.close()
                else:
                    fchunk = None

                return fchunk

            # Make the current baseline chunk of the m-ordered array (only
            # create those contained on this rank)
            def _mk_marray(mi):

                if mpiutil.rank == m_rank_map[mi]:
                    marray = np.zeros((self.telescope.nfreq, 2, (blend-blstart),
                       self.telescope.num_pol_telescope, self.telescope.num_pol_sky, self.telescope.lmax+1), dtype=np.complex128)
                else:
                    marray = None

                return marray


            # Create lists containing the relevant frequency and m arrays for this rank.
            freq_chunks = [ _load_fchunk(fi) for fi in range(nf) ]
            m_arrays = [ _mk_marray(mi) for mi in range(nm) ]


            # Iterate over all frequencies and m's passing the relevant parts
            # of the frequency matrices into m arrays. Though this might seem
            # to be more naturally done by scatter-gather, the variable
            # lengths make it difficult.

            # List to contain all the current MPI_Requests for the sends
            requests = []

            for fi in range(nf):
                f_rank = freq_rank_map[fi]
                
                if f_rank == mpiutil.rank:
                    print "Passing block from freq %i (rank %i)" % (fi, f_rank)

                for mi in range(self.telescope.mmax + 1):
                    
                    m_rank = m_rank_map[mi]

                    # Try and create a unique tag for this combination of (fi, mi)
                    tag = 2*(fi * max(nm, nf) + mi)

                    # Send and receive the messages as non-blocking passes
                    if mpiutil.rank == f_rank:
                        pos = freq_chunks[fi][..., mi].copy()
                        neg = ((-1)**mi * freq_chunks[fi][..., -mi]).conj().copy() if mi > 0 else np.zeros_like(pos)
                        #print "Passing f-block %i to m-block %i (rank %i to %i)" % (fi, mi, f_rank, m_rank)
                        requestp = mpiutil.world.Isend([pos, mpiutil.MPI.COMPLEX16], dest=m_rank, tag=tag)
                        requestm = mpiutil.world.Isend([neg, mpiutil.MPI.COMPLEX16], dest=m_rank, tag=(tag+1))


                        requests.append([requestp, requestm])

                    if mpiutil.rank == m_rank:
                        mpiutil.world.Irecv([m_arrays[mi][fi, 0], mpiutil.MPI.COMPLEX16], source=f_rank, tag=tag)
                        mpiutil.world.Irecv([m_arrays[mi][fi, 1], mpiutil.MPI.COMPLEX16], source=f_rank, tag=(tag+1))


            # For each frequency iterate over all sends and wait until completion
            for fi in range(nf):

                if freq_rank_map[fi] == mpiutil.rank:
                    
                    for request, mi in zip(requests, range(-self.telescope.mmax, self.telescope.mmax + 1)):
                        #print "Waiting on transfer f %i to m %i (rank %i to %i)" % (fi, mi, freq_rank_map[fi], m_rank_map[mi])
                        request[0].Wait()
                        request[1].Wait()

                    print "Done waiting on sends from freq %i (rank %i)" % (fi, freq_rank_map[fi])

            # Force synchronization
            mpiutil.barrier()

            # Write out the current set of chunks into the m-files.
            for mi in range(self.telescope.mmax + 1):

                if mpiutil.rank == m_rank_map[mi]:
                    mfile = h5py.File(self._mfile(mi) + '.mpi', 'r+')

                    mfile['beam_m'][:, blstart:blend] = m_arrays[mi]

                    mfile.close()

                    print "Done writing chunk to m %i (rank %i)" % (mi, m_rank_map[mi])

            # Delete the local frequency and m ordered
            # sections. Otherwise we run out of memory on the next
            # iteration as there is a brief moment where the chunks
            # exist for both old and new iterations.
            del freq_chunks
            del m_arrays

            mpiutil.barrier()

            



        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available. 
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            mfile = h5py.File(self._mfile(mi) + '.mpi', 'r+')

            # Write a few useful attributes.
            mfile.attrs['baselines'] = self.telescope.baselines
            mfile.attrs['m'] = mi
            mfile.attrs['frequencies'] = self.telescope.frequencies
            mfile.attrs['cylobj'] = self._telescope_pickle
            
            mfile.close()

        mpiutil.barrier()

        et = time.time()
        if mpiutil.rank0:
            print "=== MPI transpose took %f s ===" % (et - st)

    def _compare_m_files(self):

        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            f0p = h5py.File(self._mfile(mi), 'r')
            f0n = h5py.File(self._mfile(-mi), 'r')

            f1 = h5py.File(self._mfile(mi) + '.mpi', 'r')


            bp = f0p['beam_m'][:]
            bn = ((-1)**mi * f0n['beam_m'][:]).conj()

            if (bp == f1['beam_m'][:, 0]).all():
                print "pos m: %i identical" % mi
            else:
                print "****** pos m: %i DIFFERENT *******" % mi

            if (bn == f1['beam_m'][:, 1]).all():
                print "neg m: %i identical" % mi
            else:
                print "****** neg m: %i DIFFERENT *******" % mi


            f0p.close()
            f0n.close()
            f1.close()





    def _generate_svdfiles(self, regen=False):

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available. 
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print ("m index %i. File: %s exists. Skipping..." %
                       (mi, (self._svdfile(mi))))
                continue
            else:
                print 'm index %i. Creating SVD file: %s' % (mi, self._svdfile(mi))

            # Open positive and negative m beams for reading.
            fp = h5py.File(self._mfile(mi),  'r')
            fm = h5py.File(self._mfile(-mi), 'r')

            # Open file to write SVD results into.
            fs = h5py.File(self._svdfile(mi), 'w')

            # The size of the SVD output matrices
            svd_len = min(self.telescope.lmax+1, self.ntel)

            # Create a chunked dataset for writing the SVD beam matrix into.
            dsize_bsvd = (self.telescope.nfreq, svd_len, self.telescope.num_pol_sky, self.telescope.lmax+1)
            csize_bsvd = (1, 10, self.telescope.num_pol_sky, self.telescope.lmax+1)
            dset_bsvd = fs.create_dataset('beam_svd', dsize_bsvd, chunks=csize_bsvd, compression='lzf', dtype=np.complex128)

            # Create a chunked dataset for the stokes T U-matrix (left evecs)
            dsize_ut = (self.telescope.nfreq, svd_len, self.ntel)
            csize_ut = (1, 10, self.ntel)
            dset_ut  = fs.create_dataset('beam_ut', dsize_ut, chunks=csize_ut, compression='lzf', dtype=np.complex128)

            # Create a dataset for the singular values.
            dsize_sig = (self.telescope.nfreq, svd_len)
            dset_sig  = fs.create_dataset('singularvalues', dsize_sig, dtype=np.float64)

            ## For each frequency in the m-files read in the block, SVD it,
            ## and construct the new beam matrix, and save.
            for fi in np.arange(self.telescope.nfreq):

                # Read the positive and negative m beams, and combine into one.
                bfp = fp['beam_m'][fi][:]
                bfm = (-1)**mi * fm['beam_m'][fi][:].conj() if mi > 0 else np.zeros_like(bfp)
                bf = np.array([bfp, bfm]).reshape(self.ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1)

                noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), fi).flatten()**(-0.5)
                noisew = np.concatenate([noisew, noisew])
                bf = bf * noisew[:, np.newaxis, np.newaxis]

                # Get the T-mode only beam matrix
                bft = bf[:, 0, :]

                # Perform the SVD to find the left evecs
                u, sig, v = svd_gen(bft, full_matrices=False)
                u = u.T.conj() # We only need u^H so just keep that.

                # Save out the evecs (for transforming from the telescope frame into the SVD basis)
                dset_ut[fi] = (u * noisew[np.newaxis, :])

                # Save out the modified beam matrix (for mapping from the sky into the SVD basis)
                dset_bsvd[fi] = np.dot(u, bf.reshape(self.ntel, -1)).reshape(svd_len, self.telescope.num_pol_sky,
                                                                             self.telescope.lmax + 1)

                # Save out the singular values for each block
                dset_sig[fi] = sig

                
            # Write a few useful attributes.
            fs.attrs['baselines'] = self.telescope.baselines
            fs.attrs['m'] = mi
            fs.attrs['frequencies'] = self.telescope.frequencies
            fs.attrs['cylobj'] = self._telescope_pickle
            
            fs.close()
            fp.close()
            fm.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        
    #===================================================









    
    #====== Projection between spaces ==================

    def project_vector_sky_to_telescope(self, mi, vec):
        """Project a vector from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, npol, lmax+1]

        Returns
        -------
        tvec : np.ndarray
            Telescope vector to return.
        """

        beam = self.beam_m(mi).reshape((self.nfreq, self.ntel, self.nsky))
        
        vecf = np.zeros((self.nfreq, self.ntel), dtype=np.complex128)

        for fi in range(self.nfreq):
            vecf[fi] = np.dot(beam[fi], vec[..., fi, :].reshape(self.nsky))

        return vecf

    project_vector_forward = project_vector_sky_to_telescope
    

    def project_vector_telescope_to_sky(self, mi, vec):
        """Invert a vector from the telescope space onto the sky. This is the
        map-making process.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [freq, baseline, polarisation]

        Returns
        -------
        tvec : np.ndarray
            Sky vector to return.
        """

        ibeam = self.invbeam_m(mi).reshape((self.nfreq, self.nsky, self.ntel))
        
        vecb = np.zeros((self.nfreq, self.nsky), dtype=np.complex128)
        vec = vec.reshape((self.nfreq, self.ntel))

        for fi in range(self.nfreq):
            vecb[fi] = np.dot(ibeam[fi], vec[fi, :].reshape(self.ntel))

        return vecb.reshape((self.nfreq, self.telescope.num_pol_sky,
                             self.telescope.lmax + 1))

    project_vector_backward = project_vector_telescope_to_sky


    def project_vector_backward_dirty(self, mi, vec):

        dbeam = self.beam_m(mi).reshape((self.nfreq, self.ntel, self.nsky))
        dbeam = dbeam.transpose((0, 2, 1)).conj()
        
        vecb = np.zeros((self.nfreq, self.nsky), dtype=np.complex128)
        vec = vec.reshape((self.nfreq, self.ntel))

        for fi in range(self.nfreq):
            norm = np.dot(dbeam[fi].T.conj(), dbeam[fi]).diagonal()
            norm = np.where(norm < 1e-6, 0.0, 1.0 / norm)
            #norm = np.dot(dbeam[fi], dbeam[fi].T.conj()).diagonal()
            #norm = np.where(np.logical_or(np.abs(norm) < 1e-4, 
            #np.abs(norm) < np.abs(norm.max()*1e-2)), 0.0, 1.0 / norm)
            vecb[fi] = np.dot(dbeam[fi], vec[fi, :].reshape(self.ntel) * norm)

        return vecb.reshape((self.nfreq, self.telescope.num_pol_sky,
                             self.telescope.lmax + 1))
    

    def project_matrix_sky_to_telescope(self, mi, mat):
        """Project a covariance matrix from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [pol, pol, l, freq, freq]

        Returns
        -------
        tmat : np.ndarray
            Covariance in telescope basis.
        """       
        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1

        beam = self.beam_m(mi).reshape((self.nfreq, self.ntel, npol, lside))
        
        matf = np.zeros((self.nfreq, self.ntel, self.nfreq, self.ntel), dtype=np.complex128)


        # Should it be a +=?
        for pi in range(npol):
            for pj in range(npol):
                for fi in range(self.nfreq):
                    for fj in range(self.nfreq):
                        matf[fi, :, fj, :] += np.dot((beam[fi, :, pi, :] * mat[pi, pj, :, fi, fj]), beam[fj, :, pj, :].T.conj())

        return matf
    
    project_matrix_forward = project_matrix_sky_to_telescope


    def _svd_num(self, mi):
        ## Calculate the number of SVD modes meeting the cut for each
        ## frequency, return the number and the array bounds

        # Get the array of singular values for each mode
        sv = self.beam_singularvalues(mi)

        # Number of significant sv modes at each frequency
        svnum = (sv > sv.max() * self.svcut).sum(axis=1)

        # Calculate the block bounds within the full matrix
        svbounds = np.cumsum(np.insert(svnum, 0, 0))

        return svnum, svbounds


    def _svd_freq_iter(self, mi):
        num = self._svd_num(mi)[0]
        return [fi for fi in range(self.nfreq) if (num[fi] > 0)]


    def project_matrix_sky_to_svd(self, mi, mat, temponly=False):
        """Project a covariance matrix from the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [pol, pol, l, freq, freq]
        temponly: boolean
            Force projection of temperature (TT) part only (default: False)

        Returns
        -------
        tmat : np.ndarray [nsvd, nsvd]
            Covariance in SVD basis.
        """       
        
        npol = 1 if temponly else self.telescope.num_pol_sky

        lside = self.telescope.lmax + 1

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)
        
        # Create the output matrix
        matf = np.zeros((svbounds[-1], svbounds[-1]), dtype=np.complex128)

        # Should it be a +=?
        for pi in range(npol):
            for pj in range(npol):
                for fi in self._svd_freq_iter(mi):

                    fibeam = beam[fi, :svnum[fi], pi, :] # Beam for this pol, freq, and svcut (i)

                    for fj in self._svd_freq_iter(mi):
                        fjbeam = beam[fj, :svnum[fj], pj, :] # Beam for this pol, freq, and svcut (j)
                        lmat = mat[pi, pj, :, fi, fj] # Local section of the sky matrix (i.e C_l part)

                        matf[svbounds[fi]:svbounds[fi+1], svbounds[fj]:svbounds[fj+1]] += np.dot(fibeam * lmat, fjbeam.T.conj())

        return matf


    def project_matrix_diagonal_telescope_to_svd(self, mi, dmat):
        """Project a diagonal matrix from the telescope basis into the SVD basis.

        This slightly specialised routine is for projecting the noise
        covariance into the SVD space.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [nfreq, ntel]

        Returns
        -------
        tmat : np.ndarray [nsvd, nsvd]
            Covariance in SVD basis.
        """ 

        svdfile = h5py.File(self._svdfile(mi), 'r')

        # Get the SVD beam matrix
        beam = svdfile['beam_ut']

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)
        
        # Create the output matrix
        matf = np.zeros((svbounds[-1], svbounds[-1]), dtype=np.complex128)

        # Should it be a +=?
        for fi in self._svd_freq_iter(mi):

            fbeam = beam[fi, :svnum[fi], :] # Beam matrix for this frequency and cut
            lmat = dmat[fi, :] # Matrix section for this frequency

            matf[svbounds[fi]:svbounds[fi+1], svbounds[fi]:svbounds[fi+1]] = np.dot((fbeam * lmat), fbeam.T.conj())

        return matf


    def project_vector_telescope_to_svd(self, mi, vec):

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_ut(mi)
        
        # Create the output matrix (shape is calculated from input shape)
        vecf = np.zeros((svbounds[-1],) + vec.shape[2:], dtype=np.complex128)

        # Should it be a +=?
        for fi in self._svd_freq_iter(mi):

            fbeam = beam[fi, :svnum[fi], :] # Beam matrix for this frequency and cut
            lvec = vec[fi, :] # Matrix section for this frequency

            vecf[svbounds[fi]:svbounds[fi+1]] = np.dot(fbeam, lvec)

        return vecf


    def project_vector_sky_to_svd(self, mi, vec, temponly=False):
        """Project a vector from the the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, lmax+1]
        temponly: boolean
            Force projection of temperature part only (default: False)

        Returns
        -------
        svec : np.ndarray
            SVD vector to return.
        """
        npol = 1 if temponly else self.telescope.num_pol_sky

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)
        
        # Create the output matrix
        vecf = np.zeros((svbounds[-1],) + vec.shape[3:], dtype=np.complex128)

        for pi in range(npol):
            for fi in self._svd_freq_iter(mi):

                fbeam = beam[fi, :svnum[fi], pi, :] # Beam matrix for this frequency and cut
                lvec = vec[fi, pi] # Matrix section for this frequency

                vecf[svbounds[fi]:svbounds[fi+1]] += np.dot(fbeam, lvec)

        return vecf


    def project_vector_svd_to_sky(self, mi, vec, temponly=False, conj=False):
        """Project a vector from the the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, lmax+1]
        temponly: boolean
            Force projection of temperature part only (default: False)
        conj: boolean
            Reverse projection by applying conjugation (as opposed to pseudo-
            inverse). Default is False.

        Returns
        -------
        svec : np.ndarray
            SVD vector to return.
        """
        npol = 1 if temponly else self.telescope.num_pol_sky

        if not conj:
            raise Exception("Not implemented non conj yet.")

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)
        
        # Create the output matrix
        vecf = np.zeros((self.nfreq, 3, self.telescope.lmax + 1,) + vec.shape[1:], dtype=np.complex128)

        for pi in range(npol):
            for fi in self._svd_freq_iter(mi):

                fbeam = beam[fi, :svnum[fi], pi, :].T.conj() # Beam matrix for this frequency and cut
                lvec = vec[svbounds[fi]:svbounds[fi+1]] # Matrix section for this frequency

                vecf[fi, pi] += np.dot(fbeam, lvec)

        return vecf





    #===================================================

    #====== Dimensionality of the various spaces =======

    @property
    def ntel(self):
        """Degrees of freedom measured by the telescope (per frequency)"""
        return 2 * self.telescope.npairs * self.telescope.num_pol_telescope

    @property
    def nsky(self):
        """Degrees of freedom on the sky at each frequency."""
        return (self.telescope.lmax + 1) * self.telescope.num_pol_sky

    @property
    def nfreq(self):
        """Number of frequencies measured."""
        return self.telescope.nfreq

    @property
    def ndofmax(self):
        return min(self.ntel, self.telescope.lmax+1) * self.nfreq

    def ndof(self, mi):
        """The number of degrees of freedom at a given m."""
        return self._svd_num(mi)[1][-1]

    #===================================================







class BeamTransferNoSVD(BeamTransfer):

    svcut = 0.0

    def project_matrix_sky_to_svd(self, mi, mat, *args, **kwargs):
        return self.project_matrix_sky_to_telescope(mi, mat).reshape(self.ndof(mi), self.ndof(mi))


    def project_vector_sky_to_svd(self, mi, vec, *args, **kwargs):
        return self.project_vector_sky_to_telescope(mi, vec)


    def project_matrix_diagonal_telescope_to_svd(self, mi, dmat, *args, **kwargs):
        return np.diag(dmat.flatten())

    def project_vector_telescope_to_svd(self, mi, vec, *args, **kwargs):
        return vec

    def beam_svd(self, mi, *args, **kwargs):
        return self.beam_m(mi)



    def ndof(self, mi, *args, **kwargs):
        
        return self.ntel * self.nfreq

    @property
    def ndofmax(self):
        return self.ntel * self.nfreq

            
    