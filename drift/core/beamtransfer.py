import pickle
import os
import time

import numpy as np
import scipy.linalg as la
import h5py
from mpi4py import MPI

from drift.util import mpiutil, util, blockla


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
        return self._mdir(mi) + '/beam.hdf5'

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
    def beam_m(self, mi, fi=None):
        """Fetch the beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, 2, npairs, npol_sky, lmax+1)
        """

        return self._load_beam_m(mi, fi=fi)

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

        if single:
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
        invbeam : np.ndarray (nfreq, npol_sky, lmax+1, 2, npairs)
        """

        beam = self.beam_m(mi)

        if self.noise_weight:
            noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), 0).flatten()**(-0.5)
            beam = beam * noisew[:, np.newaxis, np.newaxis]

        beam = beam.reshape((self.nfreq, self.ntel, self.nsky))

        ibeam = blockla.pinv_dm(beam, rcond=1e-6)

        if self.noise_weight:
            # Reshape to make it easy to multiply baselines by noise level
            ibeam = ibeam.reshape((-1, self.telescope.npairs))
            ibeam = ibeam * noisew

        shape = (self.nfreq, self.telescope.num_pol_sky,
                 self.telescope.lmax + 1, self.ntel)
        
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
    def invbeam_svd(self, mi, fi=None):
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
            ibs = svdfile['invbeam_svd'][:]
        else:
            ibs = svdfile['invbeam_svd'][fi][:]
            
        svdfile.close()

        return ibs


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
        """Fetch the SVD beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, 2, npairs, npol_sky, lmax+1)
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
        ## Create all the directories required to store the beam transfers.

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
        ## Generate the beam transfers ordered by frequency.
        ## Divide frequencies between MPI processes and calculate the beams
        ## for the baselines, then write out into separate files.

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

            dsize = (self.telescope.nbase, self.telescope.num_pol_sky, self.telescope.lmax+1, 2*self.telescope.mmax+1)

            csize = (10, self.telescope.num_pol_sky, self.telescope.lmax+1, 1)

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
        ## Generate the m-mode files by reading in the frequency ordered beams
        ## and transposing between processes, the write out the beams ordered
        ## by m.

        if os.path.exists(self.directory + '/beam_m/COMPLETED'):
            if mpiutil.rank0:
                print "******* m-files already generated ********"
            return

        st = time.time()

        nf = self.telescope.nfreq
        nm = self.telescope.mmax + 1
        
        lfreq, sfreq, efreq = mpiutil.split_local(nf)
        lm, sm, em = mpiutil.split_local(nm)        

        # Calculate the number of baselines to deal with at any one time. Aim
        # to have a maximum of 4 GB in memory at any one time
        blsize = (mpiutil.split_all(nf)[0].max() * self.telescope.num_pol_sky *
                  (self.telescope.lmax+1) * (2*self.telescope.mmax+1) * 16.0)

        num_bl_per_chunk = int(1e7 / blsize) # Number of baselines to process in each chunk
        num_chunks = int(self.telescope.nbase / num_bl_per_chunk) + 1

        if mpiutil.rank0:
            print "====================================================="
            print "  Processing %i frequencies and %i m's" % (nf, nm)
            print "  Split into groups of %i and %i respectively" % (lfreq, lm)
            print
            print "  %i groups of %i baselines (size %f GB)" % (num_chunks, num_bl_per_chunk, blsize * num_bl_per_chunk / 2**30.0)
            print "====================================================="

        # Iterate over all m's and create the hdf5 files we will write into.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._mfile(mi)) and not regen:
                print ("m index %i. File: %s exists. Skipping..." % (mi, (self._mfile(mi))))
                continue
            
            f = h5py.File(self._mfile(mi), 'w')

            dsize = (self.telescope.nfreq, 2, self.telescope.nbase, self.telescope.num_pol_sky, self.telescope.lmax+1)
            csize = (1, 2, 10, self.telescope.num_pol_sky, self.telescope.lmax+1)
            f.create_dataset('beam_m', dsize, chunks=csize, compression='lzf', dtype=np.complex128)

            # Write a few useful attributes.
            f.attrs['baselines'] = self.telescope.baselines
            f.attrs['m'] = mi
            f.attrs['frequencies'] = self.telescope.frequencies
            f.attrs['cylobj'] = self._telescope_pickle

            f.close()

        mpiutil.barrier()

        # Iterate over all chunks performing a cycle of: read frequency data, transpose to m-order data, write m-data.
        for ci, blrange in enumerate(mpiutil.split_m(self.telescope.nbase, num_chunks).T):

            if mpiutil.rank0:
                print
                print "============================================="
                print "    Starting chunk %i of %i" % (ci + 1, num_chunks)
                print "============================================="
                print

            # Unpack baselines range into num, start and end
            blnum, blstart, blend = blrange

            # Array to load frequency data into
            freq_array = np.zeros((lfreq, 2, blnum, self.telescope.num_pol_sky, self.telescope.lmax + 1, self.telescope.mmax + 1), dtype=np.complex128)

            ## Read frequency data from the disk, and combine the positive and negative m parts.
            for lfi, fi in enumerate(range(sfreq, efreq)):
                ff = h5py.File(self._ffile(fi), 'r')
                fchunk = ff['beam_freq'][blstart:blend][:]
                ff.close()

                for mi in range(self.telescope.mmax + 1):
                    freq_array[lfi, 0, ..., mi] = fchunk[..., mi]
                    freq_array[lfi, 1, ..., mi] = (-1.0)**mi * fchunk[..., -mi].conj()                    

            mpiutil.barrier()

            # Perform an in memory MPI transpose
            m_array = mpiutil.transpose_blocks(freq_array, (nf, 2, blnum, self.telescope.num_pol_sky, self.telescope.lmax + 1, self.telescope.mmax + 1))

            # Write out the current set of chunks into the m-files.
            for lmi, mi in enumerate(range(sm, em)):

                mfile = h5py.File(self._mfile(mi), 'r+')
                mfile['beam_m'][:, :, blstart:blend] = m_array[..., lmi]
                mfile.close()

            print "rank %i: Done writing chunks to disk." % mpiutil.rank

            # Delete the local frequency and m ordered
            # sections. Otherwise we run out of memory on the next
            # iteration as there is a brief moment where the chunks
            # exist for both old and new iterations.
            del freq_array
            del m_array

            mpiutil.barrier()


        et = time.time()
        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(self.directory + '/beam_m/COMPLETED', 'a').close()

            # Print out timing
            print "=== MPI transpose took %f s ===" % (et - st)






    def _generate_svdfiles(self, regen=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available. 
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print ("m index %i. File: %s exists. Skipping..." %
                       (mi, (self._svdfile(mi))))
                continue
            else:
                print 'm index %i. Creating SVD file: %s' % (mi, self._svdfile(mi))

            # Open m beams for reading.
            fm = h5py.File(self._mfile(mi), 'r')

            # Open file to write SVD results into.
            fs = h5py.File(self._svdfile(mi), 'w')

            # The size of the SVD output matrices
            svd_len = min(self.telescope.lmax+1, self.ntel)

            # Create a chunked dataset for writing the SVD beam matrix into.
            dsize_bsvd = (self.telescope.nfreq, svd_len, self.telescope.num_pol_sky, self.telescope.lmax+1)
            csize_bsvd = (1, 10, self.telescope.num_pol_sky, self.telescope.lmax+1)
            dset_bsvd = fs.create_dataset('beam_svd', dsize_bsvd, chunks=csize_bsvd, compression='lzf', dtype=np.complex128)

            # Create a chunked dataset for writing the inverse SVD beam matrix into.
            dsize_ibsvd = (self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax+1, svd_len)
            csize_ibsvd = (1, self.telescope.num_pol_sky, self.telescope.lmax+1, 10)
            dset_ibsvd = fs.create_dataset('invbeam_svd', dsize_ibsvd, chunks=csize_ibsvd, compression='lzf', dtype=np.complex128)

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
                bf = fm['beam_m'][fi][:].reshape(self.ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1)

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
                bsvd = np.dot(u, bf.reshape(self.ntel, -1))
                dset_bsvd[fi] = bsvd.reshape(svd_len, self.telescope.num_pol_sky, self.telescope.lmax + 1)

                # Find the pseudo-inverse of the beam matrix and save to disk.
                dset_ibsvd[fi] = la.pinv(bsvd).reshape(self.telescope.num_pol_sky, self.telescope.lmax + 1, svd_len)

                # Save out the singular values for each block
                dset_sig[fi] = sig

                
            # Write a few useful attributes.
            fs.attrs['baselines'] = self.telescope.baselines
            fs.attrs['m'] = mi
            fs.attrs['frequencies'] = self.telescope.frequencies
            fs.attrs['cylobj'] = self._telescope_pickle
            
            fs.close()
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
            #vecf[fi] = np.dot(beam[fi], vec[..., fi, :].reshape(self.nsky))
            vecf[fi] = np.dot(beam[fi], vec[fi].reshape(self.nsky))

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

        # if not conj:
        #     raise Exception("Not implemented non conj yet.")

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_svd(mi) if conj else self.invbeam_svd(mi)
        
        # Create the output matrix
        vecf = np.zeros((self.nfreq, 3, self.telescope.lmax + 1,) + vec.shape[1:], dtype=np.complex128)

        for pi in range(npol):
            for fi in self._svd_freq_iter(mi):

                if conj:
                    fbeam = beam[fi, :svnum[fi], pi, :].T.conj() # Beam matrix for this frequency and cut
                else:
                    fbeam = beam[fi, pi, :, :svnum[fi]] # Beam matrix for this frequency and cut

                lvec = vec[svbounds[fi]:svbounds[fi+1]] # Matrix section for this frequency

                vecf[fi, pi] += np.dot(fbeam, lvec)

        return vecf





    #===================================================

    #====== Dimensionality of the various spaces =======

    @property
    def ntel(self):
        """Degrees of freedom measured by the telescope (per frequency)"""
        return 2 * self.telescope.npairs

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

            
    
