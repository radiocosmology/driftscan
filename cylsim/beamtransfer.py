import pickle
import os

import numpy as np
import scipy.linalg as la
import h5py

import mpiutil
import util
import blockla


class BeamTransfer(object):
    """A class for reading and writing Beam Transfer matrices from disk. In
    addition this provides methods for projecting vectors and matrices between
    the sky and the telescope basis.
    """

    #====== Properties giving internal filenames =======

    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.directory + "/telescopeobject.pickle"

    @property
    def _mfile(self):
        # Pattern to form the `m` ordered file.
        return (self.directory + "/beam_m_" +
                util.intpattern(self.telescope.mmax) + ".hdf5")

    @property
    def _ffile(self):
        # Pattern to form the `freq` ordered file.
        return (self.directory + "/beam_f_" +
                util.natpattern(self.telescope.nfreq) + ".hdf5")

    #===================================================

    #=========== Patterns for HDF5 datasets ============

    @property
    def _msection(self):
        # The pattern for `m` datasets in freq files.
        return "m_section/" + util.intpattern(self.telescope.mmax)

    @property
    def _fsection(self):
        # The pattern for `freq` datasets in m files.
        return "freq_section/" + util.natpattern(self.telescope.nfreq)

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


    def _load_beam_m(self, mi):
        ## Read in beam from disk
        mfile = h5py.File(self._mfile % mi, 'r')
        sh = mfile[(self._fsection % 0)].shape

        beam = np.zeros((self.telescope.nfreq,) + sh, dtype=np.complex128)

        for fi in range(self.telescope.nfreq):
            beam[fi] = mfile[(self._fsection % fi)]

        mfile.close()

        return beam


    @util.cache_last
    def beam_m(self, mi, single=False):
        """Fetch the beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        single : boolean, optional
            When set, fetch only the uncombined beam transfer (that is only
            positive or negative m). Default is False.

        Returns
        -------
        beam : np.ndarray (nfreq, 2, nbase, npol_tel, npol_sky, lmax+1)
            If `single` is set, shape (nfreq, nbase, npol_tel, npol_sky, lmax+1)
        """
        if single or self.telescope.positive_m_only:
            return self._load_beam_m(mi)

        bp = self._load_beam_m(mi)
        bm = (-1)**mi * self._load_beam_m(-mi).conj()

        # Zero out m=0 (for negative) to avoid double counting
        if mi == 0:
            bm[:] = 0.0

        bc = np.empty(bp.shape[:1] + (2,) + bp.shape[1:], dtype=bp.dtype)

        bc[:, 0] = bp
        bc[:, 1] = bm

        return bc



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
        invbeam : np.ndarray (nfreq, npol_sky, lmax+1, 2, nbase, npol_tel)
        """

        beam = self.beam_m(mi)

        if self.noise_weight:
            noisew = self.telescope.noisepower(np.arange(self.telescope.nbase), 0).flatten()**(-0.5)
            beam = beam * noisew[:, np.newaxis, np.newaxis, np.newaxis]

        beam = beam.reshape((self.nfreq, self.ntel, self.nsky))

        ibeam = blockla.pinv_dm(beam, rcond=1e-9)

        if self.noise_weight:
            # Reshape to make it easy to multiply baselines by noise level
            ibeam = ibeam.reshape((-1, self.telescope.nbase, self.telescope.num_pol_telescope))
            ibeam = ibeam * noisew[:, np.newaxis]

        shape = (self.nfreq, self.telescope.num_pol_sky,
                 self.telescope.lmax + 1, self.ntel,
                 self.telescope.num_pol_telescope)
        
        return ibeam.reshape(shape)


    @util.cache_last
    def _load_beam_freq(self, fi, fullm=False):
        
        tel = self.telescope
        mside = 2 * tel.lmax + 1 if fullm else 2 * tel.mmax + 1
        
        ffile = h5py.File(self._ffile % fi, 'r')
        sh = ffile[(self._msection % 0)].shape
        beam = np.zeros(sh + (mside,), dtype=np.complex128)
        
        for mi in range(-tel.mmax, tel.mmax + 1):
            beam[..., mi] = ffile[(self._msection % mi)]
            
        ffile.close()
        
        return beam


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



    def generate(self, regen=False):
        """Save out all beam transfer matrices to disk.

        Parameters
        ----------
        regen : boolean, optional
            Force regeneration even if cache files exist (default: False).
        """

        # For each frequency, create the HDF5 file, and write in each `m` as a
        # seperate compressed dataset. Use MPI if available. 
        for fi in mpiutil.mpirange(self.nfreq):

            if os.path.exists(self._ffile % fi) and not regen:
                print ("f index %i. File: %s exists. Skipping..." %
                       (fi, (self._ffile % fi)))
                continue
            else:
                print ('f index %i. Creating file: %s' %
                       (fi, (self._ffile % fi)))

            # Calculate transfer matrices for each frequency
            btrans = self.telescope.transfer_for_frequency(fi)
            
            f = h5py.File(self._ffile % fi, 'w')
            f.create_group('m_section')
            # Set a few useful attributes.
            f.attrs['baselines'] = self.telescope.baselines
            f.attrs['baseline_indices'] = np.arange(self.telescope.nbase)
            f.attrs['frequency_index'] = fi
            f.attrs['frequency'] = self.telescope.frequencies[fi]
            f.attrs['cylobj'] = self._telescope_pickle

            for mi in range(-self.telescope.mmax, self.telescope.mmax + 1):
                
                dset = f.create_dataset(self._msection % mi,
                                        data=btrans[..., mi],
                                        compression='gzip')
                dset.attrs['m'] = mi
        
            f.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available. 
        for mi in mpiutil.mpirange(-self.telescope.mmax,
                                   self.telescope.mmax + 1):

            if os.path.exists(self._mfile % mi) and not regen:
                print ("m index %i. File: %s exists. Skipping..." %
                       (mi, (self._mfile % mi)))
                continue
            else:
                print 'm index %i. Creating file: %s' % (mi, self._mfile % mi)

            ## Create hdf5 file for each m-mode
            f = h5py.File(self._mfile % mi, 'w')
            f.create_group('freq_section')
            ## For each frequency read in the current m-mode 
            ## and copy into file.
            for fi in np.arange(self.telescope.nfreq):

                ff = h5py.File(self._ffile % fi, 'r')

                # Check frequency is what we expect.
                if fi != ff.attrs['frequency_index']:
                    raise Exception("Bork.")
        
                dset = f.create_dataset(self._fsection % fi,
                                        data=ff[self._msection % mi],
                                        compression='gzip')
                dset.attrs['frequency_index'] = fi
                ff.close()
                
            # Write a few useful attributes.
            f.attrs['baselines'] = self.telescope.baselines
            f.attrs['m'] = mi
            f.attrs['frequencies'] = self.telescope.frequencies
            f.attrs['cylobj'] = self._telescope_pickle
            
            f.close()

        # Save pickled telescope object
        if mpiutil.rank0:
            with open(self._picklefile, 'w') as f:
                print "=== Saving Telescope object. ==="
                pickle.dump(self.telescope, f)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

    generate_cache = generate # For compatibility with old code
        

    def project_vector_forward(self, mi, vec):
        """Project a vector from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [freq, pol, l]

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

    
    def project_vector_backward(self, mi, vec):
        """Invert a vector from the telescope space onto the sky. This is the
        map-making process.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [freq, pol, l]

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
    

    def project_matrix_forward(self, mi, mat):
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

    @property
    def ntel(self):
        """Degrees of freedom measured by the telescope (per frequency)"""
        if self.telescope.positive_m_only:
            return self.telescope.nbase * self.telescope.num_pol_telescope
        else:
            return 2 * self.telescope.nbase * self.telescope.num_pol_telescope

    @property
    def nsky(self):
        """Degrees of freedom on the sky at each frequency."""
        return (self.telescope.lmax + 1) * self.telescope.num_pol_sky

    @property
    def nfreq(self):
        """Number of frequencies measured."""
        return self.telescope.nfreq






                    
            
        
