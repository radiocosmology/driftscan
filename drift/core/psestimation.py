"""Estimate powerspectra and forecast constraints from real data.
"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import os
import abc
import time

import h5py
import numpy as np
import scipy.linalg as la

from caput import config, mpiutil, mpiarray

from cora.signal import corr21cm
from cora.util import nputil

from drift.core import skymodel
from drift.util import util

from mpi4py import MPI
from future.utils import with_metaclass


def uniform_band(k, kstart, kend):
    return np.where(np.logical_and(k > kstart, k < kend), np.ones_like(k), np.zeros_like(k))


def bandfunc_2d_polar(ks, ke, ts, te):

    def band(k, mu):

        #k = (kpar**2 + kperp**2)**0.5
        theta = np.arccos(mu)

        tb = (theta >= ts) * (theta <= te)
        kb = (k >= ks) * (k < ke)

        return (kb * tb).astype(np.float64)

    return band


def bandfunc_2d_cart(kpar_s, kpar_e, kperp_s, kperp_e):

    def band(k, mu):

        kpar = k * mu
        kperp = k * (1.0 - mu**2)**0.5

        parb = (kpar >= kpar_s) * (kpar <= kpar_e)
        perpb = (kperp >= kperp_s) * (kperp < kperp_e)

        return (parb * perpb).astype(np.float64)

    return band



def range_config(lst):

    lst2 = []

    endpoint = False
    count = 1
    for item in lst:
        if isinstance(item, dict):
            if count == len(lst):
                endpoint = True
            count += 1

            if item['spacing'] == 'log':
                item = np.logspace(np.log10(item['start']), np.log10(item['stop']), item['num'], endpoint=endpoint)
            elif item['spacing'] == 'linear':
                item = np.linspace(item['start'], item['stop'], item['num'], endpoint=endpoint)

            item = np.atleast_1d(item)

            lst2.append(item)
        else:
            raise Exception("Require a dict.")

    return np.concatenate(lst2)


def decorrelate_ps(ps, fisher):
    """Decorrelate the powerspectrum estimate.

    Parameters
    ----------
    ps : np.ndarray[nbands]
        Powerspectrum estimate.
    fisher : np.ndarrays[nbands, nbands]
        Fisher matrix.

    Returns
    -------
    psd : np.narray[nbands]
        Decorrelated powerspectrum estimate.
    errors : np.ndarray[nbands]
        Errors on decorrelated bands.
    window : np.ndarray[nbands, nbands]
        Window functions for each band row-wise.
    """
    # Factorise the Fisher matrix
    fh = la.cholesky(fisher, lower=True)
    fhi = la.inv(fh)

    # Create the mixing matrix, and window functions
    m = fhi / np.sum(fh.T, axis=1)[:, np.newaxis]
    w = np.dot(m, fisher)

    # Find the decorrelated powerspectrum and its errors
    evm = np.dot(m, np.dot(fisher, m.T)).diagonal()**0.5
    psd = np.dot(w, ps)

    return psd, evm, w


def decorrelate_ps_file(fname):
    """Load and decorrelate the powerspectrum in `fname`.

    Parameters
    ----------
    fname : string
        Name of file to load.

    Returns
    -------
    psd : np.narray[nbands]
        Decorrelated powerspectrum estimate.
    errors : np.ndarray[nbands]
        Errors on decorrelated bands.
    window : np.ndarray[nbands, nbands]
        Window functions for each band row-wise.
    """
    f1 = h5py.File(fname, 'r')

    return decorrelate_ps(f1['powerspectrum'][:], f1['fisher'][:])



class PSEstimation(with_metaclass(abc.ABCMeta, config.Reader)):
    """Base class for quadratic powerspectrum estimation.

    See Tegmark 1997 for details.

    Attributes
    ----------
    bandtype : {'polar', 'cartesian'}
        Which types of bands to use (default: polar).


    k_bands : np.ndarray
        Array of band boundaries. e.g. np.array([0.0, 0.5, ]), polar only
    num_theta: integer
        Number of theta bands to use (polar only)

    kpar_bands : np.ndarray
        Array of band boundaries. e.g. np.array([0.0, 0.5, ]), cartesian only
    kperp_bands : np.ndarray
        Array of band boundaries. e.g. np.array([0.0, 0.5, ]), cartesian only

    threshold : scalar
        Threshold for including eigenmodes (default is 0.0, i.e. all modes)

    unit_bands : boolean
        If True, bands are sections of the exact powerspectrum (such that the
        fiducial bin amplitude is 1).

    zero_mean : boolean
        If True (default), then the fiducial parameters have zero mean.
    """


    bandtype = config.Property(proptype=str, default='cartesian')

    # Properties to control polar bands
    k_bands = config.Property(proptype=range_config, default=[ {'spacing' : 'linear', 'start' : 0.0, 'stop' : 0.4, 'num' : 20 }])
    num_theta = config.Property(proptype=int, default=1)

    # Properties for cartesian bands
    kpar_bands = config.Property(proptype=range_config, default=[ {'spacing' : 'linear', 'start' : 0.0, 'stop' : 0.2, 'num' : 3 }])
    kperp_bands = config.Property(proptype=range_config, default=[ {'spacing' : 'linear', 'start' : 0.0, 'stop' : 0.2, 'num' : 3 }])

    threshold = config.Property(proptype=float, default=0.0)

    unit_bands = config.Property(proptype=bool, default=True)

    zero_mean = config.Property(proptype=bool, default=True)

    crosspower = False

    clarray = None

    fisher = None
    bias = None


    def __init__(self, kltrans, subdir="ps"):
        """Initialise a PS estimator class.

        Parameters
        ----------
        kltrans : KLTransform
            The KL Transform filter to use.
        subdir : string, optional
            Subdirectory of the KLTransform directory to store results in.
            Default is 'ps'.
        """

        self.kltrans = kltrans
        self.telescope = kltrans.telescope
        self.psdir = self.kltrans.evdir + '/' + subdir + '/'

        if mpiutil.rank0 and not os.path.exists(self.psdir):
            os.makedirs(self.psdir)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()


    @property
    def nbands(self):
        """Number of powerspectrum bands."""
        return self.k_center.size


    def num_evals(self, mi):
        """Number of eigenvalues for this `m` (and threshold).

        Parameters
        ----------
        mi : integer
            m-mode index.

        Returns
        -------
        num_evals : integer
        """

        evals = self.kltrans.modes_m(mi, threshold=self.threshold)[0]

        return evals.size if evals is not None else 0


    #========== Calculate powerspectrum bands ==========

    def genbands(self):
        """Precompute the powerspectrum bands, including the P(k, mu) bands
        and the angular powerspectrum.
        """

        print("Generating bands...")

        cr = corr21cm.Corr21cm()
        cr.ps_2d = False

        # Create different sets of bands depending on whether we're using polar bins or not.
        if self.bandtype == 'polar':

            # Create the array of band bounds
            self.theta_bands = np.linspace(0.0, np.pi / 2.0, self.num_theta + 1, endpoint=True)

            # Broadcast the bounds against each other to make the 2D array of bands
            kb, tb = np.broadcast_arrays(self.k_bands[np.newaxis, :], self.theta_bands[:, np.newaxis])

            # Pull out the start, end and centre of the bands in k, mu directions
            self.k_start = kb[1:, :-1].flatten()
            self.k_end = kb[1:, 1:].flatten()
            self.k_center = 0.5 * (self.k_end + self.k_start)

            self.theta_start = tb[:-1, 1:].flatten()
            self.theta_end = tb[1:, 1:].flatten()
            self.theta_center = 0.5 * (self.theta_end + self.theta_start)

            bounds = list(zip(self.k_start, self.k_end, self.theta_start, self.theta_end))

            # Make a list of functions of the band window functions
            self.band_func = [ bandfunc_2d_polar(*bound) for bound in bounds ]

            # Create a list of functions of the band power functions
            if self.unit_bands:
                # Need slightly awkward double lambda because of loop closure scaling.
                self.band_pk = [ (lambda bandt: (lambda k, mu: cr.ps_vv(k) * bandt(k, mu)))(band) for band in self.band_func]
                self.band_power = np.ones_like(self.k_start)
            else:
                self.band_pk = self.band_func
                self.band_power = cr.ps_vv(self.k_center)

        elif self.bandtype == 'cartesian':

            # Broadcast the bounds against each other to make the 2D array of bands
            kparb, kperpb = np.broadcast_arrays(self.kpar_bands[np.newaxis, :], self.kperp_bands[:, np.newaxis])

            # Pull out the start, end and centre of the bands in k, mu directions
            self.kpar_start = kparb[1:, :-1].flatten()
            self.kpar_end = kparb[1:, 1:].flatten()
            self.kpar_center = 0.5 * (self.kpar_end + self.kpar_start)

            self.kperp_start = kperpb[:-1, 1:].flatten()
            self.kperp_end = kperpb[1:, 1:].flatten()
            self.kperp_center = 0.5 * (self.kperp_end + self.kperp_start)

            bounds = list(zip(self.kpar_start, self.kpar_end, self.kperp_start, self.kperp_end))

            self.k_center = (self.kpar_center**2 + self.kperp_center**2)**0.5

            # Make a list of functions of the band window functions
            self.band_func = [ bandfunc_2d_cart(*bound) for bound in bounds ]

        else:
            raise Exception('Bandtype %s is not supported.' % self.bandtype)


        # Create a list of functions of the band power functions
        if self.unit_bands:
            # Need slightly awkward double lambda because of loop closure scaling.
            self.band_pk = [ (lambda bandt: (lambda k, mu: cr.ps_vv(k) * bandt(k, mu)))(band) for band in self.band_func]
            self.band_power = np.ones_like(self.k_center)
        else:
            self.band_pk = self.band_func
            self.band_power = cr.ps_vv(self.k_center)

        # Use new parallel map to speed up computaiton of bands
        if self.clarray is None:

            self.make_clzz_array()


        print("Done.")


    def make_clzz(self, pk):
        """Make an angular powerspectrum from the input matter powerspectrum.

        Uses the lmax and frequencies from the telescope object.

        Parameters
        ----------
        pk : function, np.ndarray -> np.ndarray
            The input powerspectrum (must be vectorized).

        Returns
        -------
        aps : np.ndarray[lmax+1, nfreq, nfreq]
            The angular powerspectrum.
        """
        crt = corr21cm.Corr21cm(ps=pk, redshift=1.5)
        crt.ps_2d = True

        clzz = skymodel.im21cm_model(self.telescope.lmax, self.telescope.frequencies,
                                     self.telescope.num_pol_sky, cr = crt, temponly=True)

        print("Rank: %i - Finished making band." % mpiutil.rank)
        return clzz


    def make_clzz_array(self):

        nbands = self.nbands
        nfreq = self.telescope.nfreq
        lmax = self.telescope.lmax

        self.p_bands, self.s_bands, self.e_bands = mpiutil.split_all(nbands)
        self.p_loc, self.s_band_loc, self.e_band_loc = mpiutil.split_local(nbands)

        # which communicator to use ? can we set this when initializing the class?
        self.clarray = mpiarray.MPIArray(
            (nbands, lmax + 1, nfreq, nfreq), axis=0, dtype=np.float64, comm=MPI.COMM_WORLD)

        self.clarray[:] = 0.0

        for bl, bg in self.clarray.enumerate(axis=0):
            print("Make clzz", bl,bg)
            self.clarray[bl] = self.make_clzz(self.band_pk[bg])


    def delbands(self):
        """Delete power spectrum bands to save memory."""

        self.clarray = None




    #===================================================


    #==== Calculate the per-m Fisher matrix/bias =======

    def fisher_bias_m(self, mi):
        """Generate the Fisher matrix and bias for a specific m.

        Parameters
        ----------
        mi : integer
            m-mode to calculate for.

        """

        if self.num_evals(mi) > 0:
            print("Making fisher (for m=%i)." % mi)

            fisher, bias = self._work_fisher_bias_m(mi)

        else:
            print("No evals (for m=%i), skipping." % mi)

            fisher = np.zeros((self.nbands, self.nbands), dtype=np.complex128)
            bias = np.zeros((self.nbands,), dtype=np.complex128)

        return fisher, bias

    @abc.abstractmethod
    def _work_fisher_bias_m(self, mi):
        """Worker routine for calculating the Fisher and bias for a given m.

        This routine should be overriden for a new method of generating the Fisher matrix.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Returns
        -------
        fisher : np.ndarray[nbands, nbands]
            Fisher matrix.
        bias : np.ndarray[nbands]
            Bias vector.
        """
        pass

    #===================================================


    #==== Calculate the total Fisher matrix/bias =======

    def generate(self, regen=False):
        """Calculate the total Fisher matrix and bias and save to a file.

        Parameters
        ----------
        regen : boolean, optional
            Force regeneration if products already exist (default `False`).
        """

        if mpiutil.rank0:
            st = time.time()
            print("======== Starting PS calculation ========")

        ffile = self.psdir +'/fisher.hdf5'

        if os.path.exists(ffile) and not regen:
            print ("Fisher matrix file: %s exists. Skipping..." % ffile)
            return

        mpiutil.barrier()

        # Pre-compute all the angular power spectra for the bands.
        # MPIArray of clzz basis fcts is now distributed over bands see make_clzz_array
        self.genbands()

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        m_chunks, low_bound, upp_bound = self.split_single((self.telescope.mmax + 1), size)
        num_chunks = m_chunks.shape[0]
        print("m chunks", m_chunks, low_bound, upp_bound)


        # Create an MPI array for the qa's distributed over bands?
        num_realisations = 500
        n = num_realisations

        self.qa = mpiarray.MPIArray(
            (self.telescope.mmax + 1, self.nbands, num_realisations), axis=1, dtype=np.float64, comm=MPI.COMM_WORLD)

        self.qa[:] = 0.0

        # This is designed such that at each iteration one rank gets one m-mode.
        for ci, num_m in enumerate(m_chunks):
            if ci == 4:
                break
            if mpiutil.rank0:
                print("Starting chunk %i of %i" % (ci + 1, num_chunks))

            loc_num, loc_start, loc_end = mpiutil.split_local(num_m)
            print("local number of m's", loc_num)
            glob_num, glob_start, glob_end = mpiutil.split_all(num_m)

            # This doesn't work when you have multiple processes on a node... figure out why.
            mi = np.arange(low_bound[ci], upp_bound[ci])[loc_start:loc_end]
            print("This is mi", mi)
            print("Processing m-mode %i on rank %i" % (mi, rank))

            nfreq = self.telescope.nfreq
            lmax = self.telescope.lmax

            if loc_num > 0:

                if self.num_evals(mi) > 0:
                    # Generate random data
                    x = self.gen_sample(mi, n)
                    vec1, vec2 = self.project_vector_kl_to_sky(mi, x)
                    # Select temperature part only for q-estimation
                    # Make array contiguous
                    vec1 = np.ascontiguousarray(vec1[:, 0]).reshape(loc_num, nfreq, lmax + 1, n)
                    vec2 = np.ascontiguousarray(vec2[:, 0]).reshape(loc_num, nfreq, lmax + 1, n)
                    print("vec1.shape in if:", vec1.shape)

                # If I don't have evals at the moment return zero vector
                else:
                    vec1 = np.zeros((loc_num, nfreq, lmax + 1, n),
                                    dtype=np.complex128)
                    print("vec1.shape in else", vec1.shape)
                    vec2 = vec1

            else:
                vec1 = np.zeros((loc_num, nfreq, lmax + 1, n), dtype=np.complex128)
                vec2 = vec1
                print("vec1.shape", vec1.shape)

            noise = False
            lside = self.telescope.lmax + 1

            for ir in range(size):
                #for bi, bg in enumerate(range(s,e)):
                for bi, bg in self.qa.enumerate(axis=1):
                    print("bi, bg", bi, bg)
                    for li in range(lside):
                        lxvec = vec1[:, :, li]
                        lyvec = vec1[:, :, li]
                        # This code doesn't work if you have more than 1 process on a node....
                        #self.qa[mi, bi, :] += np.sum(lyvec.conj() * np.dot(self.clarray[bi][li].astype(np.complex128), lxvec), axis=0).astype(np.float64) # TT only.
                        self.qa[mi, bi, :] += np.sum(lyvec.conj() * np.matmul(self.clarray[bi][li].astype(np.complex128), lxvec), axis=1).astype(np.float64)

                print("Finished calculating qa")
                # To DO: in loop if noise block
                # The MPI communications we only have to do (size - 1) times
                if ir == (size - 1):
                    break

                print("before receive send")
                recv_buffer = self.recv_send_data(vec1, axis=-1)
                vec1 = recv_buffer

        # Once done with all the m's, redistribute qa array over m's
        self.qa = self.qa.redistribute(axis=0)

        print("qa shape", self.qa.shape)

        # Make an array for local fisher an bias
        fisher_loc = np.zeros((self.nbands, self.nbands), dtype=np.float64)
        bias_loc = np.zeros((self.nbands,), dtype=np.float64)

        # Calculate fisher for each m
        for ml, mg in self.qa.enumerate(axis=0):
            fisher_m = np.cov(self.qa[ml])
            bias_m = np.mean(self.qa[ml], axis=1)
            # Sum over all local m-modes to get the over all Fisher and bias per process
            fisher_loc += fisher_m.real # be careful with the real?!
            bias_loc += bias_m.real

        self.fisher = mpiutil.allreduce(fisher_loc, op=MPI.SUM)
        self.bias = mpiutil.allreduce(bias_loc, op=MPI.SUM)
        
        # Write out all the PS estimation products
        if mpiutil.rank0:
            et = time.time()
            print("======== Ending PS calculation (time=%f) ========" % (et - st))

            # Check to see ensure that Fisher matrix isn't all zeros.
            if not (self.fisher == 0).all():
                # Generate derived quantities (covariance, errors..)
                cv = la.pinv(self.fisher, rcond=1e-8)
                err = cv.diagonal()**0.5
                cr = cv / np.outer(err, err)
            else:
                cv = np.zeros_like(self.fisher)
                err = cv.diagonal()
                cr = np.zeros_like(self.fisher)

            f = h5py.File(self.psdir + '/fisher.hdf5', 'w')
            f.attrs['bandtype'] = np.string_(self.bandtype)  # HDF5 string issues

            f.create_dataset('fisher/', data=self.fisher)
            f.create_dataset('bias/', data=self.bias)
            f.create_dataset('covariance/', data=cv)
            f.create_dataset('errors/', data=err)
            f.create_dataset('correlation/', data=cr)


            f.create_dataset('band_power/', data=self.band_power)

            if self.bandtype == 'polar':
                f.create_dataset('k_start/', data=self.k_start)
                f.create_dataset('k_end/', data=self.k_end)
                f.create_dataset('k_center/', data=self.k_center)

                f.create_dataset('theta_start/', data=self.theta_start)
                f.create_dataset('theta_end/', data=self.theta_end)
                f.create_dataset('theta_center/', data=self.theta_center)

                f.create_dataset('k_bands', data=self.k_bands)
                f.create_dataset('theta_bands', data=self.theta_bands)

            elif self.bandtype == 'cartesian':

                f.create_dataset('kpar_start/', data=self.kpar_start)
                f.create_dataset('kpar_end/', data=self.kpar_end)
                f.create_dataset('kpar_center/', data=self.kpar_center)

                f.create_dataset('kperp_start/', data=self.kperp_start)
                f.create_dataset('kperp_end/', data=self.kperp_end)
                f.create_dataset('kperp_center/', data=self.kperp_center)

                f.create_dataset('kpar_bands', data=self.kpar_bands)
                f.create_dataset('kperp_bands', data=self.kperp_bands)


            f.close()
            
        """
            scatter_buf = np.zeros((loc_num, self.nbands, num_realisations), dtype=np.float64)

            # This doesn't work can't be redistributed over m's and subbands!! One idea is to create an MPIArray
            # at beginning which is distributed over subbands and just
            qa_size = self.nbands * num_realisations

            qa_sizes = glob_num * qa_size
            qa_displ = glob_start * qa_size
            print("glob_start", glob_start)
            loc_qa_size = loc_num * qa_size

            print("glob_sizes", qa_sizes)
            print("glob_displ", qa_displ)
            print("loc_sizes", loc_qa_size)

            print("qa all", qa[:, :, 0])

            comm.Scatterv([qa, qa_sizes, qa_displ, MPI.DOUBLE], [scatter_buf, loc_qa_size, MPI.DOUBLE])
        """

        """
            # Calculate q_a for noise power (x0^H N x0 = |x0|^2) -> put this in extra function?
            if noise:

                # If calculating crosspower don't include instrumental noise
                noisemodes = 0.0 if self.crosspower else 1.0
                noisemodes = noisemodes + (evals if self.zero_mean else 0.0)

                qa[-1] = np.sum((x0 * y0.conj()).T.real * noisemodes, axis=-1)
        """

    def recv_send_data(self, data, axis):

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        shape = data.shape
        dtype = data.dtype
        
        recv_rank = (rank - 1) % size
        send_rank = (rank + 1) % size

        print("This is rank %i receiving from rank %i" % (rank, recv_rank))

        recv_buffer = np.zeros(shape, dtype=dtype)

        # Need to send in 4GB chunks due to some MPI library. 
        message_size = 4 * 2 ** 30.0
        dsize = np.prod(shape) * 16.0
        num_messages = int(np.ceil(dsize / message_size))
        print("number of messages: %i", num_messages)
        num, sm, em = mpiutil.split_m(shape[axis], num_messages)

        for i in range(num_messages):
            print("message %i", i)
            slc = [slice(None)] * len(shape)
            slc[axis] = slice(sm[i], em[i])
            print(slc)

            # Initiate non-blocking receive
            request = comm.Irecv([recv_buffer[slc], MPI.DOUBLE_COMPLEX], source=recv_rank, tag=send_rank)
            # Initiate send
            comm.Send([data[slc], MPI.DOUBLE_COMPLEX], dest=send_rank, tag=rank)
            # Wait for receive
            request.Wait()
            print("Waiting to receive from source %i with tag %i on rank %i" % (recv_rank, send_rank, rank))
        
        return recv_buffer

    def split_single(self, m, size):

        quotient = m // size
        rem = m % size

        if rem != 0:
            m_chunks = np.append((size * np.ones(quotient, dtype=int)), rem)
        else:
            m_chunks = size * np.ones(quotient, dtype=int)

        bound = np.cumsum(np.insert(m_chunks, 0, 0))
        ms = m_chunks.shape[0]

        low_bound = bound[:ms]
        upp_bound = bound[1:(ms + 1)]

        return m_chunks, low_bound, upp_bound

        """
        # Partition list based on MPI rank
        llist = mpiutil.partition_list_mpi(zlist)
        # Operate on sublist
        fisher_bias_list = [self.fisher_bias_m(item) for ind, item in llist]

        # Unpack into separate lists of the Fisher matrix and bias
        fisher_loc, bias_loc = zip(*fisher_bias_list)

        # Sum over all local m-modes to get the over all Fisher and bias pe process
        fisher_loc = np.sum(np.array(fisher_loc), axis=0).real # Be careful of the .real here
        bias_loc = np.sum(np.array(bias_loc), axis=0).real # Be careful of the .real here

        self.fisher = mpiutil.allreduce(fisher_loc, op=MPI.SUM)
        self.bias = mpiutil.allreduce(bias_loc, op=MPI.SUM)
        """
    #===================================================


    def fisher_file(self):
        """Fetch the h5py file handle for the Fisher matrix.

        Returns
        -------
        file : h5py.File
            File pointing at the hdf5 file with the Fisher matrix.
        """
        return h5py.File(self.psdir + 'fisher.hdf5', 'r')


    def fisher_bias(self):

        with h5py.File(self.psdir + '/fisher.hdf5', 'r') as f:

            return f['fisher'][:], f['bias'][:]

    #===================================================


    #====== Estimate the q-parameters from data ========

    def project_vector_kl_to_sky(self, mi, vec1, vec2=None):
        """
        Parameters
        ----------
        mi : integer
            The m-mode we are calculating for.
        vec : np.ndarray[num_kl, num_realisatons]
            The vector(s) of data in the KL-basis

        Returns
        -------
        x2, y2 : np.ndarray[nfreq, lmax+1, num_realisations]
            The vectors(s) of data in the sky basis.
        """

        evals, evecs = self.kltrans.modes_m(mi)

        #if evals is None:
        #    return np.zeros((self.nbands + 1 if noise else self.nbands,))

        # Weight by C**-1 (transposes are to ensure broadcast works for 1 and 2d vecs)
        x0 = (vec1.T / (evals + 1.0)).T

        # Project back into SVD basis
        x1 = np.dot(evecs.T.conj(), x0)

        # Project back into sky basis
        print("Reading beam transfer matrix for m=%i" %mi)
        x2 = self.kltrans.beamtransfer.project_vector_svd_to_sky(mi, x1, temponly=True, conj=True)

        if vec2 is not None:
            y0 = (vec2.T / (evals + 1.0)).T
            y1 = np.dot(evecs.T.conj(), x0)
            y2 = self.kltrans.beamtransfer.project_vector_svd_to_sky(mi, x1, temponly=True,conj=True)
        else:
            y0 = x0
            y2 = x2

        return x2, y2

    def q_estimator(self, mi, vec1, vec2=None, noise=False):
        """Estimate the q-parameters from given data (see paper).

        Parameters
        ----------
        mi : integer
            The m-mode we are calculating for.
        noise : boolean, optional
            Whether we should project against the noise matrix. Used for
            estimating the bias by Monte-Carlo. Default is False.

        Returns
        -------
        qa : np.ndarray[numbands]
            Array of q-parameters. If noise=True then the array is one longer,
            and the last parameter is the projection against the noise.
        """
        sky_vec1, sky_vec2 = self.project_vector_kl_to_sky(mi, vec1, vec2=None)

        # Create empty q vector (length depends on if we're calculating the noise term too)
        qa = np.zeros((self.nbands + 1 if noise else self.nbands,) + vec1.shape[1:])

        lside = self.telescope.lmax + 1

        # Calculate q_a for each band
        for bi in range(self.nbands):

            for li in range(lside):

                lxvec = sky_vec1[:, 0, li]
                lyvec = sky_vec2[:, 0, li]

                qa[bi] += np.sum(lyvec.conj() * np.dot(self.clarray[bi][li].astype(np.complex128), lxvec), axis=0).astype(np.float64) # TT only.

        # Calculate q_a for noise power (x0^H N x0 = |x0|^2)
        if noise:

            # If calculating crosspower don't include instrumental noise
            noisemodes = 0.0 if self.crosspower else 1.0
            noisemodes = noisemodes + (evals if self.zero_mean else 0.0)

            qa[-1] = np.sum((x0 * y0.conj()).T.real * noisemodes, axis=-1)

        return qa.real

    #===================================================






class PSExact(PSEstimation):
    """PS Estimation class with exact calculation of the Fisher matrix.
    """

    @property
    def _cfile(self):
        # Pattern to form the `m` ordered cache file.
        return self.psdir + "/ps_c_m_" + util.intpattern(self.telescope.mmax) + "_b_" + util.natpattern(len(self.bands)-1) + ".hdf5"



    def makeproj(self, mi, bi):
        """Project angular powerspectrum band into KL-basis.

        Parameters
        ----------
        mi : integer
            m-mode.
        bi : integer
            band index.

        Returns
        -------
        klcov : np.ndarray[nevals, nevals]
            Covariance in KL-basis.
        """
        #print "Projecting to eigenbasis."
        #nevals = self.kltrans.modes_m(mi, threshold=self.threshold)[0].size

        # if nevals < 1000:
        #     return self.kltrans.project_sky_matrix_forward_old(mi, self.clarray[bi], self.threshold)
        # else:
        #return self.kltrans.project_sky_matrix_forward(mi, self.clarray[bi], self.threshold)

        clarray = self.clarray[bi].reshape((1, 1) + self.clarray[bi].shape)
        svdmat = self.kltrans.beamtransfer.project_matrix_sky_to_svd(mi, clarray, temponly=True)
        return self.kltrans.project_matrix_svd_to_kl(mi, svdmat, self.threshold)


    def cacheproj(self, mi):
        """Cache projected covariances on disk.

        Parameters
        ----------
        mi : integer
            m-mode.
        """

        ## Don't generate cache for small enough matrices
        if self.num_evals(mi) < 500:
            self._bp_cache = []

        for i in range(len(self.clarray)):
            print("Generating cache for m=%i band=%i" % (mi, i))
            projm = self.makeproj(mi, i)

            ## Don't generate cache for small enough matrices
            if self.num_evals(mi) < 500:
                self._bp_cache.append(projm)

            else:
                print("Creating cache file:" + self._cfile % (mi, i))
                f = h5py.File(self._cfile % (mi, i), 'w')
                f.create_dataset('proj', data=projm)
                f.close()


    def delproj(self, mi):
        """Deleted cached covariances from disk.

        Parameters
        ----------
        mi : integer
            m-mode.
        """
        ## As we don't cache for small matrices, just return
        if self.num_evals(mi) < 500:
            self._bp_cache = []

        for i in range(len(self.clarray)):

            fn = self._cfile % (mi, i)
            if os.path.exists(fn):
                print("Deleting cache file:" + fn)
                os.remove(self._cfile % (mi, i))


    def getproj(self, mi, bi):
        """Fetch cached KL-covariance (either from disk or just calculate if small enough).

        Parameters
        ----------
        mi : integer
            m-mode.
        bi : integer
            band index.

        Returns
        -------
        klcov : np.ndarray[nevals, nevals]
            Covariance in KL-basis.
        """
        fn = self._cfile % (mi, bi)

        ## For small matrices or uncached files don't fetch cache, just generate
        ## immediately
        if self.num_evals(mi) < 500:# or not os.path.exists:
            proj = self._bp_cache[bi]
            #proj = self.makeproj(mi, bi)
        else:
            f = h5py.File(fn, 'r')
            proj = f['proj'][:]
            f.close()

        return proj



    def _work_fisher_bias_m(self, mi):
        """Worker routine for calculating the Fisher and bias for a given m.

        This method exactly calculates the quantities by forward projecting
        the correlations.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Returns
        -------
        fisher : np.ndarray[nbands, nbands]
            Fisher matrix.
        bias : np.ndarray[nbands]
            Bias vector.
        """

        evals = self.kltrans.evals_m(mi, self.threshold)

        fisher = np.zeros((self.nbands, self.nbands), dtype=np.complex128)
        bias = np.zeros(self.nbands, dtype=np.complex128)

        self.cacheproj(mi)

        ci = 1.0 / (evals + 1.0)**0.5
        ci = np.outer(ci, ci)

        for ia in range(self.nbands):
            c_a = self.getproj(mi, ia)
            fisher[ia, ia] = np.sum(c_a * c_a.T * ci**2)

            for ib in range(ia):
                c_b = self.getproj(mi, ib)
                fisher[ia, ib] = np.sum(c_a * c_b.T * ci**2)
                fisher[ib, ia] = np.conj(fisher[ia, ib])

        self.delproj(mi)

        return fisher, bias
