
from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import skymodel


from cylsim import mpiutil

from simulations import corr21cm

from scipy.integrate import quad

import numpy as np

import os
import h5py



def uniform_band(k, kstart, kend):
    return np.where(np.logical_and(k > kstart, k < kend), np.ones_like(k), np.zeros_like(k))



class PSEstimation(object):

    kltrans = None
    telescope = None

    bands = np.logspace(-3.0, 0.0, 10)

    def __init__(self, kltrans, subdir = 'ps/'):

        self.kltrans = kltrans
        self.telescope = kltrans.telescope
        self.psdir = self.kltrans.evdir + '/' + subdir
        
        if mpiutil.rank0 and not os.path.exists(self.psdir):
            os.makedirs(self.psdir)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()
        

    def genbands(self):
        self.band_pk = [((lambda bs, be: (lambda k: uniform_band(k, bs, be)))(b_start, b_end), b_start, b_end) for b_start, b_end in zip(self.bands[:-1], self.bands[1:])]
        
        cr = corr21cm.Corr21cm()

        self.bpower = np.array([quad(cr.ps_vv, bs, be)[0] for pk, bs, be in self.band_pk])

        self.clarray = [self.make_clzz(pk) for pk, bs, be in self.band_pk]

        
    def make_clzz(self, pk):
        print "Making C_l(z,z')"
        crt = corr21cm.Corr21cm(ps = pk)

        clzz = skymodel.im21cm_model(self.telescope.lmax, self.telescope.frequencies,
                                     self.telescope.num_pol_sky, cr = crt)
        return clzz

    
    def makeproj(self, mi, clzz):
        print "Projecting to eigenbasis."
        return self.kltrans.project_sky_matrix_forward(mi, clzz)


    def fisher_m(self, mi):
        
        evals, evecs = self.kltrans.modes_m(mi)

        if evals is not None:
            c = [self.makeproj(mi, clzz) for clzz in self.clarray]
            ci = np.diag(1.0 / (evals + 1.0))
            print "Making fisher."
            fab = 0.5 * np.array([ [ np.trace(np.dot(np.dot(c_a, ci), np.dot(c_b, ci))) for c_b in c] for c_a in c])
        else:
            l = self.bands.size - 1
            fab = np.zeros((l, l))
        return fab


    def _fisher_section(self, mlist):

        return [ (mi, self.fisher_m(mi)) for mi in mlist ]


    def fisher_mpi(self, mlist = None):
        if mlist is None:
            mlist = range(-self.telescope.mmax, self.telescope.mmax + 1)

        mpart = mpiutil.partition_list_mpi(mlist)

        f_m = self._fisher_section(mpart)

        f_all = mpiutil.world.gather(f_m, root=0)

        if mpiutil.rank0:
            nb = self.bands - 1
            fisher = np.zeros((2*self.telescope.mmax+1, nb, nb), dtype=np.complex128)

            for proc_rank in f_all:
                for fm in proc_rank:
                    fisher[fm[0]] = fisher[fm[1]]

            f = h5py.File(self.psdir + 'fisher.hdf5')

            f.create_dataset('fisher_m/', data=fisher)
            f.create_dataset('fisher_all/', data=np.sum(fisher, axis=0))
            f.create_dataset('bandpower/', data=self.bpower)

            f.close()
            
    
    def fisher_section(self, mlist = None):

        if mlist is None:
            mlist = range(-self.telescope.mmax, self.telescope.mmax + 1)
            
        mpart = mpiutil.partition_list_mpi(mlist)

        fab_t = np.zeros((self.bands.size - 1, self.bands.size - 1), dtype=np.complex128)

        for mi in mpart:

            fab_m = self.fisher_m(mi)

            fab_t += fab_m

        return fab_t

