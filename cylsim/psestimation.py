
from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import skymodel


from cylsim import mpiutil

from simulations import corr21cm

from scipy.integrate import quad

import numpy as np




def uniform_band(k, kstart, kend):
    return np.where(np.logical_and(k > kstart, k < kend), np.ones_like(k), np.zeros_like(k))



class PSEstimation(object):

    kltrans = None
    telescope = None

    bands = np.logspace(-3.0, 0.0, 10)

    def __init__(self, kltrans):

        self.kltrans = kltrans
        self.telescope = kltrans.telescope


    def genbands(self):
        self.band_pk = [((lambda bs, be: (lambda k: uniform_band(k, bs, be)))(b_start, b_end), b_start, b_end) for b_start, b_end in zip(self.bands[:-1], self.bands[1:])]
        
        cr = corr21cm.Corr21cm()

        self.bpower = [quad(cr.ps_vv, bs, be)[0] for pk, bs, be in self.band_pk]

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
        
        c = [self.makeproj(mi, clzz) for clzz in self.clarray]

        evals, evecs = self.kltrans.modes_m(mi)

        ci = np.diag(1.0 / (evals + 1.0))
        
        print "Making fisher."
        fab = 0.5 * np.array([ [ np.trace(np.dot(np.dot(c_a, ci), np.dot(c_b, ci))) for c_b in c] for c_a in c])

        return fab


    def fisher_all(self, mlist = None):

        if mlist is None:
            mlist = range(-self.telescope.mmax, self.telescope.mmax + 1)

        mpart = mpiutil.partition_list_mpi(mlist)

        fab_t = np.zeros((self.bands.size - 1, self.bands.size - 1), dtype=np.complex128)

        for mi in mpart:

            fab_m = self.fisher_m(mi)

            fab_t += fab_m

        return fab_t


    
