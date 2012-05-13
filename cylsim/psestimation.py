
from cylsim import beamtransfer
from cylsim import kltransform
from cylsim import skymodel


from cylsim import mpiutil
from cylsim import util

from simulations import corr21cm

from scipy.integrate import quad

import numpy as np

import os
import h5py



def uniform_band(k, kstart, kend):
    return np.where(np.logical_and(k > kstart, k < kend), np.ones_like(k), np.zeros_like(k))


def range_config(lst):

    lst2 = []

    for item in lst:
        if isinstance(item, dict):
            
            if item['spacing'] == 'log':
                item = np.logspace(np.log10(item['start']), np.log10(item['stop']), item['num'], endpoint=False)
            elif item['spacing'] == 'linear':
                item = np.linspace(item['start'], item['stop'], item['num'], endpoint=False)

        item = np.atleast_1d(item)

        lst2.append(item)

    return np.concatenate(lst2)


class PSEstimation(util.ConfigReader):



    bands = np.concatenate((np.linspace(0.0, 0.25, 25, endpoint=False),     np.logspace(np.log10(0.25), np.log10(3.0), 16)))
    threshold = 0.0


    __config_table_ =   {   'bands'     : [range_config,    'bands'],
                            'threshold' : [float,           'threshold']
                        }

    @property
    def _cfile(self):
        # Pattern to form the `m` ordered file.
        return self.psdir + "/ps_c_m_" + util.intpattern(self.telescope.mmax) + "_b_" + util.natpattern(len(self.bands)-1) + ".hdf5"


    def __init__(self, kltrans, subdir=None):

        self.kltrans = kltrans
        self.telescope = kltrans.telescope
        self.psdir = self.kltrans.evdir + '/' + ("ps/" if subdir is None else subdir)
        
        if mpiutil.rank0 and not os.path.exists(self.psdir):
            os.makedirs(self.psdir)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Add configuration options                
        self.add_config(self.__config_table_)
        

    def genbands(self):

        print "Generating bands..."
   
        self.band_pk = [((lambda bs, be: (lambda k: uniform_band(k, bs, be)))(b_start, b_end), b_start, b_end) for b_start, b_end in zip(self.bands[:-1], self.bands[1:])]

        if mpiutil.rank0:
            for i, (pk, bs, be) in enumerate(self.band_pk):
                print "Band %i: %f to %f. Centre: %g" % (i, bs, be, 0.5*(be+bs))

        
        cr = corr21cm.Corr21cm()

        self.bpower = np.array([(quad(cr.ps_vv, bs, be)[0] / (be - bs)) for pk, bs, be in self.band_pk])

        self.bstart = self.bands[:-1]
        self.bend = self.bands[1:]
        self.bcenter = 0.5*(self.bands[1:] + self.bands[:-1])

        self.clarray = [self.make_clzz(pk) for pk, bs, be in self.band_pk]
        print "Done."
        
    def make_clzz(self, pk):
        #print "Making C_l(z,z')"
        crt = corr21cm.Corr21cm(ps=pk, redshift=1.5)

        clzz = skymodel.im21cm_model(self.telescope.lmax, self.telescope.frequencies,
                                     self.telescope.num_pol_sky, cr = crt)
        return clzz


    def num_evals(self, mi):
        evals = self.kltrans.modes_m(mi, threshold=self.threshold)[0]

        return evals.size if evals is not None else 0

    
    def makeproj(self, mi, bi):
        print "Projecting to eigenbasis."
        nevals = self.kltrans.modes_m(mi, threshold=self.threshold)[0].size

        if nevals < 1000:
            return self.kltrans.project_sky_matrix_forward_old(mi, self.clarray[bi], self.threshold)
        else:
            return self.kltrans.project_sky_matrix_forward(mi, self.clarray[bi], self.threshold)


    def cacheproj(self, mi):

        ## Don't generate cache for small enough matrices
        if self.num_evals(mi) < 500:
            return

        for i in range(len(self.clarray)):
            print "Creating cache file:" + self._cfile % (mi, i)
            f = h5py.File(self._cfile % (mi, i), 'w')

            projm = self.makeproj(mi, i)

            f.create_dataset('proj', data=projm)
            f.close()

    def delproj(self, mi):

        ## As we don't cache for small matrices, just return
        if self.num_evals(mi) < 500:
            return

        for i in range(len(self.clarray)):
            
            fn = self._cfile % (mi, i)
            if os.path.exists(fn):
                print "Deleting cache file:" + fn
                os.remove(self._cfile % (mi, i))
                

    def getproj(self, mi, bi):

        fn = self._cfile % (mi, bi)

        ## For small matrices or uncached files don't fetch cache, just generate
        ## immediately
        if self.num_evals(mi) < 500 or not os.path.exists:
            proj = self.makeproj(mi, bi)
        else:
            f = h5py.File(fn, 'r')
            proj = f['proj'][:]
            f.close()
            
        return proj

    def fisher_m_old(self, mi):
        
        evals, evecs = self.kltrans.modes_m(mi, self.threshold)

        if evals is not None:
            print "Making fisher (for m=%i)." % mi

            c = [self.makeproj(mi, i) for i in range(len(self.clarray))]
            ci = np.diag(1.0 / (evals + 1.0))
            #ci = np.outer(ci, ci)
            fab = np.array([ [ np.trace(np.dot(np.dot(c_a, ci), np.dot(c_b, ci))) for c_b in c] for c_a in c])
        else:
            print "No evals (for m=%i), skipping." % mi
            l = self.bands.size - 1
            fab = np.zeros((l, l))
        return fab


    def fisher_m(self, mi):
        
        evals, evecs = self.kltrans.modes_m(mi, self.threshold)

        nbands = len(self.bands) - 1
        fab = np.zeros((nbands, nbands), dtype=np.complex128)

        if evals is not None:
            print "Making fisher (for m=%i)." % mi

            self.cacheproj(mi)

            #c = [self.makeproj(mi, clzz) for clzz in self.clarray]
            ci = 1.0 / (evals + 1.0)**0.5
            ci = np.outer(ci, ci)

            for ia in range(nbands):
                c_a = self.getproj(mi, ia)
                fab[ia, ia] = np.sum(c_a * c_a.T * ci**2)
                
                for ib in range(ia):
                    c_b = self.getproj(mi, ib)
                    fab[ia, ib] = np.sum(c_a * c_b.T * ci**2)
                    fab[ib, ia] = np.conj(fab[ia, ib])

            self.delproj(mi)
            
        else:
            print "No evals (for m=%i), skipping." % mi

        return fab


    def _fisher_section(self, mlist):

        return [ (mi, self.fisher_m(mi)) for mi in mlist ]


    def fisher_mpi(self, mlist = None):
        if mlist is None:
            mlist = range(self.telescope.mmax + 1)

        mpart = mpiutil.partition_list_mpi(mlist)

        f_m = self._fisher_section(mpart)

        mpiutil.barrier()

        f_all = mpiutil.world.gather(f_m, root=0)

        if mpiutil.rank0:
            nb = self.bands.shape[0] - 1
            fisher = np.zeros((self.telescope.mmax+1, nb, nb), dtype=np.complex128)
            #print f_all
            for proc_rank in f_all:
                #print proc_rank
                for fm in proc_rank:
                    #print fm
                    fisher[fm[0]] = fm[1]

            f = h5py.File(self.psdir + 'fisher.hdf5', 'w')

            #f.create_dataset('fisher_m/', data=fisher)
            f.create_dataset('fisher_all/', data=np.sum(fisher, axis=0))
            f.create_dataset('bandpower/', data=self.bpower)
            f.create_dataset('bandstart/', data=self.bstart)
            f.create_dataset('bandend/', data=self.bend)
            f.create_dataset('bandcenter/', data=self.bcenter)
            f.close()
            
    
    def fisher_section(self, mlist = None):

        if mlist is None:
            mlist = range(self.telescope.mmax + 1)
            
        mpart = mpiutil.partition_list_mpi(mlist)

        fab_t = np.zeros((self.bands.size - 1, self.bands.size - 1), dtype=np.complex128)

        for mi in mpart:

            fab_m = self.fisher_m(mi)

            fab_t += fab_m

        return fab_t

