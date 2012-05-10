
import numpy as np
import h5py
import healpy

from cosmoutils import hputil

from cylsim import kltransform
from cylsim import mpiutil


class Projector(object):

    mapfiles = []
    thresholds = None

    stems = []

    evec_proj = True
    beam_proj = True

    nside_out = 256

    def __init__(self, klt):

        self.kltransform = klt
        self.beamtransfer = klt.beamtransfer
        self.telescope = klt.beamtransfer.telescope


    def generate(self):

        for ind, mfile in enumerate(self.mapfiles):

            stem = self.stems[ind]

            print "============\nProjecting file %s\n============\n" % mfile

            ## Load map and perform spherical harmonic transform
            if mpiutil.rank0:
                # Calculate alm's and broadcast
                print "Read in skymap."
                f = h5py.File(mfile)
                skymap = f['map'][:]
                f.close()
                nside = healpy.get_nside(skymap[0])
                alm = hputil.sphtrans_sky(skymap, lmax=self.telescope.lmax)
            else:
                alm = None

            ## Function to write out a map from the collected array of alms
            def _write_map_from_almarray(almp, filename, attrs=None):
                if mpiutil.rank0:

                    almp = np.squeeze(np.transpose(almp, axes=(2, 1, 3, 0)))
                    almf = np.zeros((almp.shape[0], almp.shape[1], almp.shape[1]), dtype=np.complex128)
                    almf[:, :, :almp.shape[2]] = almp

                    pmap = hputil.sphtrans_inv_sky(almf, self.nside_out)

                    f = h5py.File(filename, 'w')
                    if attrs is not None:
                        for key, val in attrs.items():
                            f.attrs[repr(key)] = val
                    f.create_dataset('/map', data=pmap)
                    f.close()

            ## Broadcast set of alms to the world
            alm = mpiutil.world.bcast(alm, root=0)
            mlist = range(self.kltransform.telescope.mmax+1)
            nevals = self.beamtransfer.ntel * self.beamtransfer.nfreq


            ## Construct beam projection of map
            if self.beam_proj:

                def proj_beam(mi):
                    print "Projecting %i" % mi
                    bproj = self.beamtransfer.project_vector_forward(mi, alm[:, :, mi]).flatten()
                    return self.beamtransfer.project_vector_backward(mi, bproj)

                shape = (self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax+1)
                almp = kltransform.collect_m_array(mlist, proj_beam, shape, np.complex128)
                _write_map_from_almarray(almp, stem + "_beam.hdf5")


            ## Construct EV projection of map
            if self.evec_proj:

                def proj_evec(mi):
                    ## Worker function for mapping over list and projecting onto signal modes.
                    print "Projecting %i" % mi
                    p2 = np.zeros(nevals, dtype=np.complex128)
                    if self.kltransform.modes_m(mi)[0] is not None:
                        p1 = self.kltransform.project_sky_vector_forward(mi, alm[:, :, mi])
                        p2[-p1.size:] = p1

                    return p2

                shape = (nevals,)
                evp = kltransform.collect_m_array(mlist, proj_evec, shape, np.complex128)

                f = h5py.File(stem + "_ev.hdf5", 'w')
                f.create_dataset("/evec_proj", data=evp)
                f.close()

            ## Iterate over noise cuts and filter out noise.
            for cut in self.thresholds:

                def filt_kl(mi):
                    ## Worker function for mapping over list and projecting onto signal modes.
                    print "Projecting %i" % mi

                    mvals, mvecs = self.kltransform.modes_m(mi, threshold=cut)
                    
                    if mvals is None:
                        return None

                    ev_vec = self.kltransform.project_sky_vector_forward(mi, alm[:, :, mi], threshold=cut)
                    tel_vec = self.kltransform.project_tel_vector_backward(mi, ev_vec, threshold=cut)

                    alm2 = self.beamtransfer.project_vector_backward(mi, tel_vec)

                    return alm2

                shape = (self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax+1)
                almp = kltransform.collect_m_array(mlist, filt_kl, shape, np.complex128)
                _write_map_from_almarray(almp, stem + ("_kl_%f.hdf5" % cut), {'threshold' : cut})






