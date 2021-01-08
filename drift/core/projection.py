import os.path
import shutil

import numpy as np
import h5py
import healpy

from caput import config, mpiutil

from cora.util import hputil

from drift.core import kltransform


class Projector(config.Reader):

    maps = config.Property(proptype=list, default=[])
    thresholds = config.Property(
        proptype=(lambda x: [float(item) for item in list(x)]), default=[]
    )

    evec_proj = config.Property(proptype=bool, default=True)
    beam_proj = config.Property(proptype=bool, default=True)

    copy_orig = config.Property(proptype=bool, default=False)

    nside = config.Property(proptype=int, default=256)

    def __init__(self, klt):

        self.kltransform = klt
        self.beamtransfer = klt.beamtransfer
        self.telescope = klt.beamtransfer.telescope

    def generate(self):

        for mentry in self.maps:

            mfile = mentry["file"]
            stem = mentry["stem"]

            if mpiutil.rank0 and not os.path.exists(os.path.dirname(stem)):
                os.makedirs(os.path.dirname(stem))

            mpiutil.barrier()

            if self.copy_orig:
                shutil.copy(mfile, stem + "orig.hdf5")

            print("============\nProjecting file %s\n============\n" % mfile)

            ## Load map and perform spherical harmonic transform
            if mpiutil.rank0:
                # Calculate alm's and broadcast
                print("Read in skymap.")
                f = h5py.File(mfile, "r")
                skymap = f["map"][:]
                f.close()
                nside = healpy.get_nside(skymap[0])
                alm = hputil.sphtrans_sky(skymap, lmax=self.telescope.lmax)
            else:
                alm = None

            ## Function to write out a map from the collected array of alms
            def _write_map_from_almarray(almp, filename, attrs=None):
                if mpiutil.rank0:

                    almp = np.squeeze(np.transpose(almp, axes=(2, 1, 3, 0)))
                    almf = np.zeros(
                        (almp.shape[0], almp.shape[1], almp.shape[1]),
                        dtype=np.complex128,
                    )
                    almf[:, :, : almp.shape[2]] = almp

                    pmap = hputil.sphtrans_inv_sky(almf, self.nside)

                    f = h5py.File(filename, "w")
                    if attrs is not None:
                        for key, val in attrs.items():
                            f.attrs[repr(key)] = val
                    f.create_dataset("/map", data=pmap)
                    f.close()

            mpiutil.barrier()

            ## Broadcast set of alms to the world
            alm = mpiutil.world.bcast(alm, root=0)
            mlist = list(range(self.kltransform.telescope.mmax + 1))
            nevals = self.beamtransfer.ntel * self.beamtransfer.nfreq

            ## Construct beam projection of map
            if self.beam_proj:

                def proj_beam(mi):
                    print("Projecting %i" % mi)
                    bproj = self.beamtransfer.project_vector_forward(
                        mi, alm[:, :, mi]
                    ).flatten()
                    return self.beamtransfer.project_vector_backward(mi, bproj)

                shape = (
                    self.telescope.nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                almp = kltransform.collect_m_array(
                    mlist, proj_beam, shape, np.complex128
                )
                _write_map_from_almarray(almp, stem + "beam.hdf5")

            mpiutil.barrier()

            ## Construct EV projection of map
            if self.evec_proj:

                def proj_evec(mi):
                    ## Worker function for mapping over list and projecting onto signal modes.
                    print("Projecting %i" % mi)
                    p2 = np.zeros(nevals, dtype=np.complex128)
                    if self.kltransform.modes_m(mi)[0] is not None:
                        p1 = self.kltransform.project_sky_vector_forward(
                            mi, alm[:, :, mi]
                        )
                        p2[-p1.size :] = p1

                    return p2

                shape = (nevals,)
                evp = kltransform.collect_m_array(
                    mlist, proj_evec, shape, np.complex128
                )

                if mpiutil.rank0:
                    f = h5py.File(stem + "ev.hdf5", "w")
                    f.create_dataset("/evec_proj", data=evp)
                    f.close()

            mpiutil.barrier()

            ## Iterate over noise cuts and filter out noise.
            for cut in self.thresholds:

                def filt_kl(mi):
                    ## Worker function for mapping over list and projecting onto signal modes.
                    print("Projecting %i" % mi)

                    mvals, mvecs = self.kltransform.modes_m(mi, threshold=cut)

                    if mvals is None:
                        return None

                    ev_vec = self.kltransform.project_sky_vector_forward(
                        mi, alm[:, :, mi], threshold=cut
                    )
                    tel_vec = self.kltransform.project_tel_vector_backward(
                        mi, ev_vec, threshold=cut
                    )

                    alm2 = self.beamtransfer.project_vector_backward(mi, tel_vec)

                    return alm2

                shape = (
                    self.telescope.nfreq,
                    self.telescope.num_pol_sky,
                    self.telescope.lmax + 1,
                )
                almp = kltransform.collect_m_array(mlist, filt_kl, shape, np.complex128)
                _write_map_from_almarray(
                    almp, stem + ("kl_%g.hdf5" % cut), {"threshold": cut}
                )

                mpiutil.barrier()
