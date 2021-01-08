import numpy as np
import healpy

from cora.util import coord

from drift.telescope import cylinder
from drift.telescope import visibility, beamtransfer, kltransform


class SeoTelescope(cylinder.UnpolarisedCylinderTelescope):

    vwidth = 5.0

    num_cylinders = 1
    cylinder_width = 32.0
    num_feeds = 16
    feed_spacing = 2.0

    positive_m_only = True

    freq_lower = 400.0
    freq_upper = 450.0
    num_freq = 25

    ndays = 1825.0

    l_boost = 1.0

    def __init__(self):
        super(cylinder.UnpolarisedCylinderTelescope, self).__init__(latitude=0.0)

    def beam(self, feed, freq):

        # bm = visibility.cylinder_beam(self._angpos, self.zenith,
        #                               self.cylinder_width / self.wavelengths[freq])

        # sigma = np.radians(self.fwhm) / (8.0*np.log(2.0))**0.5 / (self.frequencies[freq] / 150.0)

        #     pointing = np.array([np.pi / 2.0 - np.radians(self.pointing), self.zenith[1]])

        #     x2 = (1.0 - coord.sph_dot(self._angpos, pointing)**2) / (4*sigma**2)
        #     self._bc_map = np.exp(-x2)

        #     self._bc_freq = freq
        #     self._bc_nside = self._nside

        uhatc, vhatc = visibility.uv_plane_cart(self.zenith)

        ## Note sinc function is normalised hence lack of pi
        bmh = np.sinc(
            np.inner(
                coord.sph_to_cart(self._angpos),
                self.cylinder_width * uhatc / self.wavelengths[freq],
            )
        )

        bmv = np.where(
            np.abs(np.inner(coord.sph_to_cart(self._angpos), vhatc)) * 180.0 / np.pi
            < self.vwidth / 2.0,
            np.ones_like(bmh),
            np.zeros_like(bmh),
        )

        bmv = healpy.smoothing(bmv, degree=True, fwhm=(self.vwidth / 10.0))

        return bmv * bmh


stel = SeoTelescope()
stel._init_trans(256)


bt = beamtransfer.BeamTransfer("./seocomp/", telescope=stel)
bt.generate()

klt = kltransform.KLTransform(bt)

klt.use_foreground = False
klt.subset = False
klt.inverse = False

klt.generate()
