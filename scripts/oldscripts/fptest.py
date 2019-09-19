# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from drift.core import focalplane, beamtransfer, kltransform, psestimation

import healpy

fpa = focalplane.FocalPlaneArray(latitude=0)

fpa.beam_spacing_u = 0.0
fpa.beam_spacing_v = 0.0

fpa.beam_num_u = 1
fpa.beam_num_v = 1

fpa.beam_size = 5.0

fpa.positive_m_only = True

fpa.num_freq = 5
fpa.freq_lower = 460.0
fpa.freq_upper = 540.0


bt = beamtransfer.BeamTransfer("cylinder/fpatest/", telescope=fpa)
# bt.generate()

klt = kltransform.KLTransform(bt)

# klt.generate()

pse = psestimation.PSEstimation(klt)

pse.bands = np.linspace(0.0, 1.0, 5)

# pse.genbands()

# pse.fisher_mpi()
