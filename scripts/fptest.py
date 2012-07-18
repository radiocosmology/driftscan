from cylsim import focalplane, beamtransfer, kltransform, psestimation

import healpy

fpa = focalplane.FocalPlaneArray(latitude=0)

fpa.beam_spacing_u = 0.0
fpa.beam_spacing_v = 2.0

fpa.beam_num_u = 1
fpa.beam_num_v = 10

fpa.beam_size = 2.0

fpa.positive_m_only = True

fpa.num_freq = 5
fpa.freq_lower = 400.0
fpa.freq_upper = 450.0

fpa._init_trans(128)

bt = beamtransfer.BeamTransfer('cylinder/fpatest/', telescope=fpa)
bt.generate()

klt = kltransform.KLTransform(bt)

klt.generate()

pse = psestimation.PSEstimation(klt)

pse.bands = np.linspace(0.0, 1.0, 5)

pse.genbands()

pse.fisher_mpi()