## An example of how to calculate the KL modes, and Power spectrum fisher matrix for a cylinder
##
import os

from drift.telescope import cylinder
from drift.core import beamtransfer
from drift.core import kltransform

cyl = cylinder.UnpolarisedCylinderTelescope()

# Check to see if there is a SCRATCH directory and use it if possible.
teldir = (
    os.environ["SCRATCH"] if "SCRATCH" in os.environ else "."
) + "/cylinder/voltest"

# Set the measured frequencies of the telescope
cyl.num_freq = 10
cyl.freq_lower = 400.0
cyl.freq_upper = 450.0

# Set the properties of the cylinders
cyl.cylinder_width = 5.0
cyl.num_cylinders = 2
cyl.feed_spacing = 0.5
cyl.num_feeds = 5

# Set the thermal noise (T_sys flat across spectrum)
cyl.tsys_flat = 10.0
# cyl.tsys_flat = 0.0
cyl.l_boost = 1.2

cyl.auto_correlations = False

cyl.positive_m_only = False
# cyl.in_cylinder = False

# Generate all the beam transfer functions
bt = beamtransfer.BeamTransfer(teldir, telescope=cyl)
bt.generate()

# Perform the KL transform (saving all modes)
klt = kltransform.KLTransform(bt)
klt.subset = False
klt.inverse = True
klt.use_foregrounds = True
klt.generate()

# Performing DoubleKL transform
# dk = doublekl.DoubleKL(bt, subdir="dk1")
# dk.subset = False
# dk.generate()
# Perform the power spectrum estimations
# ps = psmc.PSMonteCarlo(klt)
# ps.genbands()
# ps.fisher_mpi()
