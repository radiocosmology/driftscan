from drift.core import beamtransfer, manager
from drift.pipeline import timestream

from simplearray import DishArray

### Make the analysis products for the telescope. This examples focuses only
### on the m-mode products for mapmaking

# Create telescope object and set zenith
tel = DishArray(latitude=30.0, longitude=0.0)

# Create Beam Transfer manager, and generate products
bt = beamtransfer.BeamTransfer("pydriver/btdir/", telescope=tel)
bt.generate()


### Simulate and make a map froma timestream

# Create an empty ProductManager
m = manager.ProductManager()

# Set the Beam Transfers
m.beamtransfer = bt

# Create a timestream with no noise (ndays=0) from a given map (could be a list of maps)
ts = timestream.simulate(m, "pydriver/ts1/", ["simulated_map.hdf5"], ndays=0)

# Make m-mode from the timestream
ts.generate_mmodes()

# Make a Healpix map from the m-modes (with NSIDE=256)
ts.mapmake_full(256, "observed_map.hdf5")
