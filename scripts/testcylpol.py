
## Matplotlib imports for plotting figures
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
cm = matplotlib.cm.gray

import time
import numpy as np

import healpy

from cylsim import visibility
from cylsim import hputil
from cylsim.plotutil import *




# Specify latitute (by specifying zenith in spherical polars).
zenith = np.array([1.0, np.pi / 4, 0.0])

# Set cylinder properties
cylwidth = 10.0
feedspacing = 2.0
ncyl = 2
nfeed = 3


# Specify feed alignments
feedx = np.array([1.0, 0.0])  # Aligned in U-direction (i.e points E)
feedy = np.array([0.0, 1.0])  # Aligned in V-direction (i.e points N)

# Specify effective cylinder widths for each feed
cylwidthx = cylwidth
cylwidthy = cylwidth


# Estimate maximum (l,m) values required
mmax = 2*np.pi*max(cylwidthx, cylwidthy)*ncyl
lmax = (mmax**2 + (2*np.pi*feedspacing*nfeed)**2)**0.5

# Estimate corresponding sky map resolution (i.e. Healpix nside)
nside = int(2**(np.ceil(np.log((lmax + 1) / 3.0) / np.log(2.0))))

print "l-max %i, m-max %i" % (lmax, mmax)
print "Nside=%i is required." % nside

# Fetch angular positions of pixels in map.
angpos = hputil.ang_positions(nside)

# Calculate horizon function
h = visibility.horizon(angpos, zenith)
# Calculate angular beam pattern for each feed
bx = visibility.cylinder_beam(angpos, zenith, cylwidthx)
by = visibility.cylinder_beam(angpos, zenith, cylwidthy)

# Polarisation projections
pIQUxx = visibility.pol_IQU(angpos, zenith, feedx, feedx)
pIQUxy = visibility.pol_IQU(angpos, zenith, feedx, feedy)
pIQUyy = visibility.pol_IQU(angpos, zenith, feedy, feedy)

st = time.time()

# Calculate fringe for the specified visibility
f1 = visibility.fringe(angpos, zenith, [cylwidth, 3*feedspacing])

# Overall complex visibility on the sky
cvIQUxx = h * bx * bx * f1 * pIQUxx
cvIQUxy = h * bx * by * f1 * pIQUxy
cvIQUyy = h * by * by * f1 * pIQUyy

# Use healpix to transform into alm space.
almTEBxx = hputil.sphtrans_complex_pol(cvIQUxx, centered = True, lmax = int(lmax))
almTEBxy = hputil.sphtrans_complex_pol(cvIQUxy, centered = True, lmax = int(lmax))
almTEByy = hputil.sphtrans_complex_pol(cvIQUyy, centered = True, lmax = int(lmax))

et = time.time()

print "Time for visibility: %f" % (et - st)


f = plt.figure(1, figsize = [20,12])
f.subplots_adjust(left=0.04, right=0.98, bottom=0.03, top =0.94, wspace=0.12, hspace=0.1)

# Plot polarisation projections
f.suptitle("Polarisation projections")
mollview_polfeed([pIQUxx, pIQUxy, pIQUyy], min = -1, max = 1, cmap = cm)
f.savefig("polproj.png")
f.clf()

# Plot visibilities
f.suptitle("Visibilities - Real Part")
mollview_polfeed([cvIQUxx, cvIQUxy, cvIQUyy], trans = np.real, min = -1, max = 1, cmap = cm)
f.savefig("polvis.png")
f.clf()

# Plot alms
f.subplots_adjust(left=0.04, right=0.98, bottom=0.03, top =0.94, wspace=0.12, hspace=0.05)
f.suptitle("Beam Transfer in $(l,m)$ space - Real Part")
imshow_polfeed([almTEBxx, almTEBxy, almTEByy], xlabel = "$m$", ylabel="$l$", trans = np.real, cmap = cm, extent = [-lmax, lmax, 0, lmax], pollabel = ['$T$', '$E$', '$B$'], origin='lower')
f.savefig("pol_alm.png")
f.clf()
