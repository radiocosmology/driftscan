
## Matplotlib imports for plotting figures
import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt
cm = matplotlib.cm.gray

import numpy as np

import healpy

from cylsim import hputil
from cylsim import visibility



# Specify latitute (by specifying zenith in spherical polars).
zenith = np.array([np.pi / 4, 0.0])

# Set cylinder properties
cylwidth = 10.0
feedspacing = 1.5
ncyl = 2
nfeed = 4

# Estimate maximum (l,m) values required
mmax = 2*np.pi*cylwidth*ncyl
lmax = (mmax**2 + (2*np.pi*feedspacing*nfeed)**2)**0.5

# Estimate corresponding sky map resolution (i.e. Healpix nside)
#nside = 2*int(2**(np.ceil(np.log((lmax + 1) / 3.0) / np.log(2.0))))
nside = 512

print "l-max %i, m-max %i" % (lmax, mmax)
print "Nside=%i is required." % nside

# Fetch angular positions of pixels in map.
angpos = hputil.ang_positions(nside)

# Calculate horizon function
h1 = visibility.horizon(angpos, zenith)
# Calculate angular sensitivy of beam
b1 = visibility.cylinder_beam(angpos, zenith, cylwidth)**2
# Calculate fringe for the specified visibility
f1 = visibility.fringe(angpos, zenith, [cylwidth, 4*feedspacing])

# Overall complex visibility on the sky
cvis = h1*b1*f1

# Use healpix to transform into alm space.
alm = hputil.sphtrans_complex(cvis, centered = True, lmax = int(lmax))


f = plt.figure(1)

healpy.mollview(h1, fig=1, min=-1, max=1, title = 'Horizon', cmap = cm)
f.savefig("horizon.pdf")
f.clf()

healpy.mollview(b1, fig=1, min=-1, max=1, title = 'Beam', cmap = cm)
f.savefig("beam.pdf")
f.clf()


healpy.mollview((h1*b1).real, fig=1, min=-1, max=1, title = 'Horizon + Beam', cmap = cm)
f.savefig("hbeam.pdf")
f.clf()

healpy.mollview((h1*b1*f1).real, fig=1, min=-1, max=1, title = 'Fringe', cmap = cm)
f.savefig("fringe.pdf")
f.clf()



healpy.mollview(cvis.real, fig=1, min=-1, max=1, title = 'Fringe', cmap = cm)
f.savefig("visibility.pdf")
f.clf()

ax = f.add_subplot(111)

ax.imshow(alm.real, extent = (-lmax, lmax, 0, lmax), origin='lower', cmap=cm)
ax.set_title("Beam Transfer in $(l,m)$ space - Real Part")
ax.set_xlabel("$m$")
ax.set_ylabel("$l$")

f.savefig("alm.pdf")

cgconv = healpy.rotator.get_coordconv_matrix(('C', 'G'))

rotconv = healpy.rotator.rotateDirection(cgconv[0], theta=angpos[:,0], phi=angpos[:,1])

haslam = healpy.fitsfunc.read_map("/Users/richard/Downloads/haslam.fits")
haslamc = healpy.get_interp_val(haslam, theta=rotconv[0], phi=rotconv[1])


healpy.mollview(haslamc.real, fig=1, title = 'Sky')
f.savefig("haslam.pdf")
f.clf()



healpy.mollview((haslamc * f1 * h1 * b1).real, fig=1, title = 'Observed sky')
f.savefig("haslamvis.pdf")
f.clf()

