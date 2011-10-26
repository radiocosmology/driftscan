import h5py

import numpy as np

import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt


f = h5py.File('evals1f.hdf5')


evals = f['evals'][:]
ac = f['add_const'][:]

mmax = (evals.shape[0] - 1) / 2
mn = evals.shape[1]

evals2 = np.empty_like(evals)
evals2[mmax:] = evals[:(mmax+1)]
evals2[:mmax] = evals[(mmax+1):]

evals = evals2.T

f.close()

fig = plt.figure()

ax = fig.add_subplot(111)

img = ax.imshow(np.log10(3e-6 + evals), aspect=(evals.shape[1] * 1.0 / evals.shape[0]), extent=(-mmax, mmax, 0, mn), vmin=-3.0, vmax=2.0)
#img = ax.imshow((3e-6 + evals), aspect=(evals.shape[1] * 1.0 / evals.shape[0]), extent=(-mmax, mmax, 0, mn), norm = matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e4))

#cont1 = ax.contour(evals[:,:(3*mmax/4)], levels=(1e0, 1e1), extent=(-mmax, -mmax / 4, mn, 0))
cont1 = ax.contour(evals[:,:], levels=(1e0, 1e1), extent=(-mmax, mmax, mn, 0))
#cont2 = ax.contour(evals[:,(3*mmax/4):], levels=(1e0, 1e1), extent=(-mmax / 4, mmax, mn, 0))

ax.clabel(cont1, fontsize=10, fmt= "%1.1f")


ax.set_xlabel("$m$-mode")
ax.set_ylabel("Eigenmode")
ax.set_title("Signal/Noise Power Ratios")

plt.colorbar(img, format="$10^{%i}$", ticks=[-3,-2,-1, 0, 1, 2])



fig.savefig("evalsmap.pdf")
