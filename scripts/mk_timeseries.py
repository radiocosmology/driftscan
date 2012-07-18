import argparse

import numpy as np
import h5py

import healpy

from cosmoutils import hputil
from cylsim import beamtransfer, blockla


def rotate_phi(map, dphi):
    """Rotate a map from galactic co-ordinates into celestial co-ordinates.

    Parameters
    ----------
    map : np.ndarray
        Healpix map.
    x, y : {'C', 'G', 'E'}
        Coordinate system to transform to/from. Celestial, 'C'; Galactic 'G'; or
        Ecliptic 'E'.

    Returns
    -------
    rotmap : np.ndarray
        The rotated map.
    """

    angpos = hputil.ang_positions(healpy.npix2nside(map.size))
    theta, phi = angpos[:, 0], ((angpos[:, 1] - dphi) % (2*np.pi))
    return healpy.get_interp_val(map, theta, phi)

## Read arguments in.
parser = argparse.ArgumentParser(description="Create the visibility timeseries corresponding to a map.")
parser.add_argument("teldir", help="The telescope directory to use.")
parser.add_argument("mapfile", help="Input map.")
parser.add_argument("outfile", help="Output file for timeseries.")
parser.add_argument("--freq", help="Index of frequency slice to use.", default=0, type=int)
args = parser.parse_args()

## Read in cylinder system
bt = beamtransfer.BeamTransfer(args.teldir)

freq_ind = args.freq

## Load file
f = h5py.File(args.mapfile, 'r')
skymap = f['map'][:] if len(f['map'].shape) == 1 else f['map'][freq_ind]
skymap = healpy.smoothing(healpy.ud_grade(skymap, 128), degree=True, fwhm=1.0)
f.close()

tel = bt.telescope

## Perform the spherical harmonic transform, and expand it to include the negative-m
almh = hputil.sphtrans_real(skymap, lmax=tel.lmax)[:, :(tel.mmax+1)]
alm = hputil._make_full_alm(almh)

## Fetch the beam transfers
bf0 = bt.beam_freq(freq_ind, single=True)

## Rotate the array axes such that m-index is first
bf0 = np.squeeze(np.rollaxis(bf0, -1))
alm = np.squeeze(np.rollaxis(alm, -1))

## Calculate the visibility fourier modes
vis = blockla.multiply_dm_v(bf0, alm).T

## Transform to find the visibility time series (the multiplicative factor
## corrects to get the normalisation)
vist = np.fft.ifft(vis) * (2 * tel.mmax + 1)

## The time samples the visibility is calculated at
tphi = np.linspace(0, 2*np.pi, vist.shape[1], endpoint=False)

tseries = np.zeros(tel.feedmap.shape + tphi.shape, dtype=np.complex128)

wm = np.where(tel.feedmask)
wc = np.where(tel.feedconj)
tseries[wm] = vist[tel.feedmap[wm]]

tseries[wc] = tseries[wc].conj()



#f = h5py.File(outfile, 'w')

#f.create_dataset('/visibilities', data=vist)
#f.create_dataset('/phisamples', data=tphi)
#f.create_dataset('/baselines', data=tel.baselines)

#telf.attr['frequency'] = tel.frequencies[freq_ind]

tel._init_trans(128)
bm27 = tel._beam_map_single(3, 0)

vism = np.zeros(tphi.size, dtype=np.complex128)

for i in range(tphi.size):
    print i
    vism[i] = np.mean(rotate_phi(bm27, tphi[i]) * skymap) * 4*np.pi


#vism = np.mean(bm27 * skymap) * 4*np.pi


