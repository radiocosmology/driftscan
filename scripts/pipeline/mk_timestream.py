import argparse
import os
import os.path

import numpy as np
import h5py

from cosmoutils import hputil
from cylsim import mpiutil

import timestream




## Read arguments in.
parser = argparse.ArgumentParser(description="Create the visibility timeseries corresponding to a map.")
parser.add_argument("teldir", help="The telescope directory to use.")
parser.add_argument("outdir", help="Output directory for timeseries.")
parser.add_argument("--map", help="Each map argument is a map which contributes to the timeseries.", action='append')
parser.add_argument("--noise", help="Number of days of co-added data (affects noise amplitude).", metavar='NDAYS', default=0, type=int)
args = parser.parse_args()

# Create timestream object
tstream = timestream.Timestream(args.outdir, args.teldir)

## Read in telescope system
bt = tstream.beamtransfer
tel = tstream.telescope

## Load file
f = h5py.File(args.map[0], 'r')
mapshape = f['map'].shape
f.close()

lmax = tel.lmax
mmax = tel.mmax
nfreq = tel.nfreq
npol = tel.num_pol_sky
projmaps = (len(args.map) > 0)
npix = mapshape[-1]

lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
local_freq = range(sfreq, efreq)

lm, sm, em = mpiutil.split_local(mmax + 1)
local_m = range(sm, em)


col_vis = np.zeros((tel.nbase * tel.num_pol_telescope, lfreq, 2*mmax+1), dtype=np.complex128)



if projmaps:


    row_map = np.zeros((lfreq,) + mapshape[1:], dtype=np.float64)


    ## Read in and sum up the local frequencies of the supplied maps.
    for mapfile in args.map:
        f = h5py.File(mapfile)

        row_map += f['map'][sfreq:efreq]

        f.close()

    # Calculate the alm's for the local sections
    row_alm = hputil.sphtrans_sky(row_map, lmax=lmax).reshape((lfreq, npol * (lmax+1), lmax+1))

    # Reshape into a 2D array for the transposition
    #row_alm = row_alm.reshape(lfreq * npol * (lmax+1), lmax+1)

    # Perform the transposition to distribute different m's across processes. Neat
    # tip, putting a shorter value for the number of columns, trims the array at
    # the same time
    col_alm = mpiutil.transpose_blocks(row_alm, (nfreq, npol * (lmax+1), mmax+1))

    # Transpose and reshape to shift m index first.
    col_alm = np.transpose(col_alm, (2, 0, 1)).reshape(lm, nfreq, npol, lmax+1)

    # Create storage for visibility data
    vis_data = np.zeros((lm, nfreq, bt.ntel), dtype=np.complex128)

    # Iterate over m's local to this process and generate the corresponding
    # visibilities
    for mp, mi in enumerate(range(sm, em)):
        vis_data[mp] = bt.project_vector_sky_to_telescope(mi, col_alm[mp])


    ### WRITE OUT m's HERE


    # Rearrange axes such that frequency is last (as we want to divide
    # frequencies across processors)
    row_vis = vis_data.transpose((0, 2, 1))#.reshape((lm * bt.ntel, nfreq))

    # Parallel transpose to get all m's back onto the same processor
    col_vis_tmp = mpiutil.transpose_blocks(row_vis, ((mmax+1), bt.ntel, nfreq))
    col_vis_tmp = col_vis_tmp.reshape(mmax + 1, 2, tel.nbase * tel.num_pol_telescope, lfreq)


    # Transpose the local section to make the m's the last axis and unwrap the
    # positive and negative m at the same time.
    col_vis[..., 0] = col_vis_tmp[0, 0]
    for mi in range(1, mmax+1):
        col_vis[...,  mi] = col_vis_tmp[mi, 0]
        col_vis[..., -mi] = col_vis_tmp[mi, 1].conj()  # Conjugate only (not (-1)**m - see paper)


    del col_vis_tmp

if args.noise > 0:

    noise_ps = tel.noisepower(np.arange(tel.nbase)[np.newaxis, :], local_freq[:, np.newaxis]).reshape(lfreq, tel.nbase * tel.num_pol_telescope)[:, :, np.newaxis]

    noise_vis = (np.array([1.0, 1.0J]) * np.random.standard_normal(col_vis.shape + (2,))).sum(axis=-1)
    noise_vis *= noise_ps

    col_vis += noise_vis

    del noise_vis


vis_stream = np.fft.ifft(col_vis, axis=-1) * (2 * tel.mmax + 1)
vis_stream = vis_stream.reshape(tel.nbase * tel.num_pol_telescope, lfreq, vis_stream.shape[-1])



## The time samples the visibility is calculated at
tphi = np.linspace(0, 2*np.pi, vis_stream.shape[-1], endpoint=False)

for lfi, fi in enumerate(local_freq):

    if not os.path.exists(tstream._fdir(fi)):
        os.makedirs(tstream._fdir(fi))

    f = h5py.File(tstream._ffile(fi), 'w')

    f.create_dataset('/timestream', data=vis_stream[:, lfi])
    #f.create_dataset('/noise_timeseries', data=nseries)
    f.create_dataset('/phi', data=tphi)
    f.create_dataset('/feedmap', data=tel.feedmap)
    f.create_dataset('/feedconj', data=tel.feedconj)
    f.create_dataset('/feedmask', data=tel.feedmask)
    f.create_dataset('/uniquepairs', data=tel.uniquepairs)
    f.create_dataset('/baselines', data=tel.baselines)

    f.attrs['beamtransfer_path'] = os.path.abspath(args.teldir)

    f.close()

tstream.save()
