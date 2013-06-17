#!/bin/bash

## Copy over the products we wil test against. Only use when updating the
## products after verifying that they are correct.

# Copy over beam properties
cp tmptestdir/testdir/bt/beam_f/2/beam.hdf5 saved_products/beam_f_2.hdf5
cp tmptestdir/testdir/bt/beam_m/14/beam.hdf5 saved_products/beam_m_14.hdf5
cp tmptestdir/testdir/bt/beam_m/14/svd.hdf5 saved_products/svd_m_14.hdf5
cp tmptestdir/testdir/bt/svdspectrum.hdf5 saved_products/svdspectrum.hdf5

# Copy over eigenvalue products
cp tmptestdir/testdir/bt/dk/ev_m_33.hdf5 saved_products/ev_dk_m_33.hdf5
cp tmptestdir/testdir/bt/kl/ev_m_26.hdf5 saved_products/ev_kl_m_26.hdf5
cp tmptestdir/testdir/bt/dk/evals.hdf5 saved_products/evals_dk.hdf5
cp tmptestdir/testdir/bt/kl/evals.hdf5 saved_products/evals_kl.hdf5

# Copy Fisher matrices
cp tmptestdir/testdir/bt/dk/ps2/fisher.hdf5 saved_products/fisher_dk.hdf5
cp tmptestdir/testdir/bt/kl/ps1/fisher.hdf5 saved_products/fisher_kl.hdf5