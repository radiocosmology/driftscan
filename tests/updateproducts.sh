#!/bin/bash

## Copy over the products we wil test against. Only use when updating the
## products after verifying that they are correct.

TEMP_PROD_DIR=$(mktemp -d -t tempprodXXX)

# Copy over beam properties
cp $1/testdir/bt/beam_m/14/beam.hdf5 $TEMP_PROD_DIR/beam_m_14.hdf5
cp $1/testdir/bt/beam_m/14/svd.hdf5 $TEMP_PROD_DIR/svd_m_14.hdf5
cp $1/testdir/bt/svdspectrum.hdf5 $TEMP_PROD_DIR/svdspectrum.hdf5

# Copy over eigenvalue products
cp $1/testdir/bt/dk/ev_m_38.hdf5 $TEMP_PROD_DIR/ev_dk_m_38.hdf5
cp $1/testdir/bt/kl/ev_m_26.hdf5 $TEMP_PROD_DIR/ev_kl_m_26.hdf5
cp $1/testdir/bt/dk/evals.hdf5 $TEMP_PROD_DIR/evals_dk.hdf5
cp $1/testdir/bt/kl/evals.hdf5 $TEMP_PROD_DIR/evals_kl.hdf5

# Copy Fisher matrices
cp $1/testdir/bt/dk/ps2/fisher.hdf5 $TEMP_PROD_DIR/fisher_dk.hdf5
cp $1/testdir/bt/kl/ps1/fisher.hdf5 $TEMP_PROD_DIR/fisher_kl.hdf5

# Create the tarball
cd $TEMP_PROD_DIR
tar czf drift_testproducts.tar.gz *
cd ~-
mv $TEMP_PROD_DIR/drift_testproducts.tar.gz .

rm -r $TEMP_PROD_DIR