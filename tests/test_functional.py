"""Functional test suite for checking integrity of the analysis product
generation."""

import shutil
import os
import subprocess
import tarfile
import sys

import numpy as np
import pytest
import h5py

from pathlib import Path, PurePath
from platform import python_version
from urllib.request import urlretrieve

# Ensure we're using the correct package
_basedir = Path(__file__).parent.resolve()


def approx(x, rel=1e-4, abs=1e-8):
    """Pytest approx with changed defaults."""
    return pytest.approx(x, rel=rel, abs=abs)


def orth_equal_approx(x, y, abs=1e-8):
    """Tet if two basis sets are roughly equal."""

    overlap = np.dot(x, y.T.conj())
    d = np.abs(np.abs(overlap) - np.identity(y.shape[0]))

    return (d < abs).all()


@pytest.fixture(scope="module")
def products_run(tmpdir_factory):

    # If DRIFT_TESTDIR is set then use that
    if "DRIFT_TESTDIR" in os.environ:
        _base = os.environ["DRIFT_TESTDIR"]
    else:
        _base = str(tmpdir_factory.mktemp("testdrift"))

    # allow parallel tests with different python versions without them writing to the same files
    testdir = Path(f"{_base}/tmptestdir/python_{python_version()}")

    # If the data already exists then we don't need to re-run the tests
    if not testdir.exists():
        Path.mkdir(testdir, parents=True)

        shutil.copy("testparams.yaml", testdir / "params.yaml")

        cmd = "drift-makeproducts run params.yaml"

        # If we're not on macOS try running under MPI
        # On macOS this has recently been giving problems when running the MPI
        # job from within pytest
        if sys.platform != "darwin":
            nproc = 2  # Use a fixed number to check that the MPI code works
            cmd = ("mpirun -np %i " % nproc) + cmd

        print("Running test in: %s" % testdir)
        print("Generating products:", cmd)
        retval = subprocess.call(cmd.split(), cwd=testdir)
        print("Done.")
    else:
        retval = 0

    # Can't import this until the subprocess call is done, otherwise the nested
    # MPI environments will fail
    from drift.core import manager as pm

    manager = pm.ProductManager.from_config(testdir / "params.yaml")

    return retval, testdir, manager


@pytest.fixture()
def return_code(products_run):
    return products_run[0]


@pytest.fixture()
def testdir(products_run):
    return products_run[1]


@pytest.fixture()
def manager(products_run):
    return products_run[2]


@pytest.fixture(scope="module")
def saved_products(tmpdir_factory):

    _base = str(tmpdir_factory.mktemp("saved_products"))

    prodfile = PurePath.joinpath(_basedir, "drift_testproducts.tar.gz")

    # Download the test products if they don't exist locally
    if not Path.exists(prodfile):
        print("Downloading test verification data.")
        url = "http://bao.chimenet.ca/testcache/drift_testproducts.tar.gz"
        urlretrieve(url, prodfile)

    with tarfile.open(prodfile, "r:gz") as tf:
        tf.extractall(path=_base)

    def _load(fname):
        path = PurePath.joinpath(Path(_base), Path(fname))

        if not Path.exists(path):
            raise ValueError("Saved product %s does not exist" % path)

        return h5py.File(path, "r")

    return _load


def test_return_code(return_code):
    """Test that the products exited cleanly."""
    code = return_code // 256

    assert code == 0


def test_signal_exit(return_code):
    """Test that the products exited cleanly."""
    signal = return_code % 256

    assert signal == 0


def test_manager(manager, testdir):
    """Check that the product manager code loads properly."""

    mfile = Path(manager.directory)
    tfile = testdir / "testdir"

    assert Path.samefile(mfile, tfile)  # Manager does not see same directory


# This works despite the non-determinism because the elements are small.
def test_beam_m(manager, saved_products):
    """Check the consistency of the m-ordered beams."""

    with saved_products("beam_m_14.hdf5") as f:
        bm_saved = f["beam_m"][:]
    bm = manager.beamtransfer.beam_m(14)

    assert bm_saved.shape == bm.shape  # Beam matrix (m=14) shape has changed

    assert bm == approx(bm_saved)  # Beam matrix (m=14) is incorrect


def test_svd_spectrum(manager, saved_products):
    """Test the SVD spectrum."""
    with saved_products("svdspectrum.hdf5") as f:
        svd_saved = f["singularvalues"][:]

    svd = manager.beamtransfer.svd_all()

    assert svd_saved.shape == svd.shape  # SVD spectrum shapes not equal
    assert svd == approx(svd_saved, rel=1e-3, abs=400)  # SVD spectrum is incorrect


def test_kl_spectrum(manager, saved_products):
    """Check the KL spectrum (for the foregroundless model)."""

    with saved_products("evals_kl.hdf5") as f:
        ev_saved = f["evals"][:]

    ev = manager.kltransforms["kl"].evals_all()

    assert ev_saved.shape == ev.shape  # KL spectrum shapes not equal
    assert ev == approx(ev_saved)  # KL spectrum is incorrect


@pytest.mark.skip(reason="Non determinstic SHT (libsharp), means this doesn't work")
def test_kl_mode(manager, saved_products):
    """Check a KL mode (m=26) for the foregroundless model."""

    with saved_products("ev_kl_m_26.hdf5") as f:
        evecs_saved = f["evecs"][:]

    evals, evecs = manager.kltransforms["kl"].modes_m(26)

    assert evecs_saved.shape == evecs.shape  # KL mode shapes not equal
    assert orth_equal_approx(evecs, evecs_saved, abs=1e-5)  # KL mode is incorrect


@pytest.mark.skip(reason="Non determinstic SHT (libsharp), means this doesn't work")
def test_dk_mode(manager, saved_products):
    """Check a KL mode (m=38) for the model with foregrounds."""

    with saved_products("ev_dk_m_38.hdf5") as f:
        evecs_saved = f["evecs"][:]

    evals, evecs = manager.kltransforms["dk"].modes_m(38)

    assert evecs_saved.shape == evecs.shape  # DK mode shapes not equal
    assert evecs == approx(evecs_saved)  # DK mode is incorrect


def test_kl_fisher(manager, saved_products):
    """Test the Fisher matrix consistency. Use an approximate test as Monte-Carlo."""

    with saved_products("fisher_kl.hdf5") as f:
        fisher_saved = f["fisher"][:]
        bias_saved = f["bias"][:]

    ps = manager.psestimators["ps1"]
    fisher, bias = ps.fisher_bias()

    assert fisher_saved.shape == fisher.shape  # KL Fisher shapes not equal
    assert fisher == approx(fisher_saved, rel=3e-2, abs=1)  # KL Fisher is incorrect

    assert bias_saved.shape == bias.shape  # KL bias shapes not equal
    assert bias == approx(bias_saved, rel=3e-2, abs=1)  # KL bias is incorrect.


def test_dk_fisher(manager, saved_products):
    """Test the DK Fisher matrix consistency. Use an approximate test as Monte-Carlo."""

    with saved_products("fisher_dk.hdf5") as f:
        fisher_saved = f["fisher"][:]
        bias_saved = f["bias"][:]

    ps = manager.psestimators["ps2"]
    fisher, bias = ps.fisher_bias()

    assert fisher_saved.shape == fisher.shape  # DK Fisher shapes not equal
    assert fisher == approx(fisher_saved, rel=3e-2, abs=1)  # DK Fisher is incorrect

    assert bias_saved.shape == bias.shape  # DK bias shapes not equal
    assert bias == approx(bias_saved, rel=3e-2, abs=1)  # DK bias is incorrect.


@pytest.mark.skip(reason="Non determinstic SHT (libsharp), means this doesn't work")
def test_svd_mode(manager, saved_products):
    """Test that the SVD modes are correct."""

    with saved_products("svd_m_14.hdf5") as f:
        svd_saved = f["beam_svd"][:]
        invsvd_saved = f["invbeam_svd"][:]
        ut_saved = f["beam_ut"][:]

    svd = manager.beamtransfer.beam_svd(14)
    invsvd = manager.beamtransfer.invbeam_svd(14)
    ut = manager.beamtransfer.beam_ut(14)

    assert svd_saved.shape == svd.shape  # SVD beam matrix (m=14) shape has changed
    assert svd == approx(svd_saved)  # SVD beam matrix (m=14) is incorrect
    assert invsvd == approx(invsvd_saved)  # Inverse SVD beam matrix (m=14) is incorrect
    assert ut == approx(ut_saved)  # SVD UT matrix (m=14) is incorrect


def test_dk_spectrum(manager, saved_products):
    """Check the KL spectrum (for the model with foregrounds)."""

    with saved_products("evals_dk.hdf5") as f:
        ev_saved = f["evals"][:]

    ev = manager.kltransforms["dk"].evals_all()

    assert ev_saved.shape == ev.shape  # DK spectrum shapes not equal
    assert ev == approx(ev_saved, rel=1e-2)  # DK spectrum is incorrect
