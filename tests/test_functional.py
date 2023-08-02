"""Functional test suite for checking integrity of the analysis product generation.

The tests using the KL spectrum are effected by changes to the default cosmology/power
spectrum in cora, and there is no easy way to pin this during the tests. Best option is
to just update the products *if* the defaults change.

Also, due to problems with double forking MPI processes this test suite can only run
once within a single process. In particular you can't run this test suite followed by
`test_functional_skip.py`
"""

import shutil
import os
import subprocess
import tarfile

import numpy as np
import pytest
import h5py

from pathlib import Path
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
def test_dir(tmpdir_factory):
    # If DRIFT_TESTDIR is set then use that
    if "DRIFT_TESTDIR" in os.environ:
        _base = Path(os.environ["DRIFT_TESTDIR"])
    else:
        _base = Path(str(tmpdir_factory.mktemp("testdrift")))

    return _base


# NOTE: we can't run this twice in the same test run. I think this is because
# MPI refuses to startup if you try to fork and MPI process from within an MPI
# process. It works once because the MPI env is not initialised until the
# `ProductManager` call which occurs *after* the product generation.
def _gen_prod(output_dir: Path, config: Path):
    # If the data already exists then we don't need to re-run the tests
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

        shutil.copy(config, output_dir / "params.yaml")

        cmd = "drift-makeproducts run params.yaml"

        # If we're not on macOS try running under MPI
        # On macOS this has recently been giving problems when running the MPI
        # job from within pytest
        if "DRIFT_NO_MPI" not in os.environ:
            nproc = 2  # Use a fixed number to check that the MPI code works
            cmd = ("mpirun -np %i --oversubscribe -bind-to none " % nproc) + cmd

        print(f"Running test in: {output_dir}")
        print("Generating products:", cmd)
        proc = subprocess.run(cmd.split(), cwd=output_dir)
        print("Done.")
        retval = proc.returncode
    else:
        retval = 0

    # Can't import this until the subprocess call is done, otherwise the nested
    # MPI environments will fail
    from drift.core import manager as pm

    manager = pm.ProductManager.from_config(output_dir / "params.yaml")

    return retval, output_dir, manager


@pytest.fixture(scope="module")
def products_run(test_dir):
    # Generate the standard test products
    return _gen_prod(
        test_dir / f"prod_python_{python_version()}", _basedir / "testparams.yaml"
    )


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
def saved_products(test_dir: Path):
    _base = test_dir / "saved_products"
    _base.mkdir(parents=True, exist_ok=True)

    # Download the products into the root directory such that they don't need
    # to be downloaded on each test run
    prodfile = _basedir / "drift_testproducts.tar.gz"

    # Download the test products if they don't exist locally
    if not prodfile.exists():
        print("Downloading test verification data.")
        url = "http://bao.chimenet.ca/testcache/drift_testproducts.tar.gz"
        urlretrieve(url, prodfile)

    with tarfile.open(prodfile, "r:gz") as tf:
        tf.extractall(path=_base)

    def _load(fname):
        path = _base / fname

        if not path.exists():
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

    assert mfile.samefile(tfile)  # Manager does not see same directory


# Add padding to start of the btm dataset to account for the compact storage
def _pad_btm_m(fh):
    bm_saved = fh["beam_m"][:]
    m = int(fh.attrs["m"])
    final_pad = [(0, 0)] * (bm_saved.ndim - 1) + [(m, 0)]
    return np.pad(bm_saved, final_pad, mode="constant", constant_values=0)


# This works despite the non-determinism because the elements are small.
def test_beam_m(manager, saved_products):
    """Check the consistency of the m-ordered beams."""

    # Load cached beam transfer and insert the zeros that are missing from the beginning
    # of the l axis
    with saved_products("beam_m_14.hdf5") as f:
        bm_saved = _pad_btm_m(f)
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
