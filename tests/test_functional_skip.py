from platform import python_version

import numpy as np
import pytest

from test_functional import (
    approx,
    test_dir,
    saved_products,
    _gen_prod,
    _basedir,
    _pad_btm_m,
)


@pytest.fixture(scope="module")
def products_run(test_dir):
    # Generate separate test product that skip entries in the beam transfers
    return _gen_prod(
        test_dir / f"skipprod_python_{python_version()}",
        _basedir / "testskipparams.yaml",
    )


@pytest.fixture()
def manager(products_run):
    return products_run[2]


def test_skip_beam_m(manager, saved_products):
    """Check the consistency of the m-ordered beams."""

    # Load cached beam transfer and insert the zeros that are missing from the beginning
    # of the l axis
    with saved_products("beam_m_14.hdf5") as f:
        bm_saved = _pad_btm_m(f)
    bm = manager.beamtransfer.beam_m(14)

    # Check the shapes are consistent
    assert bm_saved.shape == bm.shape

    # Test that the beam has actually changed significantly
    assert not (bm == approx(bm_saved))

    # Apply the masking from the skipped entries to the saved data
    skipped_freq = manager.config["telescope"]["skip_freq"]
    bm_saved[skipped_freq] = 0.0
    skipped_baselines = manager.config["telescope"]["skip_baselines"]
    bm_saved[:, :, skipped_baselines] = 0.0
    if manager.config["telescope"]["skip_pol"]:
        bm_saved[:, :, :, 1:] = 0.0

    # Test that the entries expected to be missing are indeed missing
    assert bm == approx(bm_saved)


def test_skip_beam_m_iter(manager, saved_products):
    """Check the consistency of the m-ordered beams."""

    with saved_products("beam_m_14.hdf5") as f:
        bm_saved = _pad_btm_m(f)

    # Apply the masking from the skipped entries to the saved data
    skipped_freq = manager.config["telescope"]["skip_freq"]
    bm_saved[skipped_freq] = 0.0
    skipped_baselines = manager.config["telescope"]["skip_baselines"]
    bm_saved[:, :, skipped_baselines] = 0.0
    if manager.config["telescope"]["skip_pol"]:
        bm_saved[:, :, :, 1:] = 0.0

    for fi in range(manager.telescope.nfreq):
        bm = manager.beamtransfer.beam_m(14, fi=fi)

        # Check the shapes are consistent
        assert bm_saved[fi].shape == bm.shape

        # Test that the entries expected to be missing are indeed missing
        assert bm == approx(bm_saved[fi])


def test_project_s2t(manager, saved_products):
    """Check the consistency of the m-ordered beams."""

    with saved_products("beam_m_14.hdf5") as f:
        bm_saved = _pad_btm_m(f)

    # Apply the masking from the skipped entries to the saved data
    skipped_freq = manager.config["telescope"]["skip_freq"]
    bm_saved[skipped_freq] = 0.0
    skipped_baselines = manager.config["telescope"]["skip_baselines"]
    bm_saved[:, :, skipped_baselines] = 0.0

    if manager.config["telescope"]["skip_pol"]:
        bm_saved[:, :, :, 1:] = 0.0

    nfreq = manager.telescope.nfreq
    ntel = 2 * manager.telescope.nbase
    lmax = manager.telescope.lmax

    # Generate some test data
    test_vec = np.zeros((nfreq, 4, lmax + 1), dtype=np.complex128)
    test_vec.real.reshape(-1)[:] = np.arange(test_vec.size)
    test_vec.imag.reshape(-1)[:] = test_vec.size - np.arange(test_vec.size)

    # Calculate the project entry exactly
    bm_saved = bm_saved.reshape(nfreq, ntel, 4 * (lmax + 1))
    tel_vec_exact = np.zeros((nfreq, ntel), dtype=np.complex128)
    for fi in range(nfreq):
        tel_vec_exact[fi] = np.dot(bm_saved[fi], test_vec[fi].flatten())

    tel_vec = manager.beamtransfer.project_vector_sky_to_telescope(14, test_vec)

    assert (tel_vec != 0).any()
    assert tel_vec == approx(tel_vec_exact)
