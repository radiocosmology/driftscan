from platform import python_version

import pytest

from test_functional import approx, test_dir, saved_products, _gen_prod, _basedir


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

    with saved_products("beam_m_14.hdf5") as f:
        bm_saved = f["beam_m"][:]
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
        bm_saved = f["beam_m"][:]

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
