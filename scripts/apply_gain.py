"""Apply gain fluctuations.

Usage:
	apply_gain.py <mdir> <tsdir> <sigma_abs> <sigma_phase>
"""

import numpy as np
import h5py

from caput import mpiutil

from drift.core import manager
from drift.pipeline import timestream

from docopt import docopt


args = docopt(__doc__)


sigma_g_abs = float(args["<sigma_abs>"])
sigma_g_phase = float(args["<sigma_phase>"])

m = manager.ProductManager.from_config(args["<mdir>"])
ts = timestream.Timestream(args["<tsdir>"], args["<mdir>"])

t = m.telescope


def mk_gain_fluctuation(sigma_abs, sigma_phase):

    g_phase = np.random.standard_normal(m.telescope.nfeed) * 2 * np.pi * sigma_phase
    g_abs = np.random.standard_normal(m.telescope.nfeed) * sigma_abs

    g = (1.0 + g_abs) * np.exp(1.0j * g_phase)

    gmat = np.outer(g, g.conj())

    wmask = np.where(np.logical_and(t.feedmask, np.logical_not(t.feedconj)))

    bf0 = np.bincount(t.feedmap[wmask])
    bf1r = np.bincount(t.feedmap[wmask], weights=gmat[wmask].real)
    bf1i = np.bincount(t.feedmap[wmask], weights=gmat[wmask].imag)

    ug = (bf1r + 1.0j * bf1i) / bf0

    return ug


# Distribute over frequencies to apply gain fluctuations.
for fi in mpiutil.mpirange(t.nfreq):

    tsf = h5py.File(ts._ffile(fi))

    # Generate gain fluctuations for this frequency
    gain_fluc = mk_gain_fluctuation(sigma_g_abs, sigma_g_phase)

    # Apply to each visibility
    for ti in range(ts.ntime):
        tsf["timestream"][:, ti] *= gain_fluc

    tsf.close()
