from drift.util import blockla

import numpy as np

import scipy.linalg as la


def test_blocksvd():
    a1 = np.zeros((4, 6))
    b1 = np.random.standard_normal((2, 2, 3))

    a1[:2, :3] = b1[0]
    a1[2:4, 3:6] = b1[1]

    ua, sa, va = la.svd(a1)
    ub, sb, vb = blockla.svd_dm(b1, full_matrices=True)

    sas = np.sort(sa.flat)
    sbs = np.sort(sb.flat)

    assert np.allclose(sas, sbs)

    assert np.allclose(np.dot(ub[0, :, 0], ub[0, :, 1]), 0.0)
    assert np.allclose(np.dot(ub[1, :, 0], ub[1, :, 1]), 0.0)

    assert np.allclose(np.dot(vb[0, :, 0], vb[0, :, 2]), 0.0)
