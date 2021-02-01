import numpy as np
import scipy.linalg


def svd_dm(matrix, full_matrices=True):
    """Perform the SVD of a block diagonal matrix.

    Parameters
    ----------
    matrix : (nblocks, n, m) np.ndarray
        A array containing `nblocks` diagonal blocks of size (`n`, `m`).
    full_matrices : boolean
        Whether to return the full size SVD matrices, or truncate. See
        documentation for `scipy.linalg.svd`

    Returns
    -------
    u : (nblocks, n, k1) np.ndarray
        The left eigenvectors in block form. `k1` is `n` if `full_matrices` is
        set, otherwise ``k1 = min(n, m)``
    sig : (nblocks, k) np.ndarray
        The singular values in block form, ``k = min(n, m)``
    v : (nblocks, k2, m) np.ndarray
        The right eigenvectors in block form. `k2` is `m` if `full_matrices` is
        set, otherwise ``k2 = min(n, m)``
    """
    nblocks, n, m = matrix.shape
    dt = matrix.dtype
    k = min(n, m)

    sig = np.zeros((nblocks, k), dtype=dt)

    if full_matrices:
        u = np.zeros((nblocks, n, n), dtype=dt)
        v = np.zeros((nblocks, m, m), dtype=dt)
    else:
        u = np.zeros((nblocks, n, k), dtype=dt)
        v = np.zeros((nblocks, k, m), dtype=dt)

    for ib in range(nblocks):
        u[ib], sig[ib], v[ib] = scipy.linalg.svd(
            matrix[ib], full_matrices=full_matrices
        )

    return u, sig, v


def multiply_dm_v(matrix, vector, conj=False):
    """Multiply a block diagonal matrix by a blocked vector.

    Parameters
    ----------
    matrix : (nblocks, n, m) np.ndarray
        An array containing `nblocks` diagonal blocks of size (`n`, `m`).
    vector : (nblocks, m) np.ndarray
        An array containing the blocks of the vector, each of length `m`.
    conj : boolean, optional
        Whether to multiply by the Hermitian conjugate of the matrix.

    Returns
    -------
    newvector : (nblocks, n) np.ndarray
         An array containing the blocks of the vector, each of length `n`.
    """
    if conj:
        matrix = np.swapaxes(matrix, -1, -2).conj()

    nblocks, n, m = matrix.shape

    if vector.shape != (nblocks, m):
        raise Exception("Shapes not compatible.")

    # Check dtype
    dt = np.dot(matrix[0], vector[0]).dtype

    nvector = np.empty((nblocks, n), dtype=dt)

    for i in range(nblocks):
        nvector[i] = np.dot(matrix[i], vector[i])

    return nvector


def multiply_dm_dm(matrix1, matrix2):
    """Multiply a block diagonal matrix by another diagonal matrix.

    Parameters
    ----------
    matrix1 : (nblocks, n, m) np.ndarray
        An array containing `nblocks` diagonal blocks of size (`n`, `m`).
    matrix2 : (nblocks, m, k) np.ndarray
        An array containing `nblocks` diagonal blocks of size (`m`, `k`).

    Returns
    -------
    nmatrix : (nblocks, n, k) np.ndarray
         An array containing `nblocks` diagonal blocks of size (`n`, `k`).
    """

    nblocks, n, m = matrix1.shape
    k = matrix2.shape[2]

    if matrix2.shape[:2] != (nblocks, m):
        raise Exception("Shapes not compatible.")

    # Check dtype
    dt = np.promote_types(matrix1.dtype, matrix2.dtype)

    nmatrix = np.empty((nblocks, n, k), dtype=dt)

    for i in range(nblocks):
        nmatrix[i] = np.dot(matrix1[i], matrix2[i])

    return nmatrix


def pinv_dm(matrix, *args, **kwargs):
    """Construct the pseudo-inverse of a block diagonal matrix.

    Parameters
    ----------
    matrix : (nblocks, n, m) np.ndarray
        An array containing `nblocks` diagonal blocks of size (`n`, `m`).

    Returns
    -------
    pinv_matrix : (nblocks, m, n) np.ndarray
         An array containing the pseudo-inverse.
    """

    nblocks, n, m = matrix.shape

    pinv_matrix = np.empty((nblocks, m, n), dtype=matrix.dtype)

    for i in range(nblocks):
        pinv_matrix[i] = scipy.linalg.pinv(matrix[i], *args, **kwargs)

    return pinv_matrix
