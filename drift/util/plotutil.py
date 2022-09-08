import numpy as np


def regrid_polar(polar_img, r_bins, theta_bins, res=1024):
    """Regrid an (r, theta) quarter plane onto (rx, ry).

    Useful for breaking out (k, theta) Fisher errors into (kpar, kperp).

    Parameters
    ----------
    polar_img : np.ndarray[num_r, num_theta]
        The values on the polar grid.
    r_bins : np.ndarray[num_r + 1]
        The bin boundaries in the r direction.
    theta_bins : np.ndarray[num_theta + 1]
        The bin boundaries in the theta direction.
    res : integer, optional (default=1024)
        The number of pixels on each side of the cartesian grid.

    Returns
    -------
    cart_img : np.ndarray[res, res]
        The regridded image.
    """
    ra = np.linspace(r_bins[0], r_bins[-1], res, endpoint=True)

    rpar = ra[:, np.newaxis]
    rperp = ra[np.newaxis, :]

    r = (rpar**2 + rperp**2) ** 0.5
    th = np.arccos(rpar / r)
    th[0, 0] = 0.0

    rbin = (np.digitize(r.flatten(), r_bins) - 1).reshape(r.shape)
    tbin = (np.digitize(th.flatten(), theta_bins) - 1).reshape(th.shape)

    ia = np.where(
        np.logical_and(tbin < (len(theta_bins) - 1), rbin < (len(r_bins) - 1))
    )

    cart_img = np.zeros((res, res), dtype=polar_img.dtype)
    cart_img[:] = np.nan

    cart_img[ia] = polar_img[rbin[ia], tbin[ia]]

    return cart_img
