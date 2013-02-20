import numpy as np
import scipy.linalg as la

import blockla


def intpattern(n):
    """Pattern that prints out a number upto `n` (integer - always shows sign)."""
    return ("%+0" + repr(int(np.ceil(np.log10(n + 1))) + 1) + "d")


def natpattern(n):
    """Pattern that prints out a number upto `n` (natural number - no sign)."""
    return ("%0" + repr(int(np.ceil(np.log10(n + 1)))) + "d")


def cache_last(func):
    """A simple decorator to cache the result of the last call to a function.
    """
    arg_cache = [None]
    kw_cache = [None]
    ret_cache = [None]

    def decorated(*args, **kwargs):

        if args != arg_cache[0] or kwargs != kw_cache[0]:
            # Generate cache value
            ret_cache[0] = func(*args, **kwargs)
            arg_cache[0] = args
            kw_cache[0] = kwargs
        # Fetch from cache
        return ret_cache[0]

    return decorated





def proj_mblock(hpmap, vec_mblock):

    lmax = vec_mblock.shape[-1] - 1
    alm = hputil.sphtrans_complex(hpmap, lmax=lmax).T.copy()

    almproj = blockla.multiply_dm_v(vec_mblock, blockla.multiply_dm_v(vec_mblock, alm), conj=True)

    hpproj = healpy.alm2map(hputil.pack_alm(almproj.T), healpy.npix2nside(hpmap.size))

    return hpproj


class ConfigReader(object):

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        c = cls(*args, **kwargs)
        c.read_config(config)

        return c

    def add_config(self, config_options):

        if not hasattr(self, '_config_dict'):
            self._config_dict = config_options
        else:
            self._config_dict.update(config_options)


    def read_config(self, config):

        if not hasattr(self, '_config_dict'):
            return

        keys = set(config).intersection(self._config_dict)

        for key in keys:
            ctype, cname = self._config_dict[key]
            cval = config[key]

            print "Setting %s to %s" % (cname, cval)

            self.__setattr__(cname, ctype(cval))

