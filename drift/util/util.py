import functools

import numpy as np


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

    @functools.wraps(func)
    def decorated(*args, **kwargs):

        if args != arg_cache[0] or kwargs != kw_cache[0]:
            # Generate cache value
            ret_cache[0] = func(*args, **kwargs)
            arg_cache[0] = args
            kw_cache[0] = kwargs
        # Fetch from cache
        return ret_cache[0]

    return decorated


class ConfigReader(object):
    """A class for applying attribute values from a supplied dictionary.

    DEPRECATED!! Use routines from drift.util.config instead.
    """
    
    @classmethod
    def from_config(cls, config, *args, **kwargs):
        """Create an instance of the class from the supplied config dictionary.

        Parameters
        ----------
        config : dict
            Dictionary of config options.

        Returns
        -------
        obj : cls
            Instance of the class.
        """
        c = cls(*args, **kwargs)
        c.read_config(config)

        return c

    def add_config(self, config_options):
        """Add the set of configuration options to those understoof by this class.

        Parameters
        ----------
        config_options : dict
            Configuration options supplied like this: 
            { 'paramkey1' : [ function_to_apply, 'attributename'], 'paramkey2' : ...}
        """
        if not hasattr(self, '_config_dict'):
            self._config_dict = config_options
        else:
            self._config_dict.update(config_options)


    def read_config(self, config):
        """Set attributes from configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary of configuration values.
        """
        if not hasattr(self, '_config_dict'):
            return

        keys = set(config).intersection(self._config_dict)

        for key in keys:
            ctype, cname = self._config_dict[key]
            cval = config[key]

            print "Setting %s to %s" % (cname, cval)

            self.__setattr__(cname, ctype(cval))
