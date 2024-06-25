import functools

import numpy as np


def intpattern(n):
    """Pattern that prints out a number upto `n` (integer - always shows sign)."""
    return "%+0" + repr(int(np.ceil(np.log10(n + 1))) + 1) + "d"


def natpattern(n):
    """Pattern that prints out a number upto `n` (natural number - no sign)."""
    return "%0" + repr(int(np.ceil(np.log10(n + 1)))) + "d"


def cache_last(func):
    """A simple decorator to cache the result of the last call to a function."""
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
