# Drift Scan Telescope Analysis

This is a Python project for simulating and analysing the transit radio
telescopes, with a particular focus on 21cm Cosmology.

## Installation

The primary dependency of this project is the `cora` package. This can be
fetched from `here <http://github.com/radiocosmology/cora>`_. In addition to its
dependencies we also require an installation of `h5py
<http://h5py.alfven.org/>`_ for storing results in `hdf5` files.

This package is installable by the usual methods, either the standard ::

    $ python setup.py install [--user]

or to develop the package ::

    $ python setup.py develop [--user]

It should also be installable directly with `pip` using the command::

	$ pip install [-e] git+ssh://git@github.com/radiocosmology/driftscan


## Documentation
 The full documentation of `driftscan` is at https://radiocosmology.github.io/driftscan/.
