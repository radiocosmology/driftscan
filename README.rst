====================================
Cylinder Radio Telescope Simulations
====================================

This is a Python project for simulating the performance of cylinder radio
telescopes such as CHIME, but it is also suited to analysing other types of
intensity mapping experiments.


Installation
============

The primary dependency of this project is the `simulations_21cm` package. This
can be fetched from `here <http://github.com/jrs65/simulations_21cm>`_. In
addition to its dependencies we also require an installation of `h5py
<http://h5py.alfven.org/>`_ for storing results in `hdf5` files.

You'll need to place this directory in your python module search path, for it to
run the scripts correctly. This can be done by::

    > export PYTHONPATH=$PYTHONPATH:"/<path>/<to>/cylinder_simulation"

As with the `simulations_21cm` package it is very highly recommended that you
use optimized versions of `Numpy` and `Scipy`.



