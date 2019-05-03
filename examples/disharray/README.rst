==================
Dish Array Example
==================

This code example contains a custom telescope class that simulates an
interferometric grid of dishes, it then uses this to construct a synthetic
timestream, and then it forms the m-modes to make a map out of them.

The custom telescope is contained in the file ``simplearray.py``.

The example can be run in two ways, first using the python program in
``driver.py`` which can be run by simply doing::

	$ python driver.py

or to speed things up, you can run::

	$ export OMP_NUM_THREADS=<DESIRED THREAD NUM>
	$ mpirun -np N python driver.py

where the ``OMP_NUM_THREADS`` is only useful if you are using threaded
numerical libraries (libsharp for SHT, MKL for linear algebra).


The second way to run the code is to use the `YAML` configuration file
interface. Simply run::

	$ drift-makeproducts run prod_params.yaml
	$ drift-runpipeline run pipe_params.yaml


In both cases you'll need a sky map to simulate. To get a quick map of the
galaxy, try running::

	$ cora-makesky --filename=simulated_map.hdf5 --nside=256 --freq 100.0 150.0 5 --pol=zero galaxy

