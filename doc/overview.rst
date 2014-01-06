=====================
Overview of Driftscan
=====================

driftscan is a package for the analysis of data from transit radio
interferometers using the m-mode formalism which is described in 
`arXiv:1302.3267`_ and `arXiv:1401.XXXX`_.

Given a design of a telescope, this package can:

* Generate a set of products used to analyse data from it and simulate
  timestreams.
* Construct a filter which can be used to extract cosmological 21 cm emission
  from astrophysical foregrounds, such as our galaxy and radio point sources.
* Estimate the 21cm power spectrum using an optimal quadratic estimator

There are essentially two separate parts to running driftscan: generating the
analysis products, and running the pipeline. We describe how these work below.

.. _`arXiv:1302.3267`: http://arxiv.org/abs/1302.3267
.. _`arXiv:1401.XXXX`: http://arxiv.org/abs/1401.XXXX

Generating the Analysis Products
================================

The first step in running driftscan is to give a model for the telescope. This
consists of:

* A description of the primary beam of each feed. This is a two component
  vector at every at every point in the sky to describe the electric field
  response of the beam.
* The locations of each feed which are assumed to be co-planar and located at
  a specified latitude.
* A model of the instrument noise. The noise is assumed to be stationary and
  Gaussian and so is uniquely described by its power spectrum.

All of these are specified by implementing the
``driftscan.core.telescope.TransitTelescope`` class.


Now the fun can begin. The next step is to generate the Beam Transfer matrices
for each m-mode. This is conceptually straightforward:

1. Make sky maps of the polarised response for each feed pair at all observed
   frequencies.
2. Take the spherical harmonic transform of each polarised set of maps.
3. Transpose to group by the m, of each spherical harmonic transform. We must
   also conjugate the negative `m` modes to group them with the positive `m`
   modes.

In practice this can be numerically challenging due to the shear quantity of
frequencies and feed pairs present. This step is ``MPI`` parallelised and
proceeds by distributing subsets of the responses to geneate and transform
across many nodes, and performing an in memory transpose across all these
nodes to group by m. It then processes the next subset, and repeats until we
have generated the complete set.

Running the Pipeline
====================

Meh 2.