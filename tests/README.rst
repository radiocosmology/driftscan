===============
Driftscan Tests
===============

These tests check that the code will run (in MPI mode) to generate the
various analysis products, and then test that they outputs are consistent
with a known good version (saved as a Git LFS blob in the repo).

To run the tests you'll need to have `pytest` and `git-lfs` installed. The
tests themselves can be run by simply calling ::

    $ pytest

If you want access to the output products for debugging you can set the
environment variable `DRIFT_TESTDIR` and all the products will be generated
within that directory.
