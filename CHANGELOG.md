# [20.5.0](https://github.com/radiocosmology/driftscan/compare/v20.2.0...v20.5.0) (2020-05-06)


### Bug Fixes

* **version:** remove version prefix v ([28cb07f](https://github.com/radiocosmology/driftscan/commit/28cb07fac0ad5ba21a37a7da15f1de6c0ff97fdc))



# [20.2.0](https://github.com/radiocosmology/driftscan/compare/v1.2.0...v20.2.0) (2020-02-17)

Note: we have switched to calendar versioning for this release.


### Bug Fixes

* **makeproducts:** add call to cli() at end of script needed for batch jobs ([5baa761](https://github.com/radiocosmology/driftscan/commit/5baa7613c27b682904f0d06cf221ec8bf485a279)), closes [#85](https://github.com/radiocosmology/driftscan/issues/85)
* **telescope:** incorrect name in freq calculation code when binning ([746c345](https://github.com/radiocosmology/driftscan/commit/746c34566aa9bb9b342407acdc570d1ca75bfc30))
* catch pinv linalg error ([ac74fff](https://github.com/radiocosmology/driftscan/commit/ac74fff1daf26b4afe46e0e23345a346d5ad5b79))
* disable MPI during tests on macOS ([84d2c5e](https://github.com/radiocosmology/driftscan/commit/84d2c5ea0ce77f2a0a0902baff5e406b786eef20))
* rename polarization property to polarisation ([#83](https://github.com/radiocosmology/driftscan/issues/83)) ([cb32033](https://github.com/radiocosmology/driftscan/commit/cb320339770653f79769bf96d6f3735772fdef83))
* zenith issue in DishArray example ([73bf3ab](https://github.com/radiocosmology/driftscan/commit/73bf3ab63cae930759967db92d8a147f678c135a))


### Features

* added versioneer for generating better version numbers ([2a7328d](https://github.com/radiocosmology/driftscan/commit/2a7328d6848627e8f0ec601d34992b9c78da95c0))
* **beamtransfer:** option to skip inverting svd beamtranfers ([0d2ac02](https://github.com/radiocosmology/driftscan/commit/0d2ac023396af27ff5808f4c1c2225efd2aeba55))
* **SimplePolarizedTelescope:** Add polarization property ([02bd22f](https://github.com/radiocosmology/driftscan/commit/02bd22f56ffacc8d63d0ba8a7d01020ceaefa612))
* **telescope:** add a default `input_index` implementation ([09ace31](https://github.com/radiocosmology/driftscan/commit/09ace3178575cb1b31d08e0c57e79f59f16b380c))
* **telescope:** change frequency specification to match cora ([cddd69d](https://github.com/radiocosmology/driftscan/commit/cddd69d729944c58a58db46f7eda084512619cc3))
* add PertCylinder to telescope dictionary in manager ([f0eac2b](https://github.com/radiocosmology/driftscan/commit/f0eac2b5dfdf263a16847f2150b9225eac84d483))
* Python 3 support ([e8596ae](https://github.com/radiocosmology/driftscan/commit/e8596aed11b3bd43a0ff9a65ad851aee261bf307))


### Performance Improvements

* optimise calculation of uniquepairs ([d2a7921](https://github.com/radiocosmology/driftscan/commit/d2a7921a17bdd2c357acc1a1047722e6bcc175fa)), closes [#78](https://github.com/radiocosmology/driftscan/issues/78)


### Reverts

* Revert "No account by default." ([beec9d5](https://github.com/radiocosmology/driftscan/commit/beec9d556af38627332f9dc16fe97d6efd6dcdb2))



# [1.2.0] (2017-06-24)

### Bug Fixes

* Fixed a casting bug in the quadratic estimator code that caused a crash with
  new numpy versions.


### Features

* A projection routine from the SVD basis back into the telescope basis.


# [1.1.0] (2016-08-12)

### Bug Fixes

* Fixed issue in calculation of baseline equivalence


### Features

* A local_origin property to `TransitTelescope` that determines whether the
  azimuthal coordinate system is centred at the observer. This determines if the
  rotation angle is *Local Stellar Angle* or *Earth Rotation Angle*.
* This `CHANGELOG` file.
* Finer control over the properties that saved when `TransitTelescope` is pickled.


### Changed

* Changed TransitTelescope to be a subclass of `caput.time.Observer`. This means
  that the location of the telescope is now set via latitude and longitude
  properties, not the zenith property (which is now read only). This could cause
  a small breakage if you were setting zenith through a YAML file.
* Updated dependencies in `setup.py`
* Moved utility modules `mpiutil` and `config` into `caput`.

