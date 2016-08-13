# Change Log

All notable changes to this project will be documented in this file. This
project adheres to [Semantic Versioning](http://semver.org/), with the exception
that I'm using PEP440 to denote pre-releases.

## [1.1.0] - 2016-08-12

### Added

- A local_origin property to `TransitTelescope` that determines whether the
  azimuthal coordinate system is centred at the observer. This determines if the
  rotation angle is *Local Stellar Angle* or *Earth Rotation Angle*.
- This `CHANGELOG` file.
- Finer control over the properties that saved when `TransitTelescope` is pickled.

### Changed

- Changed TransitTelescope to be a subclass of `caput.time.Observer`. This means
  that the location of the telescope is now set via latitude and longitude
  properties, not the zenith property (which is now read only). This could cause
  a small breakage if you were setting zenith through a YAML file.
- Updated dependencies in `setup.py`
- Moved utility modules `mpiutil` and `config` into `caput`.

### Fixes

- Fixed issue in calculation of baseline equivalence
