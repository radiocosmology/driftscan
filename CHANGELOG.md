# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [0.2.0] - 2016-08-12

### Added

- A local_origin property to TransitTelescope that determines whether the
  azimuthal coordinate system is centred at the observer. This determines if the
  rotation angle is Local Stellar Angle or Earth Rotation Angle.
- A CHANGELOG file.

### Changed

- Changed TransitTelescope to be a sublass of caput.time.Observer. This means
  that the location of the telescope is now set via latitude and longitude
  properties, not the zenith property (which is now read only).
- Updated dependencies in setup.py
