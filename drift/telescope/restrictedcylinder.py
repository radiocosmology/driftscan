import numpy as np

from caput import config

from drift.telescope import cylinder


def gaussian_fwhm(x, fwhm):
    sigma = fwhm / (8.0 * np.log(2.0)) ** 0.5
    x2 = x**2 / (2 * sigma**2)

    return np.exp(-x2)


class RestrictedBeam(cylinder.CylinderTelescope):
    beam_height = config.Property(proptype=float, default=30.0)
    beam_type = config.Property(proptype=str, default="box")

    def bmask_gaussian(self, feed, freq):
        pointing = self.zenith
        bdist = self._angpos - pointing[np.newaxis, :]
        bdist = np.abs(
            np.where(
                (bdist[:, 1] < np.pi)[:, np.newaxis],
                bdist,
                bdist - np.array([0, 2 * np.pi])[np.newaxis, :],
            )
        )

        bmask = gaussian_fwhm(bdist[:, 0], np.radians(self.beam_height))

        return bmask

    def bmask_box(self, feed, freq):
        pointing = self.zenith
        bdist = self._angpos - pointing[np.newaxis, :]
        bdist = np.abs(
            np.where(
                (bdist[:, 1] < np.pi)[:, np.newaxis],
                bdist,
                bdist - np.array([0, 2 * np.pi])[np.newaxis, :],
            )
        )
        bmask = np.abs(bdist[:, 0] / np.radians(self.beam_height)) < 0.5

        return bmask


class RestrictedCylinder(RestrictedBeam, cylinder.UnpolarisedCylinderTelescope):
    def beam(self, *args, **kwargs):
        bdict = {"gaussian": self.bmask_gaussian, "box": self.bmask_box}

        return bdict[self.beam_type](
            *args, **kwargs
        ) * cylinder.UnpolarisedCylinderTelescope.beam(self, *args, **kwargs)


class RestrictedPolarisedCylinder(RestrictedBeam, cylinder.PolarisedCylinderTelescope):
    def beamx(self, *args, **kwargs):
        bdict = {"gaussian": self.bmask_gaussian, "box": self.bmask_box}

        return bdict[self.beam_type](*args, **kwargs)[
            :, np.newaxis
        ] * cylinder.PolarisedCylinderTelescope.beamx(self, *args, **kwargs)

    def beamy(self, *args, **kwargs):
        bdict = {"gaussian": self.bmask_gaussian, "box": self.bmask_box}

        return bdict[self.beam_type](*args, **kwargs)[
            :, np.newaxis
        ] * cylinder.PolarisedCylinderTelescope.beamy(self, *args, **kwargs)


class RestrictedExtra(RestrictedCylinder):
    extra_feeds = config.Property(proptype=np.array, default=[])

    def feed_positions_cylinder(self, cylinder_index):
        pos = super(RestrictedExtra, self).feed_positions_cylinder(cylinder_index)

        nextra = self.extra_feeds.shape[0]

        pos2 = np.zeros((pos.shape[0] + nextra, 2), dtype=np.float64)

        pos2[nextra:] = pos

        pos2[:nextra, 0] = cylinder_index * self.cylinder_spacing
        pos2[:nextra, 1] = self.extra_feeds

        return pos2
