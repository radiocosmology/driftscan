
import numpy as np

from drift.telescope import cylinder
from drift.util import config


class RandomCylinder(cylinder.UnpolarisedCylinderTelescope):

    pos_sigma = 0.5

    def feed_positions_cylinder(self, cylinder_index):

        pos = super(RandomCylinder, self).feed_positions_cylinder(cylinder_index)

        rs = np.random.get_state()
        np.random.seed(cylinder_index)

        p1 = np.sort(pos[:, 1] + self.pos_sigma * self.feed_spacing * np.random.standard_normal(pos.shape[0]))

        np.random.set_state(rs)

        pos[:, 1] = p1
        return pos



class GradientCylinder(cylinder.UnpolarisedCylinderTelescope):

    min_spacing = config.Property(proptype=float, default=-1.0)
    max_spacing = config.Property(proptype=float, default=20.0)


    def feed_positions_cylinder(self, cylinder_index):

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        nf = self.num_feeds
        sp = self.feed_spacing

        # Parameters for gradient feedspacing
        a = self.wavelengths[-1] / 2.0 if self.min_spacing < 0.0 else self.min_spacing
        #b = 2 * (sp - a) / nf
        b = 2.0*(self.max_spacing - a * (nf-1)) / (nf-1)**2.0
        
        pos = np.empty([nf, 2], dtype=np.float64)

        i = np.arange(nf)

        pos[:, 0] = cylinder_index * self.cylinder_spacing
        pos[:, 1] = a*i + 0.5 * b*i**2

        return pos



class CylinderExtra(cylinder.UnpolarisedCylinderTelescope):

    extra_feeds = config.Property(proptype=np.array, default=[])


    def feed_positions_cylinder(self, cylinder_index):

        pos = super(CylinderExtra, self).feed_positions_cylinder(cylinder_index)


        nextra = self.extra_feeds.shape[0]

        pos2 = np.zeros((pos.shape[0] + nextra, 2), dtype=np.float64)

        pos2[nextra:] = pos

        pos2[:nextra, 0] = cylinder_index * self.cylinder_spacing
        pos2[:nextra, 1] = self.extra_feeds

        return pos2

        


