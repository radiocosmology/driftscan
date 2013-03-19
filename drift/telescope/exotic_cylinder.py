
import numpy as np

from drift.telescope import cylinder


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

    min_spacing = None
    max_spacing = 20.0

    __config_table_ = { 'min_spacing'   : [float, 'min_spacing'],
                        'max_spacing'   : [float, 'max_spacing'] }


    def __init__(self, *args, **kwargs):
        super(cylinder.UnpolarisedCylinderTelescope, self).__init__(*args, **kwargs)

        self.add_config(self.__config_table_)

    def feed_positions_cylinder(self, cylinder_index):

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        nf = self.num_feeds
        sp = self.feed_spacing

        # Parameters for gradient feedspacing
        a = self.wavelengths[-1] / 2.0 if self.min_spacing is None else self.min_spacing
        #b = 2 * (sp - a) / nf
        b = 2.0*(self.max_spacing - a * (nf-1)) / (nf-1)**2.0
        
        pos = np.empty([nf, 2], dtype=np.float64)

        i = np.arange(nf)

        pos[:, 0] = cylinder_index * self.cylinder_spacing
        pos[:, 1] = a*i + 0.5 * b*i**2

        return pos

