import numpy as np

from cylsim import cylinder, util


def gaussian_fwhm(x, fwhm):

    sigma = fwhm / (8.0*np.log(2.0))**0.5
    x2 = x**2 / (2*sigma**2)
    
    return np.exp(-x2)



class RestrictedBeam(cylinder.CylinderTelescope):

    beam_height = 30.0
    beam_type = 'box'

    __config_table_ =   {
                          'beam_height'   : [float, 'beam_height'],
                          'beam_type'     : [str,   'beam_type']
                        }


    def __init__(self, *args, **kwargs):
        super(RestrictedBeam, self).__init__(*args, **kwargs)

        self.add_config(self.__config_table_)


    def bmask_gaussian(self, feed, freq):

        pointing = self.zenith
        bdist = (self._angpos - pointing[np.newaxis, :])
        bdist = np.abs(np.where((bdist[:, 1] < np.pi)[:, np.newaxis], bdist, bdist - np.array([0, 2*np.pi])[np.newaxis, :]))

        bmask =  gaussian_fwhm(bdist[:, 0], np.radians(self.beam_height))

        return bmask


    def bmask_box(self, feed, freq):

        pointing = self.zenith
        bdist = (self._angpos - pointing[np.newaxis, :])
        bdist = np.abs(np.where((bdist[:, 1] < np.pi)[:, np.newaxis], bdist, bdist - np.array([0, 2*np.pi])[np.newaxis, :]))
        bmask =  (np.abs(bdist[:, 0] / np.radians(self.beam_height)) < 0.5)

        return bmask





class RestrictedCylinder(RestrictedBeam, cylinder.UnpolarisedCylinderTelescope):


    def beam(self, *args, **kwargs):
        bdict = {
                  'gaussian' : self.bmask_gaussian,
                  'box'      : self.bmask_box
                }

        return bdict[self.beam_type](*args, **kwargs) * cylinder.UnpolarisedCylinderTelescope.beam(self, *args, **kwargs)




class RestrictedPolarisedCylinder(RestrictedBeam, cylinder.PolarisedCylinderTelescope):


    def beamx(self, *args, **kwargs):
        bdict = {
                  'gaussian' : self.bmask_gaussian,
                  'box'      : self.bmask_box
                }

        return bdict[self.beam_type](*args, **kwargs)[:, np.newaxis] * cylinder.PolarisedCylinderTelescope.beamx(self, *args, **kwargs)


    def beamy(self, *args, **kwargs):
        bdict = {
                  'gaussian' : self.bmask_gaussian,
                  'box'      : self.bmask_box
                }

        return bdict[self.beam_type](*args, **kwargs)[:, np.newaxis] * cylinder.PolarisedCylinderTelescope.beamy(self, *args, **kwargs)

