
import numpy as np
from scipy.special import jn

from cosmoutils import coord, units
from cylsim import telescope, util


def jinc(x):
    return 0.5 * (jn(0, x) + jn(2, x))


def beam_circular(angpos, zenith, uv_diameter):
    """Beam pattern for a circular dish.
    
    Parameters
    ----------
    angpos : np.ndarray
        Array of angular positions
    zenith : np.ndarray
        Co-ordinates of the zenith.
    uv_diameter : scalar
        Diameter of the dish (in units of wavelength).
    
    Returns
    -------
    beam : np.ndarray
        Beam pattern at each position in angpos.
    """
    
    x = (1.0 - coord.sph_dot(angpos, zenith)**2)**0.5 * np.pi * uv_diameter
    
    return 2*jinc(x)


def gaussian_beam(angpos, pointing, fwhm):

    sigma = np.radians(fwhm) / (8.0*np.log(2.0))**0.5
    x2 = (1.0 - coord.sph_dot(angpos, pointing)**2) / (4*sigma**2)
    
    return np.exp(-x2)


class FocalPlaneArray(telescope.UnpolarisedTelescope):


    beam_num_u = 10
    beam_num_v = 10


    beam_spacing_u = 0.1
    beam_spacing_v = 0.1

    beam_size = 0.1
    beam_pivot = 400.0

    __config_table_ = { 'beam_num_u'        : [int,     'beam_num_u'],
                        'beam_num_v'        : [int,     'beam_num_v'],
                        'beam_spacing_u'    : [float,   'beam_spacing_u'],
                        'beam_spacing_v'    : [float,   'beam_spacing_v'],
                        'beam_size'         : [float,   'beam_size'],
                        'beam_pivot'        : [float,   'beam_pivot']
                      }

    def __init__(self, *args, **kwargs):
        """Initialise a telescope object.
        """

        super(FocalPlaneArray, self).__init__(*args, **kwargs)

        self.add_config(self.__config_table_)

    @property
    def beam_pointings(self):
        pnt_u = self.beam_spacing_u * (np.arange(self.beam_num_u) - (self.beam_num_u - 1) / 2.0)
        pnt_v = self.beam_spacing_v * (np.arange(self.beam_num_v) - (self.beam_num_v - 1) / 2.0)

        pnt_u = np.radians(pnt_u) + self.zenith[1]
        pnt_v = np.radians(pnt_v) + self.zenith[0]

        pnt = np.zeros((self.beam_num_u, self.beam_num_v, 2))
        pnt[:, :, 1] = pnt_u[:, np.newaxis]
        pnt[:, :, 0] = pnt_v[np.newaxis, :]

        return pnt.reshape(-1, 2)

    #== Methods for calculating the unique baselines ===
    

    @util.cache_last
    def beam(self, feed, freq):

        pointing = self.beam_pointings[feed]
        fwhm = self.beam_size * self.frequencies[freq] / self.beam_pivot

        return gaussian_beam(self._angpos, pointing, fwhm)
        

    @property
    def dish_width(self):
        lpivot = (units.c / self.beam_pivot * 1e-6)
        return (lpivot / np.radians(self.beam_size))
    
    @property
    def u_width(self):
        return self.dish_width

    @property
    def v_width(self):
        return self.dish_width

    @property
    def nfeed(self):
        return self.beam_num_u * self.beam_num_v

    @property
    def feedpositions(self):
        """Feed positions (all zero in FPA).
        """
        return np.zeros([self.nfeed, 2])

    def _unique_beams(self):

        beam_mask = np.identity(self.nfeed, dtype=np.bool)
        beam_map = telescope._remap_keyarray(np.diag(np.arange(self.nfeed)), mask=beam_mask)

        return beam_map, beam_mask




