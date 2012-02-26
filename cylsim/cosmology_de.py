import numpy as np

from utils import cosmology

from utils.units import mega_parsec


class CosmologyDE(cosmology.Cosmology):

    w_0 = -1.0
    w_a =  0.0

    def H(self, z=0.0):

        H  = self.H0 * (self.omega_r * (1 + z)**4 +  self.omega_m * (1 + z)**3
                        + self.omega_k * (1 + z)**2
                        + self.omega_l * (1 + z)**(3*(1+self.w_0 + self.w_a)) * np.exp(-3*self.w_a * z / (1 + z)))**0.5
        
        # Convert to SI
        return H * 1000.0 / mega_parsec
        
