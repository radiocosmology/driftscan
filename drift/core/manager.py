

import os.path
import shutil
import warnings

import yaml

from drift.util import mpiutil

from drift.telescope import cylinder, gmrt, focalplane, restrictedcylinder, exotic_cylinder
from drift.core import beamtransfer

from drift.core import kltransform, doublekl, wiener
from drift.core import psestimation, psmc
from drift.core import skymodel



teltype_dict =  {   'UnpolarisedCylinder'   : cylinder.UnpolarisedCylinderTelescope,
                    'PolarisedCylinder'     : cylinder.PolarisedCylinderTelescope,
                    'GMRT'                  : gmrt.GmrtUnpolarised,
                    'FocalPlane'            : focalplane.FocalPlaneArray,
                    'RestrictedCylinder'    : restrictedcylinder.RestrictedCylinder,
                    'RestrictedPolarisedCylinder'    : restrictedcylinder.RestrictedPolarisedCylinder,
                    'RestrictedExtra'       : restrictedcylinder.RestrictedExtra,                  
                    'GradientCylinder'       : exotic_cylinder.GradientCylinder
                }


## KLTransform configuration
kltype_dict =   {   'KLTransform'   : kltransform.KLTransform,
                    'DoubleKL'      : doublekl.DoubleKL,
                    'Wiener'        : wiener.Wiener
                }



## Power spectrum estimation configuration
pstype_dict =   {   'Full'          : psestimation.PSEstimation,
                    'MonteCarlo'    : psmc.PSMonteCarlo,
                    'MonteCarloAlt'    : psmc.PSMonteCarloAlt
                }

class ProductManager(object):

    directory = None

    gen_beams = False
    gen_kl = False
    gen_ps = False
    gen_proj = False


    @classmethod
    def from_config(cls, configfile):
        c = cls()
        c.load_config(configfile)

        return c


    def load_config(self, configfile):

        with open(configfile) as f:
            yconf = yaml.safe_load(f)

        ## Global configuration
        ## Create output directory and copy over params file.
        if 'config' not in yconf:
            raise Exception('Configuration file must have an \'config\' section.')

        self.directory = yconf['config']['output_directory']

        if not os.path.isabs(self.directory):
            self.directory = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(configfile)), self.directory))

        if mpiutil.rank0:
            print "Product directory:", self.directory

        ## Telescope configuration
        if 'telescope' not in yconf:
            raise Exception('Configuration file must have an \'telescope\' section.')

        teltype = yconf['telescope']['type']


        if teltype not in teltype_dict:
            raise Exception("Unsupported telescope type.")

        self.telescope = teltype_dict[teltype].from_config(yconf['telescope'])


        if 'reionisation' in yconf['config']:
            if yconf['config']['reionisation']:
                skymodel._reionisation = True

        ## Beam transfer generation
        if 'nosvd' in yconf['config'] and yconf['config']['nosvd']:
            self.beamtransfer = beamtransfer.BeamTransferNoSVD(self.directory + '/bt/', telescope=self.telescope)    
        else:
            self.beamtransfer = beamtransfer.BeamTransfer(self.directory + '/bt/', telescope=self.telescope)

        ## Set the singular value cut for the beamtransfers
        if 'svcut' in yconf['config']:
            self.beamtransfer.svcut = float(yconf['config']['svcut'])

        if yconf['config']['beamtransfers']:
            self.gen_beams = True


        self.kltransforms  = {}

        if 'kltransform' in yconf:

            for klentry in yconf['kltransform']:
                kltype = klentry['type']
                klname = klentry['name']

                if kltype not in kltype_dict:
                    raise Exception("Unsupported transform.")

                kl = kltype_dict[kltype].from_config(klentry, self.beamtransfer, subdir=klname)
                self.kltransforms[klname] = kl

        if yconf['config']['kltransform']:
            self.gen_kl = True



        self.psestimators = {}

        if yconf['config']['psfisher']:
            self.gen_ps = True

            if 'psfisher' not in yconf:
                raise Exception('Require a psfisher section if config: psfisher is Yes.')

        if 'psfisher' in yconf:
            for psentry in yconf['psfisher']:
                pstype = psentry['type']
                klname = psentry['klname']
                psname = psentry['name'] if 'name' in psentry else 'ps'
                
                if pstype not in pstype_dict:
                    raise Exception("Unsupported PS estimation.")

                if klname not in self.kltransforms:
                    warnings.warn('Desired KL object (name: %s) does not exist.' % klname)
                    self.psestimators[psname] = None
                else:
                    self.psestimators[psname] = pstype_dict[pstype].from_config(psentry, self.kltransforms[klname], subdir=psname)



        # ## Projections code
        # if yconf['config']['projections']:
        #     if 'projections' not in yconf:
        #         raise Exception('Require a projections section if config: projections is Yes.')

        #     for projentry in yconf['projections']:
        #         klname = projentry['klname']

        #         # Override default and ensure we copy the original maps
        #         if 'copy_orig' not in projentry:
        #             projentry['copy_orig'] = True

        #         for mentry in projentry['maps']:
        #             if 'stem' not in mentry:
        #                 raise Exception('No stem in mapentry %s' % mentry['file'])

        #             mentry['stem'] = self.directory + '/projections/' + klname + '/' + mentry['stem'] + '/'
                
        #         if klname not in self.kltransforms:
        #             raise Exception('Desired KL object does not exist.')

        #         proj = projection.Projector.from_config(projentry, self.kltransforms[klname])
        #         proj.generate()



    def generate(self):


        # Create directory if required
        if mpiutil.rank0:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            # Copy config file into output directory (check it's not already there first)
            sfile = os.path.realpath(os.path.abspath(configfile))
            dfile = os.path.realpath(os.path.abspath(self.directory + '/config.yaml'))

            if sfile != dfile:
                shutil.copy(sfile, dfile)

        if self.gen_beams:
            self.beamtransfer.generate()


        if self.gen_kl:

            for klname, klobj in self.kltransforms.items():
                klobj.generate()

        if self.gen_ps:

            for psname, psobj in self.psestimators.items():
                psobj.generate()


        if mpiutil.rank0:
            print "========================================"
            print "=                                      ="
            print "=           DONE AT LAST!!             ="
            print "=                                      ="
            print "========================================"




