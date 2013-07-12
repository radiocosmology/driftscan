

import os.path
import shutil
import warnings

import yaml

from drift.util import mpiutil

from drift.telescope import cylinder, gmrt, focalplane, restrictedcylinder, exotic_cylinder
from drift.core import beamtransfer

from drift.core import kltransform, doublekl
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
                }



## Power spectrum estimation configuration
pstype_dict =   {   'Full'          : psestimation.PSExact,
                    'MonteCarlo'    : psmc.PSMonteCarlo,
                    'MonteCarloAlt'    : psmc.PSMonteCarloAlt
                }



def _resolve_class(clstype, clsdict, objtype=''):
    # If clstype is a dict, try and resolve the class from `module` and
    # `class` properties. If it's a string try and resolve the class from
    # either its name and a lookup dictionary.

    if isinstance(clstype, dict):
        # Lookup custom type

        modname = clstype['module']
        clsname = clstype['class']

        if 'file' in clstype:
            import imp
            module = imp.load_source(modname, clstype['file'])
        else:
            module = __import__(modname)
        cls_ref = module.__dict__[clsname]


    elif clstype in clsdict:
        cls_ref = clsdict[clstype]
    else:
        raise Exception("Unsupported %s" % objtype)

    return cls_ref



class ProductManager(object):

    directory = None

    gen_beams = False
    gen_kl = False
    gen_ps = False
    gen_proj = False


    @classmethod
    def from_config(cls, configfile):

        configfile = os.path.normpath(os.path.expandvars(os.path.expanduser(configfile)))

        if not os.path.exists(configfile):
            raise Exception("Configuration file does not exist.")

        if os.path.isdir(configfile):
            configfile = configfile + '/config.yaml'

        # Read in config file to fetch output directory
        with open(configfile) as f:
            yconf = yaml.safe_load(f)

        outdir = yconf['config']['output_directory']


        if mpiutil.rank0:

            # Create directory if required
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # Path of directory-local config.yaml file
            dfile = os.path.join(outdir, 'config.yaml')

            # Rewrite config file to make output path absolute (and put in <outdir>/config.yaml)
            if not os.path.exists(dfile) or not os.path.samefile(configfile, dfile):

                outdir_orig = outdir
                # Work out absolute path
                if not os.path.isabs(outdir):
                    outdir = os.path.normpath(os.path.join(os.path.dirname(configfile), outdir))

                with open(configfile, 'r') as f:
                    config_contents = f.read()

                # Rewrite path in config file if not absolute
                if outdir_orig != outdir:
                    config_contents = config_contents.replace(outdir_orig, outdir)

                # Write config file into local copy
                with open(dfile, 'w+') as f:
                    f.write(config_contents)


        # Load config into a new class and return
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


        telclass = _resolve_class(teltype, teltype_dict, 'telescope')

        self.telescope = telclass.from_config(yconf['telescope'])


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

        ## Set the singular value cut for the *polarisation* beamtransfers
        if 'polsvcut' in yconf['config']:
            self.beamtransfer.polsvcut = float(yconf['config']['polsvcut'])



        if yconf['config']['beamtransfers']:
            self.gen_beams = True


        self.kltransforms  = {}

        if 'kltransform' in yconf:

            for klentry in yconf['kltransform']:
                kltype = klentry['type']
                klname = klentry['name']

                klclass = _resolve_class(kltype, kltype_dict, 'KL filter')

                kl = klclass.from_config(klentry, self.beamtransfer, subdir=klname)
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
                
                psclass = _resolve_class(pstype, pstype_dict, 'PS estimator')

                if klname not in self.kltransforms:
                    warnings.warn('Desired KL object (name: %s) does not exist.' % klname)
                    self.psestimators[psname] = None
                else:
                    self.psestimators[psname] = psclass.from_config(psentry, self.kltransforms[klname], subdir=psname)




    def generate(self):


        if self.gen_beams:
            self.beamtransfer.generate()


        if self.gen_kl:

            for klname, klobj in self.kltransforms.items():
                klobj.generate()

        if self.gen_ps:

            for psname, psobj in self.psestimators.items():
                psobj.generate()
                psobj.delbands()

        if mpiutil.rank0:
            print "========================================"
            print "=                                      ="
            print "=           DONE AT LAST!!             ="
            print "=                                      ="
            print "========================================"




