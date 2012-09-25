import argparse
import os.path
import shutil

import yaml

from cylsim import mpiutil

from cylsim import cylinder, gmrt, focalplane, restrictedcylinder
from cylsim import beamtransfer

from cylsim import kltransform, doublekl
from cylsim import psestimation, psmc
from cylsim import skymodel

from cylsim import projection

parser = argparse.ArgumentParser(description='Run a simulation job.')
parser.add_argument('configfile', type=argparse.FileType('r'), help='The configuration file to use.')
args = parser.parse_args()
yconf = yaml.safe_load(args.configfile)




## Global configuration
## Create output directory and copy over params file.
if 'config' not in yconf:
    raise Exception('Configuration file must have an \'config\' section.')

outdir = yconf['config']['output_directory']

# Create directory if required
if mpiutil.rank0 and not os.path.exists(outdir):
    os.makedirs(outdir)

# Copy config file into output directory (check it's not already there first)
sfile = os.path.realpath(os.path.abspath(args.configfile.name))
dfile = os.path.realpath(os.path.abspath(outdir + '/config.yaml'))

if sfile != dfile:
    shutil.copy(sfile, dfile)



## Telescope configuration
if 'telescope' not in yconf:
    raise Exception('Configuration file must have an \'telescope\' section.')

teltype = yconf['telescope']['type']

teltype_dict =  {   'UnpolarisedCylinder'   : cylinder.UnpolarisedCylinderTelescope,
                    'PolarisedCylinder'     : cylinder.PolarisedCylinderTelescope,
                    'GMRT'                  : gmrt.GmrtUnpolarised,
                    'FocalPlane'            : focalplane.FocalPlaneArray,
                    'RestrictedCylinder'    : restrictedcylinder.RestrictedCylinder,
                    'RestrictedPolarisedCylinder'    : restrictedcylinder.RestrictedPolarisedCylinder,
                    'RestrictedExtra'       : restrictedcylinder.RestrictedExtra                    
                }

if teltype not in teltype_dict:
    raise Exception("Unsupported telescope type.")

telescope = teltype_dict[teltype].from_config(yconf['telescope'])


if 'reionisation' in yconf['config']:
    if yconf['config']['reionisation']:
        skymodel._reionisation = True

## Beam transfer generation
if 'chunked' in yconf['config'] and yconf['config']['chunked']:
    bt = beamtransfer.BeamTransferChunked(outdir + '/bt/', telescope=telescope)    
else:
    bt = beamtransfer.BeamTransfer(outdir + '/bt/', telescope=telescope)

if yconf['config']['beamtransfers']:
    bt.generate()




## KLTransform configuration
kltype_dict =   {   'KLTransform'   : kltransform.KLTransform,
                    'DoubleKL'      : doublekl.DoubleKL,
                }
klobj_dict  = {}


if 'kltransform' in yconf:

    for klentry in yconf['kltransform']:
        kltype = klentry['type']
        klname = klentry['name']

        if kltype not in kltype_dict:
            raise Exception("Unsupported transform.")

        kl = kltype_dict[kltype].from_config(klentry, bt, subdir=klname)
        klobj_dict[klname] = kl

        if yconf['config']['kltransform']:
            kl.generate()




## Power spectrum estimation configuration
pstype_dict =   {   'Full'          : psestimation.PSEstimation,
                    'MonteCarlo'    : psmc.PSMonteCarlo
                }

if yconf['config']['psfisher']:
    if 'psfisher' not in yconf:
        raise Exception('Require a psfisher section if config: psfisher is Yes.')

    for psentry in yconf['psfisher']:
        pstype = psentry['type']
        klname = psentry['klname']
        psname = psentry['name'] if 'name' in psentry else 'ps'
        
        if pstype not in pstype_dict:
            raise Exception("Unsupported PS estimation.")

        if klname not in klobj_dict:
            raise Exception('Desired KL object does not exist.')


        ps = pstype_dict[pstype].from_config(psentry, klobj_dict[klname], subdir=psname)
        ps.genbands()
        ps.fisher_mpi()




## Projections code
if yconf['config']['projections']:
    if 'projections' not in yconf:
        raise Exception('Require a projections section if config: projections is Yes.')

    for projentry in yconf['projections']:
        klname = projentry['klname']

        # Override default and ensure we copy the original maps
        if 'copy_orig' not in projentry:
            projentry['copy_orig'] = True

        for mentry in projentry['maps']:
            if 'stem' not in mentry:
                raise Exception('No stem in mapentry %s' % mentry['file'])

            mentry['stem'] = outdir + '/projections/' + klname + '/' + mentry['stem'] + '/'
        
        if klname not in klobj_dict:
            raise Exception('Desired KL object does not exist.')

        proj = projection.Projector.from_config(projentry, klobj_dict[klname])
        proj.generate()







