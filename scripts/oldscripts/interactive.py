import argparse
import os.path
import shutil

import yaml

from drift.util import mpiutil

from drift.telescope import cylinder, gmrt, focalplane, restrictedcylinder
from drift.core import beamtransfer

from drift.core import kltransform, doublekl
from drift.core import skymodel

parser = argparse.ArgumentParser(description="Run a simulation job.")
parser.add_argument(
    "configfile", type=argparse.FileType("r"), help="The configuration file to use."
)
args = parser.parse_args()
yconf = yaml.safe_load(args.configfile)


## Global configuration
## Create output directory and copy over params file.
if "config" not in yconf:
    raise Exception("Configuration file must have an 'config' section.")

outdir = yconf["config"]["output_directory"]

# Create directory if required
if mpiutil.rank0 and not os.path.exists(outdir):
    os.makedirs(outdir)

# Copy config file into output directory (check it's not already there first)
sfile = os.path.realpath(os.path.abspath(args.configfile.name))
dfile = os.path.realpath(os.path.abspath(outdir + "/config.yaml"))

if sfile != dfile:
    shutil.copy(sfile, dfile)


## Telescope configuration
if "telescope" not in yconf:
    raise Exception("Configuration file must have an 'telescope' section.")

teltype = yconf["telescope"]["type"]

teltype_dict = {
    "UnpolarisedCylinder": cylinder.UnpolarisedCylinderTelescope,
    "PolarisedCylinder": cylinder.PolarisedCylinderTelescope,
    "GMRT": gmrt.GmrtUnpolarised,
    "FocalPlane": focalplane.FocalPlaneArray,
    "RestrictedCylinder": restrictedcylinder.RestrictedCylinder,
    "RestrictedPolarisedCylinder": restrictedcylinder.RestrictedPolarisedCylinder,
    "RestrictedExtra": restrictedcylinder.RestrictedExtra,
}

if teltype not in teltype_dict:
    raise Exception("Unsupported telescope type.")

telescope = teltype_dict[teltype].from_config(yconf["telescope"])


if "reionisation" in yconf["config"]:
    if yconf["config"]["reionisation"]:
        skymodel._reionisation = True


## Beam transfer generation
if "nosvd" in yconf["config"] and yconf["config"]["nosvd"]:
    bt = beamtransfer.BeamTransferNoSVD(outdir + "/bt/", telescope=telescope)
else:
    bt = beamtransfer.BeamTransfer(outdir + "/bt/", telescope=telescope)

## Set the singular value cut for the beamtransfers
if "svcut" in yconf["config"]:
    bt.svcut = float(yconf["config"]["svcut"])

# if yconf['config']['beamtransfers']:
#    bt.generate()


## KLTransform configuration
kltype_dict = {"KLTransform": kltransform.KLTransform, "DoubleKL": doublekl.DoubleKL}
klobj_dict = {}


if "kltransform" in yconf:

    for klentry in yconf["kltransform"]:
        kltype = klentry["type"]
        klname = klentry["name"]

        if kltype not in kltype_dict:
            raise Exception("Unsupported transform.")

        kl = kltype_dict[kltype].from_config(klentry, bt, subdir=klname)
        klobj_dict[klname] = kl

#        if yconf['config']['kltransform']:
#            kl.generate()

kl = klobj_dict
