import argparse
import os.path

import yaml

from drift.util import mpiutil

from drift.telescope import (
    cylinder,
    gmrt,
    focalplane,
    restrictedcylinder,
    exotic_cylinder,
)
from drift.core import beamtransfer

from drift.core import kltransform, doublekl
from drift.core import psestimation, psmc
from drift.core import skymodel

from drift.core import projection

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

cfile = os.path.realpath(os.path.abspath(args.configfile.name))
args.configfile.close()

outdir_orig = outdir
if not os.path.isabs(outdir):
    outdir = os.path.normpath(os.path.join(os.path.dirname(cfile), outdir))


# Create directory if required
if mpiutil.rank0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(cfile, "r") as f:
        config_contents = f.read()

    # Construct new file path
    dfile = os.path.join(outdir, "config.yaml")

    # Rewrite path in config file
    if outdir_orig != outdir:
        config_contents = config_contents.replace(outdir_orig, outdir)

    # Write config file into local copy
    with open(dfile, "w+") as f:
        f.write(config_contents)


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
    "CylinderExtra": exotic_cylinder.CylinderExtra,
    "GradientCylinder": exotic_cylinder.GradientCylinder,
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

if yconf["config"]["beamtransfers"]:
    bt.generate()


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

        if yconf["config"]["kltransform"]:
            kl.generate()


## Power spectrum estimation configuration
pstype_dict = {
    "Full": psestimation.PSEstimation,
    "MonteCarlo": psmc.PSMonteCarlo,
    "MonteCarloAlt": psmc.PSMonteCarloAlt,
}

if yconf["config"]["psfisher"]:
    if "psfisher" not in yconf:
        raise Exception("Require a psfisher section if config: psfisher is Yes.")

    for psentry in yconf["psfisher"]:
        pstype = psentry["type"]
        klname = psentry["klname"]
        psname = psentry["name"] if "name" in psentry else "ps"

        if pstype not in pstype_dict:
            raise Exception("Unsupported PS estimation.")

        if klname not in klobj_dict:
            raise Exception("Desired KL object does not exist.")

        ps = pstype_dict[pstype].from_config(psentry, klobj_dict[klname], subdir=psname)
        ps.generate()


## Projections code
if yconf["config"]["projections"]:
    if "projections" not in yconf:
        raise Exception("Require a projections section if config: projections is Yes.")

    for projentry in yconf["projections"]:
        klname = projentry["klname"]

        # Override default and ensure we copy the original maps
        if "copy_orig" not in projentry:
            projentry["copy_orig"] = True

        for mentry in projentry["maps"]:
            if "stem" not in mentry:
                raise Exception("No stem in mapentry %s" % mentry["file"])

            mentry["stem"] = (
                outdir + "/projections/" + klname + "/" + mentry["stem"] + "/"
            )

        if klname not in klobj_dict:
            raise Exception("Desired KL object does not exist.")

        proj = projection.Projector.from_config(projentry, klobj_dict[klname])
        proj.generate()


if mpiutil.rank0:
    print("========================================")
    print("=                                      =")
    print("=           DONE AT LAST!!             =")
    print("=                                      =")
    print("========================================")
