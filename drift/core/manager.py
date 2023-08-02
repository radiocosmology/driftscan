"""Manage access to and generation of driftscan analysis products."""

import logging
import os.path
import warnings

import yaml

from caput import mpiutil

from drift.telescope import (
    cylinder,
    gmrt,
    focalplane,
    restrictedcylinder,
    exotic_cylinder,
)
from drift.core import beamtransfer

from drift.core import kltransform, doublekl
from drift.core import psestimation, psmc, crosspower
from drift.core import skymodel


logger = logging.getLogger(__name__)


teltype_dict = {
    "UnpolarisedCylinder": cylinder.UnpolarisedCylinderTelescope,
    "PolarisedCylinder": cylinder.PolarisedCylinderTelescope,
    "GMRT": gmrt.GmrtUnpolarised,
    "FocalPlane": focalplane.FocalPlaneArray,
    "RestrictedCylinder": restrictedcylinder.RestrictedCylinder,
    "RestrictedPolarisedCylinder": restrictedcylinder.RestrictedPolarisedCylinder,
    "RestrictedExtra": restrictedcylinder.RestrictedExtra,
    "GradientCylinder": exotic_cylinder.GradientCylinder,
    "PertCylinder": exotic_cylinder.CylinderPerturbed,
}


## KLTransform configuration
kltype_dict = {"KLTransform": kltransform.KLTransform, "DoubleKL": doublekl.DoubleKL}


## Power spectrum estimation configuration
pstype_dict = {
    "Full": psestimation.PSExact,
    "MonteCarlo": psmc.PSMonteCarlo,
    "MonteCarloAlt": psmc.PSMonteCarloAlt,
    "Cross": crosspower.CrossPower,
}


def _resolve_class(clstype, clsdict, objtype=""):
    # If clstype is a dict, try and resolve the class from `module` and
    # `class` properties. If it's a string try and resolve the class from
    # either its name and a lookup dictionary.

    if isinstance(clstype, dict):
        # Lookup custom type

        modname = clstype["module"]
        clsname = clstype["class"]

        if "file" in clstype:
            import imp

            module = imp.load_source(modname, clstype["file"])
        else:
            import importlib

            module = importlib.import_module(modname)
        cls_ref = module.__dict__[clsname]

    elif clstype in clsdict:
        cls_ref = clsdict[clstype]
    else:
        raise Exception(f"Unsupported {objtype}")

    return cls_ref


class ProductManager(object):
    """Manage access and generation to analysis products.

    This is telescope objects, beam transfer matrices, KL filters and power
    spectrum estimators.
    """

    directory = None

    gen_beams = False
    gen_kl = False
    gen_ps = False
    gen_proj = False

    skip_svd = False
    skip_svd_inv = False

    @classmethod
    def from_config(cls, configfile):
        """Create a ProductManager from a config file.

        This will create both the directory specified as the output directory
        *and* copy the configuration file into it.

        Parameters
        ----------
        configfile : string
            Path to configuration file to load.

        Returns
        -------
        m : ProductManager
        """

        configfile = os.path.normpath(
            os.path.expandvars(os.path.expanduser(configfile))
        )

        if not os.path.exists(configfile):
            raise Exception(f"Configuration file does not exist {configfile}.")

        if os.path.isdir(configfile):
            configfile = configfile + "/config.yaml"

        # Read in config file to fetch output directory
        with open(configfile, "r") as f:
            yconf = yaml.safe_load(f)

        outdir = yconf["config"]["output_directory"]

        # Path of directory-local config.yaml file
        dfile = os.path.join(outdir, "config.yaml")

        ## Create output directory and copy over params file.
        if mpiutil.rank0:
            # Create directory if required
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # Rewrite config file to make output path absolute (and put in <outdir>/config.yaml)
            if not os.path.exists(dfile) or not os.path.samefile(configfile, dfile):
                outdir_orig = outdir
                # Work out absolute path
                if not os.path.isabs(outdir):
                    outdir = os.path.abspath(
                        os.path.normpath(
                            os.path.join(os.path.dirname(configfile), outdir)
                        )
                    )

                with open(configfile, "r") as f:
                    config_contents = f.read()

                # Rewrite path in config file if not absolute
                if outdir_orig != outdir:
                    config_contents = config_contents.replace(outdir_orig, outdir)

                # Write config file into local copy
                with open(dfile, "w+") as f:
                    f.write(config_contents)

        # Need to wait until the dumped file has been created by rank=0
        mpiutil.barrier()

        # Load config into a new class and return
        c = cls()

        with open(dfile) as f:
            yconf = yaml.safe_load(f)

        c.apply_config(yconf)

        return c

    def apply_config(self, yconf):
        """Apply config from a dictionary.

        This does not create anything on disk.

        Parameters
        ----------
        yconf : dict
            Dictionary containing the configuration of the Product Manager.
        """

        # Check for required sections in files
        if "config" not in yconf:
            raise ValueError("Configuration file must have an 'config' section.")

        if "telescope" not in yconf:
            raise ValueError("Configuration file must have an 'telescope' section.")

        # Keep a copy of the config
        self.config = yconf

        ## Global configuration
        self.directory = yconf["config"]["output_directory"]
        self.directory = os.path.expanduser(self.directory)
        self.directory = os.path.expandvars(self.directory)

        if mpiutil.rank0:
            logger.info(f"Product directory: {self.directory}")

        ## Telescope configuration
        teltype = yconf["telescope"]["type"]
        telclass = _resolve_class(teltype, teltype_dict, "telescope")
        self.telescope = telclass.from_config(yconf["telescope"])

        if yconf["config"].get("reionisation"):
            skymodel._reionisation = True

        ## Beam transfer generation

        # Decide on type of beam transfers
        btclass = beamtransfer.BeamTransfer
        if yconf["config"].get("nosvd"):  # Use no SVD if requested
            btclass = beamtransfer.BeamTransferNoSVD
        if yconf["config"].get("fullsvd"):  # Use the full SVD if requested
            btclass = beamtransfer.BeamTransferFullSVD

        # Create the beam transfer manager
        self.beamtransfer = btclass(self.directory + "/bt/", telescope=self.telescope)

        # Read any configuration options for the beam transfer generation
        self.beamtransfer.read_config(yconf["config"])

        if yconf["config"].get("beamtransfers"):
            self.gen_beams = True

        if yconf["config"].get("skip_svd"):
            self.skip_svd = True

        ## Configure the KL Transforms
        self.kltransforms = {}

        if "kltransform" in yconf:
            for klentry in yconf["kltransform"]:
                kltype = klentry["type"]
                klname = klentry["name"]

                klclass = _resolve_class(kltype, kltype_dict, "KL filter")

                kl = klclass.from_config(klentry, self.beamtransfer, subdir=klname)
                self.kltransforms[klname] = kl

        if yconf["config"].get("kltransform"):
            self.gen_kl = True

        ## Configure the PS estimators
        self.psestimators = {}

        if yconf["config"].get("psfisher"):
            self.gen_ps = True

            if "psfisher" not in yconf:
                raise Exception(
                    "Require a psfisher section if config: psfisher is Yes."
                )

        if "psfisher" in yconf:
            for psentry in yconf["psfisher"]:
                pstype = psentry["type"]
                klname = psentry["klname"]
                psname = psentry["name"] if "name" in psentry else "ps"

                psclass = _resolve_class(pstype, pstype_dict, "PS estimator")

                if klname not in self.kltransforms:
                    warnings.warn(f"Desired KL object (name: {klname}) does not exist.")
                    self.psestimators[psname] = None
                else:
                    self.psestimators[psname] = psclass.from_config(
                        psentry, self.kltransforms[klname], subdir=psname
                    )

    def generate(self):
        """Calculate the analysis products."""

        # Create the directory if it does not exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Dump the config from the internal setup
        with open(os.path.join(self.directory, "configdump.yaml"), "w") as fh:
            yaml.dump(self.config, fh)

        # Generate the transfer matrices
        if self.gen_beams:
            self.beamtransfer.generate(skip_svd=self.skip_svd)

        # Generate the KLs
        if self.gen_kl:
            for klname, klobj in self.kltransforms.items():
                klobj.generate()

        # Generate the PS estimators
        if self.gen_ps:
            for psname, psobj in self.psestimators.items():
                psobj.generate()
                psobj.delbands()

        if mpiutil.rank0:
            logger.info("DONE GENERATING PRODUCTS")
