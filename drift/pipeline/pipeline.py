import os.path

import yaml

from caput import config

from drift.core import manager
from drift.pipeline import timestream


def fixpath(path):
    """Fix up path (expanding variables etc.)"""
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    path = os.path.normpath(path)

    return path


class PipelineManager(config.Reader):
    """Manage and run the pipeline.

    Attributes
    ----------
    timestream_directory : string
        Directory that the timestream is stored in.
    product_directory : string
        Directory that the analysis products are stored in.
    output_directory : string
        Directory to store timestream outputs in.

    generate_modes : boolean
        Calculate m-modes and svd-modes.
    generate_klmodes : boolean
        Calculate KL-modes?
    generate_powerspectra : boolean
        Estimate powerspectra?

    klmodes : list
        List of KL-filters to apply ['klname1', 'klname2', ...]
    powerspectra : list
        List of powerspectra to apply. Requires entries to be dicts
        like [ { 'psname' : 'ps1', 'klname' : 'dk'}, ...]
    """

    # Directories
    product_directory = config.Property(proptype=str, default="")

    # Actions to perform
    generate_modes = config.Property(proptype=bool, default=True)
    generate_klmodes = config.Property(proptype=bool, default=True)
    generate_powerspectra = config.Property(proptype=bool, default=True)
    generate_maps = config.Property(proptype=bool, default=True)

    no_m_zero = config.Property(proptype=bool, default=True)

    # Specific products to use.
    klmodes = config.Property(proptype=list, default=[])
    powerspectra = config.Property(proptype=list, default=[])
    klmaps = config.Property(proptype=list, default=[])
    crosspower = []

    # Specific map-making options
    nside = config.Property(proptype=int, default=128)
    wiener = config.Property(proptype=bool, default=False)

    timestreams = {}
    simulations = {}
    manager = None

    collect_klmodes = config.Property(proptype=bool, default=True)

    @classmethod
    def from_configfile(cls, configfile):
        c = cls()
        c.load_configfile(configfile)

        return c

    def load_configfile(self, configfile):
        with open(configfile, "r") as f:
            yconf = yaml.safe_load(f)

        ## Global configuration
        ## Create output directory and copy over params file.
        if "config" not in yconf:
            raise Exception("Configuration file must have an 'config' section.")

        # Load config in from file.
        self.read_config(yconf["config"])

        # Load in timestream information
        if "timestreams" not in yconf:
            raise Exception("Configuration file must have an 'timestream' section.")

        for tsconf in yconf["timestreams"]:
            name = tsconf["name"]
            tsdir = fixpath(tsconf["directory"])

            # Load ProductManager and Timestream
            pm = manager.ProductManager.from_config(self.product_directory)
            ts = timestream.Timestream(tsdir, pm)

            if "output_directory" in tsconf:
                outdir = fixpath(tsconf["output_directory"])
                ts.output_directory = outdir

            ts.no_m_zero = self.no_m_zero

            self.timestreams[name] = ts

            if "simulate" in tsconf:
                self.simulations[name] = tsconf["simulate"]

        if "crosspower" in yconf:
            self.crosspower = [xp for xp in yconf["crosspower"]]

    def simulate(self):
        for tsname, simconf in self.simulations.items():
            ts = self.timestreams[tsname]

            if os.path.exists(ts._ffile(0)):
                print("Looks like timestream already exists. Skipping....")
            else:
                m = manager.ProductManager.from_config(simconf["product_directory"])
                timestream.simulate(m, ts.directory, **simconf)

    def generate(self):
        """Generate pipeline outputs."""

        if self.generate_modes:
            for tsname, tsobj in self.timestreams.items():
                print("Generating modes (%s)" % tsname)

                tsobj.generate_mmodes()
                tsobj.generate_mmodes_svd()

        if self.generate_klmodes:
            for tsname, tsobj in self.timestreams.items():
                for klname in self.klmodes:
                    print("Generating KL filter (%s:%s)" % (tsname, klname))

                    tsobj.set_kltransform(klname)
                    tsobj.generate_mmodes_kl()

                    if self.collect_klmodes:
                        tsobj.collect_mmodes_kl()

        if self.generate_powerspectra:
            for tsname, tsobj in self.timestreams.items():
                for ps in self.powerspectra:
                    psname = ps["psname"]
                    klname = ps["klname"]

                    print("Estimating powerspectra (%s:%s)" % (tsname, psname))

                    tsobj.set_kltransform(klname)
                    tsobj.set_psestimator(psname)

                    tsobj.powerspectrum()

            for xp in self.crosspower:
                psname = xp["psname"]
                klname = xp["klname"]

                tslist = []

                for tsname in xp["timestreams"]:
                    tsobj = self.timestreams[tsname]

                    tsobj.set_kltransform(klname)
                    tsobj.set_psestimator(klname)

                    tslist.append(tsobj)

                psfile = os.path.abspath(
                    os.path.expandvars(os.path.expanduser(xp["psfile"]))
                )

                timestream.cross_powerspectrum(tslist, psname, psfile)

        if self.generate_maps:
            for tsname, tsobj in self.timestreams.items():
                for klname in self.klmaps:
                    print("Generating KL map (%s:%s)" % (tsname, klname))

                    mapfile = "map_%s.hdf5" % klname

                    tsobj.set_kltransform(klname)
                    tsobj.mapmake_kl(self.nside, mapfile, wiener=self.wiener)

                print("Generating SVD map (%s)" % tsname)
                tsobj.mapmake_svd(self.nside, "map_svd.hdf5")

                print("Generating full map (%s)" % tsname)
                tsobj.mapmake_full(self.nside, "map_full.hdf5")

    run = generate
