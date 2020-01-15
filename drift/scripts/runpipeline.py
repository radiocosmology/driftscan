#!/usr/bin/env python

# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import click

pipe = None


@click.group()
def cli():
    """Generate pipeline products.

    This is deprecated. You should really use `draco` instead.
    """
    pass


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
def run_config(configfile):
    from drift.pipeline import pipeline

    pl = pipeline.PipelineManager.from_configfile(configfile)
    pl.simulate()
    pl.generate()


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
def interactive_config(configfile):
    from drift.pipeline import pipeline

    global pipe
    pipe = pipeline.PipelineManager.from_configfile(configfile)


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
@click.option(
    "--submit/--nosubmit", default=True, help="Submit the job to the queue (or not)"
)
def queue_config(configfile, submit):

    import os
    import os.path
    import shutil
    import yaml

    yconf = yaml.safe_load(configfile)

    ## Global configuration
    ## Create output directory and copy over params file.
    if "config" not in yconf:
        raise Exception("Configuration file must have an 'config' section.")

    conf = yconf["config"]

    outdir = (
        conf["output_directory"]
        if "output_directory" in conf
        else conf["timestream_directory"]
    )
    outdir = os.path.normpath(os.path.expandvars(os.path.expanduser(outdir)))

    if not os.path.isabs(outdir):
        raise Exception("Output directory path must be absolute.")

    pbsdir = os.path.normpath(outdir + "/pbs/")

    # Create directory if required
    if not os.path.exists(pbsdir):
        os.makedirs(pbsdir)

    # Copy config file into output directory (check it's not already there first)
    sfile = os.path.realpath(os.path.abspath(configfile.name))
    dfile = os.path.realpath(os.path.abspath(pbsdir + "/config.yaml"))

    if sfile != dfile:
        shutil.copy(sfile, dfile)

    conf["mpiproc"] = conf["nodes"] * conf["pernode"]
    conf["pbsdir"] = pbsdir
    conf["scriptpath"] = os.path.realpath(__file__)

    script = """#!/bin/bash
#PBS -l nodes=%(nodes)i:ppn=%(ppn)i
#PBS -q %(queue)s
#PBS -r n
#PBS -m abe
#PBS -V
#PBS -l walltime=%(time)s
#PBS -N %(name)s


cd %(pbsdir)s
export OMP_NUM_THREADS=%(ompnum)i

mpirun --mca btl self,sm,tcp -np %(mpiproc)i -npernode %(pernode)i python %(scriptpath)s run config.yaml &> jobout.log
"""

    script = script % conf

    scriptname = pbsdir + "/jobscript.sh"

    with open(scriptname, "w") as f:
        f.write(script)

    if submit:
        os.system("cd %s; qsub jobscript.sh" % pbsdir)
