#!/usr/bin/env python

import click

products = None


@click.group()
def cli():
    """Generate data to allow modelling and analysis of driftscan interferometers.

    This command can take a configuration file (in yaml format) describing the telescope to
    simulate and generate products such as beam transfer matrices, KL
    foreground filters and power spectrum estimators.
    """
    pass


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
def run(configfile):
    """Immediately run the yaml formatted CONFIGFILE to generate products."""
    from drift.core import manager

    m = manager.ProductManager.from_config(configfile)
    m.generate()


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
def interactive(configfile):
    """Load the yaml formatted CONFIGFILE, but do not start generating the products.

    This command can be used for interactive exploration of existing sets of
    data products by doing:

    $ ipython -i $(which drift-makeproducts) interactive config.yaml
    """
    from drift.core import manager

    global products
    products = manager.ProductManager.from_config(configfile)

    print("*** Access analysis products through the global variable `products` ***")


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
@click.option(
    "--submit/--nosubmit", default=True, help="Submit the job to the queue (or not)"
)
def queue(configfile, submit):
    """Submit the CONFIGFILE to be processed in a batch queue on a cluster.

    This will queue up a job (with parameters from the `cluster` section of
    the config), that will generate all the products in a batch job.
    """

    import os.path
    import shutil
    import yaml

    with open(configfile, "r") as f:
        yconf = yaml.safe_load(f)

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

    # Get the name of the scheduler
    if "queue_sys" in conf:
        queue_sys = conf["queue_sys"]
    else:
        queue_sys = "pbs"
        raise Warning("Queueing system not set, defaulting to PBS")

    # Use it to create submitdir
    submitdir = os.path.normpath(outdir + "/" + queue_sys + "/")

    # Create directory if required
    if not os.path.exists(submitdir):
        os.makedirs(submitdir)

    # Copy config file into output directory (check it's not already there first)
    sfile = os.path.realpath(os.path.abspath(configfile))
    dfile = os.path.realpath(os.path.abspath(submitdir + "/config.yaml"))

    if sfile != dfile:
        shutil.copy(sfile, dfile)

    clusterconf = {}

    # Set up required PBS vars
    if "nodes" not in conf:
        raise Exception("Nodes is required.")
    clusterconf["nodes"] = conf["nodes"]

    if "time" not in conf:
        raise Exception("Job time is required.")
    clusterconf["time"] = conf["time"]

    # Set queueing system w. defaults
    cluster_defaults = {
        "pbs": {"ppn": 8, "mem": "16000M", "account": None, "submit": "qsub"},
        "slurm": {"ppn": 32, "mem": "0", "account": None, "submit": "sbatch"},
    }

    if queue_sys in cluster_defaults:

        clusterconf["queue_sys"] = queue_sys
        clusterconf["ppn"] = (
            conf["ppn"] if "ppn" in conf else cluster_defaults[queue_sys]["ppn"]
        )
        clusterconf["mem"] = (
            conf["mem"] if "mem" in conf else cluster_defaults[queue_sys]["mem"]
        )
        clusterconf["account"] = (
            conf["account"]
            if "account" in conf
            else cluster_defaults[queue_sys]["account"]
        )

        submit_command = (
            conf["submit_command"]
            if "submit_command" in conf
            else cluster_defaults[queue_sys]["submit"]
        )

    else:
        clusterconf["queue_sys"] = queue_sys
        clusterconf["ppn"] = conf["ppn"] if "ppn" in conf else 8
        clusterconf["mem"] = conf["mem"] if "mem" in conf else "0"
        clusterconf["account"] = conf["account"] if "account" in conf else None

        if "submit_command" in conf:
            submit_command = conf["submit_command"]
        else:
            raise Exception("Need to specify submit command for unknown scheduler.")

        if "script_template" in conf:
            script_templates[queue_sys] = conf["script_template"]
        else:
            raise Exception(
                "Need to specify submit script string for unknown scheduler."
            )

    # Set up optional PBS vars
    clusterconf["ompnum"] = conf["ompnum"] if "ompnum" in conf else 8
    clusterconf["queue"] = conf["queue"] if "queue" in conf else "batch"
    clusterconf["pernode"] = conf["pernode"] if "pernode" in conf else 1
    clusterconf["name"] = conf["name"] if "name" in conf else "job"

    # Set vars only needed to create script
    clusterconf["mpiproc"] = clusterconf["nodes"] * clusterconf["pernode"]
    clusterconf["workdir"] = outdir
    clusterconf["scriptpath"] = os.path.realpath(__file__)
    clusterconf["logpath"] = submitdir + "/jobout.log"
    clusterconf["configpath"] = submitdir + "/config.yaml"

    # Set up virtualenv
    if "venv" in conf:
        if not os.path.exists(conf["venv"] + "/bin/activate"):
            raise Exception("Could not find virtualenv")

        clusterconf["venv"] = conf["venv"] + "/bin/activate"
    else:
        clusterconf["venv"] = "/dev/null"

    script = script_templates[queue_sys] % clusterconf

    scriptname = submitdir + "/jobscript.sh"

    with open(scriptname, "w") as f:
        f.write(script)

    if submit:
        os.system("cd %s; %s jobscript.sh" % (submitdir, submit_command))


# Put script templates for different schedulers here and add
# them to the dictionary.

pbs_script = """#!/bin/bash
#PBS -l nodes=%(nodes)i:ppn=%(ppn)i
#PBS -q %(queue)s
#PBS -r n
#PBS -m abe
#PBS -V
#PBS -l walltime=%(time)s
#PBS -N %(name)s
source %(venv)s
cd %(workdir)s
export OMP_NUM_THREADS=%(ompnum)i
mpirun -np %(mpiproc)i -npernode %(pernode)i -bind-to none python %(scriptpath)s run %(configpath)s &> %(logpath)s
"""

slurm_script = """#!/bin/bash
#SBATCH --account=%(account)s
#SBATCH --nodes=%(nodes)i
#SBATCH --ntasks-per-node=%(pernode)i # number of MPI processes
#SBATCH --cpus-per-task=%(ompnum)i # number of OpenMP processes
#SBATCH --mem=%(mem)s # memory per node
#SBATCH --time=%(time)s
#SBATCH --job-name=%(name)s

source %(venv)s
cd %(workdir)s

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun -np %(mpiproc)i -npernode %(pernode)i -bind-to none python %(scriptpath)s run %(configpath)s &> %(logpath)s
"""

script_templates = {}
script_templates["pbs"] = pbs_script
script_templates["slurm"] = slurm_script

# This is needed because the queue script calls this file directly.
if __name__ == "__main__":
    cli()
