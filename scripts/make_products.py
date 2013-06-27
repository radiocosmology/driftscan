import os.path
import argparse
import shutil

import yaml

from drift.core import manager


products = None


def run_config(args):
    m = manager.ProductManager.from_config(args.configfile)
    m.generate()


def interactive_config(args):
    global products
    products = manager.ProductManager.from_config(args.configfile)

    print "*** Access analysis products through the global variable `products` ***"




def queue_config(args):

    with open(args.configfile, 'r') as f:
        yconf = yaml.safe_load(f)

    ## Global configuration
    ## Create output directory and copy over params file.
    if 'config' not in yconf:
        raise Exception('Configuration file must have an \'config\' section.')

    conf = yconf['config'] 

    outdir = conf['output_directory'] if 'output_directory' in conf else conf['timestream_directory']
    outdir = os.path.normpath(os.path.expandvars(os.path.expanduser(outdir)))

    if not os.path.isabs(outdir):
        raise Exception("Output directory path must be absolute.")

    pbsdir = os.path.normpath(outdir + '/pbs/')

    # Create directory if required
    if not os.path.exists(pbsdir):
        os.makedirs(pbsdir)

    # Copy config file into output directory (check it's not already there first)
    sfile = os.path.realpath(os.path.abspath(args.configfile.name))
    dfile = os.path.realpath(os.path.abspath(pbsdir + '/config.yaml'))

    if sfile != dfile:
        shutil.copy(sfile, dfile)

    conf['mpiproc'] = conf['nodes'] * conf['pernode']
    conf['pbsdir'] = pbsdir
    conf['scriptpath'] = os.path.realpath(__file__)

    script="""#!/bin/bash
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

    with open(scriptname, 'w') as f:
        f.write(script)

    if not args.nosubmit:
        os.system('cd %s; qsub jobscript.sh' % pbsdir)





parser = argparse.ArgumentParser(description='Create/load the analysis products.')
subparsers = parser.add_subparsers(help='Command to run.', title="Commands", metavar="<command>")


parser_run = subparsers.add_parser('run', help='Make the analysis products from the given config file.')
parser_run.add_argument('configfile', type=str, help='Configuration file to use.')
parser_run.set_defaults(func=run_config)

parser_interactive = subparsers.add_parser('interactive', help='Load the analysis products for interactive use.')
parser_interactive.add_argument('configfile', type=str, help='Configuration file to use.')
parser_interactive.set_defaults(func=interactive_config)


parser_queue = subparsers.add_parser('queue', help='Create a jobscript for creating the products and add to the PBS queue.')
parser_queue.add_argument('configfile', type=str, help='Configuration file to use.')
parser_queue.add_argument('--nosubmit', action='store_true', help='Don\'t submit the job to the queue.')
parser_queue.set_defaults(func=queue_config)

args = parser.parse_args()
args.func(args)


