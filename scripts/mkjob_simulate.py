import argparse
import os
import os.path
import shutil

import yaml


parser = argparse.ArgumentParser(description='Submit a simulation job.')
parser.add_argument('configfile', type=argparse.FileType('r'), help='The configuration file to use.')

args = parser.parse_args()
yconf = yaml.safe_load(args.configfile)




## Global configuration
## Create output directory and copy over params file.
if 'config' not in yconf:
    raise Exception('Configuration file must have an \'config\' section.')

conf = yconf['config'] 

outdir = conf['output_directory']

if not os.path.isabs(outdir):
    raise Exception("Output directory path must be absolute.")

pbsdir = os.path.normpath(outdir + '/pbs/')

# Create directory if required
if not os.path.exists(pbsdir):
    os.makedirs(pbsdir)

shutil.copy(args.configfile.name, pbsdir + '/config.yaml')

conf['mpiproc'] = conf['nodes'] * conf['pernode']
conf['pbsdir'] = pbsdir
conf['scriptpath'] = os.path.dirname(os.path.realpath(__file__)) + '/simulate.py'


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
mpirun -np %(mpiproc)i -npernode %(pernode)i python %(scriptpath)s config.yaml &> jobout.log
"""

script = script % conf

scriptname = pbsdir + "/jobscript.sh"

with open(scriptname, 'w') as f:
    f.write(script)

os.system('cd %s; qsub jobscript.sh' % pbsdir)