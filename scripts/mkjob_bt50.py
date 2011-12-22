
import sys
import os
import os.path

bi = int(sys.argv[1])

name = "bt50_%i" % bi

script="""#!/bin/bash
#PBS -l nodes=40:ppn=8
#PBS -q batch
#PBS -r n
#PBS -m abe
#PBS -l walltime=9:00:00
#PBS -N %(name)s

source $HOME/pythonsetupbatch.sh

cd $HOME/code/cylinder_simulation/scripts
export OMP_NUM_THREADS=8
mpirun -np 40 -npernode 1 python chime_btcomb50.py %(bi)i &> $SCRATCH/cylinder/%(name)s.out
"""

script = script % { 'bi' : bi, 'name' : name}

scriptname = "jobscript_%s.sh" % name

with open(scriptname, 'w') as f:
    f.write(script)

os.system('cd ~; pwd; qsub  %s' % os.path.abspath(scriptname))
