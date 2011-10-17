#!/bin/bash
#PBS -l nodes=7:ppn=8
#PBS -q batch
#PBS -r n
#PBS -m abe
#PBS -l walltime=01:00:00
#PBS -N cylinder_gen2

source /home/jrs65/pythonsetupbatch.sh

cd /home/jrs65/code/cylinder_simulation/scripts
export OMP_NUM_THREADS=2
mpirun -np 28 -npernode 4 python telescope_gen2.py &> /scratch/jrs65/cylinder/t2gen.out