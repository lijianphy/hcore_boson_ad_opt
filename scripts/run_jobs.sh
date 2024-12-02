#!/bin/bash
#SBATCH -J compare_converge  # job name
#SBATCH -N 1                 # total number of nodes
#SBATCH --ntasks-per-node=4  # MPI tasks per node
#SBATCH -p batch             # partition

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$HOME/opt/petsc-3.22.1/linux-gnu-complex/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/opt/slepc-3.22.1/linux-gnu-complex/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/opt/OpenBLAS/lib:$LD_LIBRARY_PATH

## for each config file (config/*.json) in the config directory, run the job
for config in $(ls config/*.json); do
    echo "Running job for config file: $config"
    # mpirun -np 4 ./test_optimize $config
    mpirun ./test_optimize $config
done
