#!/usr/bin/env bash
#SBATCH --account=jinm60
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.er
#SBATCH --time=24:00:00
#SBATCH --job-name=t_g_wx
#SBATCH --partition=dc-cpu
module purge
source env/bin/activate
module load Python
module load GCC
module load OpenMPI
module load mpi4py/3.1.3
module load tqdm
module load scikit-learn
module load matplotlib
cd BiasAdaptation-jureca/scripts/matthias/
srun python3 jureca_run.py 0
deactivate
