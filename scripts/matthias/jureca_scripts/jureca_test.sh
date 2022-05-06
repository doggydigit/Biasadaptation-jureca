#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --account=jinm60
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.er
#SBATCH --time=24:00:00
#SBATCH --job-name=t_g_wx
#SBATCH --partition=dc-cpu
source env/bin/activate
cd BiasAdaptation/scripts/matthias/
srun python3 jureca_run.py 0
deactivate
