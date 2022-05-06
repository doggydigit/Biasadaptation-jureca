#!/bin/bash

#SBATCH --job-name=sbatch_example
#SBATCH --output=sbatch_example.out
#SBATCH --error=sbatch_example.err
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04-00:00:00    
#SBATCH --nodes=1


source env/bin/activate
cd BiasAdaptation/scripts/matthias/
srun python3 python_script_name.py > log/logfile_name.log
deactivate
