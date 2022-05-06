#!/bin/bash
sbatch << EOF 
#!/bin/bash
#SBATCH --job-name=$4
#SBATCH --output=out_$4.out
#SBATCH --error=err_$4.err
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04-00:00:00
#SBATCH --nodes=1
source env/bin/activate
cd BiasAdaptation/scripts/matthias/
srun python3 $1 $2 $3 > log/$4.log
deactivate
EOF
