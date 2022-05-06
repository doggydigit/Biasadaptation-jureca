#!/bin/bash
sbatch << EOF 
#!/bin/bash
#SBATCH --job-name=$7
#SBATCH --output=out_$7.out
#SBATCH --error=err_$7.err
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04-00:00:00
#SBATCH --nodes=1
source env/bin/activate
cd BiasAdaptation/scripts/matthias/
srun python3 $1 $2 $3 $4 $5 $6 > log/$7.log
deactivate
EOF
