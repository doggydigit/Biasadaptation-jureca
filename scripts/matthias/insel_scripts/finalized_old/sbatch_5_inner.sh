#!/bin/bash
sbatch << EOF 
#!/bin/bash
#SBATCH --job-name=$5
#SBATCH --output=out_$5.out
#SBATCH --error=err_$5.err
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04-00:00:00
#SBATCH --nodes=1
source env/bin/activate
cd BiasAdaptation/scripts/matthias/
srun python3 $1 $2 $3 $4 > log/$5.log
deactivate
EOF
