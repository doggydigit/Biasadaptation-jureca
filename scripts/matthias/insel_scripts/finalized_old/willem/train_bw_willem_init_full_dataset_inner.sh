#!/bin/bash
sbatch << EOF 
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=out_$1.out
#SBATCH --error=err_$1.err
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04-00:00:00
#SBATCH --nodes=1
source env/bin/activate
cd BiasAdaptation/scripts/matthias/
srun python3 train_full_dataset.py train_bw_willem_init $2 $3 $4 > log/$1.log
deactivate
EOF
