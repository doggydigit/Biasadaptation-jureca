#!/bin/bash
sbatch << EOF 
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=out_$1.out
#SBATCH --error=err_$1.err
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6        
source env/bin/activate
cd BiasAdaptation/scripts/matthias/
srun python3 python_script_name.py $2 $3 > log/$1.log
deactivate
EOF
