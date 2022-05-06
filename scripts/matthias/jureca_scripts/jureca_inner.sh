#!/bin/bash
sbatch << EOF 
#!/bin/bash
#SBATCH --account=jinm60
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --output=output_$1.out
#SBATCH --error=error_$1.err
#SBATCH --time=24:00:00
#SBATCH --job-name=st_gx_$1
#SBATCH --partition=dc-cpu
module purge
jutil env activate -p jinm60 -A jinm60
source env/bin/activate
module load GCC
module load OpenMPI
module load mpi4py/3.1.3
module load tqdm
module load scikit-learn
module load matplotlib
module load Python
cd Biasadaptation-jureca/scripts/matthias/
srun python3 jureca_run.py $1
deactivate
EOF
