#!/bin/bash
#
#SBATCH --job-name=codemat_sc
#SBATCH --output=codemat_sc.txt
#
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000M
#SBATCH -a 0-12


conda activate main

nh_arr=(10 25 50 100 250 500)
for task_id in {0..11}
do

nh_idx=$((task_id%6))
# get the array elements
nh=${nh_arr[nh_idx]}

if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    if [ "$task_id" -gt 5 ]
    then
        srun python3 codedataset.py --nhidden $nh --algow sc --algoc sc --path /users/wybo/Data/code_matrices/
    else
        srun python3 codedataset.py --nhidden $nh --algow sc --algoc lstsq --path /users/wybo/Data/code_matrices/
    fi
fi

done