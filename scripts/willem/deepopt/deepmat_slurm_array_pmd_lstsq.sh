#!/bin/bash
#
#SBATCH --job-name=deepmat_pmd_lstsq
#SBATCH --output=deepmat_pmd_lstsq.txt
#
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000M
#SBATCH -a 0-17


conda activate main

nh_arr=(10 25 50 100 250 500)
for task_id in {0..17}
do

nh_idx=$((task_id%6))
# get the array elements
nh=${nh_arr[nh_idx]}

if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    if [ "$task_id" -gt 11 ]
    then
            srun python3 deepweights.py --nhidden1 100 --nhidden2 $nh --algo1 pmdd --algo2 pmd --algoc lstsq --enrich True --path /users/wybo/Data/weight_matrices/
    else
        if [ "$task_id" -gt 5 ]
        then
            srun python3 deepweights.py --nhidden1 $nh --nhidden2 100 --algo1 pmdd --algo2 pmd --algoc lstsq --enrich True --path /users/wybo/Data/weight_matrices/
        else
            srun python3 deepweights.py --nhidden1 $nh --nhidden2 $nh --algo1 pmdd --algo2 pmd --algoc lstsq --enrich True --path /users/wybo/Data/weight_matrices/
        fi
    fi
fi

done