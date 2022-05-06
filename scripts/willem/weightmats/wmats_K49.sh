#!/bin/bash
#
#SBATCH --job-name=wm_K49
#SBATCH --output=wm_K49.txt
#
#SBATCH --time=150:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-2

nh_arr=(1000 2500)

weightpath="/users/wybo/Data/weight_matrices2/"
datasetpath="/work/users/wybo/Data/"

for task_id in {0..5}
do
    # number of hidden units
    nh=${nh_arr[task_id]}

    if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
    then
        srun python3 allopt.py --methods pmdd --nhidden $nh --ndata 100000 --datasets K49 --path $weightpath --datapath $datasetpath
    fi

done