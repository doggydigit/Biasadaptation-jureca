#!/bin/bash
#
#SBATCH --job-name=usagewmat_sc
#SBATCH --output=usagewmat_sc.txt
#
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-5

nh_arr=(10 25 50 100 250 500)

respath="/users/wybo/Data/results/biasopt/"
weightpath="/users/wybo/Data/weight_matrices/"
datasetpath="/work/users/wybo/Data/"

for task_id in {0..5}
do
    # number of hidden units
    nh=${nh_arr[task_id]}

    if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
    then
        srun python3 usagewmats.py --algo1 scd --nhidden1 $nh --nperbatch 50000 --ndiffmat 100000 --respath $respath --datasetpath $datasetpath --weightpath $weightpath
    fi

done