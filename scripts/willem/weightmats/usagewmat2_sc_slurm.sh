#!/bin/bash
#
#SBATCH --job-name=usagewmat2_sc
#SBATCH --output=usagewmat2_sc.txt
#
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-17

nh_arr=(10 25 50 100 250 500)
bool_arr=(0 1 2)


respath="/users/wybo/Data/results/biasopt/"
weightpath="/users/wybo/Data/weight_matrices/"
datasetpath="/work/users/wybo/Data/"

for task_id in {0..17}
do

    b_id=$((task_id%3))
    nh_id=$((task_id/3))

    # number of hidden units
    nh=${nh_arr[nh_id]}
    bval=${bool_arr[b_id]}


    if [ $bval -eq 0 ]
    then
        nh1=${nh}
        nh2=100
    else
        if [ $bval -eq 1 ]
        then
            nh1=100
            nh2=${nh}
        else
            nh1=${nh}
            nh2=${nh}
        fi
    fi

    if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
    then
        srun python3 usagewmats.py --algo1 scd --algo2 sc --algoc sc --nhidden1 $nh1 --nhidden2 $nh2 --nperbatch 50000 --ndiffmat 100000 --respath $respath --datasetpath $datasetpath --weightpath $weightpath
    fi

done