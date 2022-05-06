#!/bin/bash
#
#SBATCH --job-name=deepopt_sc_sc_reweighted
#SBATCH --output=deepopt_sc_sc_reweighted.txt
#
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-17


conda activate main

nh_arr=(10 25 50 100 250 500)
bool_arr=(0 1 2)

N1=3

for task_id in {0..17}
do


nh_idx=$((task_id/N1))
b_idx=$((task_id%N1))

# get the array elements
nh=${nh_arr[nh_idx]}
bval=${bool_arr[b_idx]}

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
    srun python3 deepopt.py --nhidden1 $nh1 --nhidden2 $nh2 --algo1 scd --algo2 sc --algoc sc --ntask 47 --tasktype 1vall --readout "tanh" --reweighted True --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/
fi

done