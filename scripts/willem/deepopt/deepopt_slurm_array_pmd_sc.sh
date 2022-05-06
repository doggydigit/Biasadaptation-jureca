#!/bin/bash
#
#SBATCH --job-name=deepopt_pmd_sc
#SBATCH --output=deepopt_pmd_sc.txt
#
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-54


conda activate main

nh_arr=(10 25 50 100 250 500)
task_arr=("1vall" "1vall" "1vall")
ntask_arr=(47 47 47)
ro_arr=("tanh" "sigmoid1" "sigmoid10")
bool_arr=(0 1 2)

N1=6
N2=3
NX=$((N1*N2))

for task_id in {0..53}
do

t_id=$((task_id%NX))

nh_idx=$((t_id%N1))
ro_idx=$((t_id/N1))

tasktype_idx=$((task_id/NX))
ntask_idx=$((task_id/NX))
b_idx=$((task_id/NX))

# get the array elements
nh=${nh_arr[nh_idx]}
ro=${ro_arr[ro_idx]}
tasktype=${task_arr[tasktype_idx]}
ntask=${ntask_arr[ntask_idx]}
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
    srun python3 deepopt.py --nhidden1 $nh1 --nhidden2 $nh2 --algo1 pmdd --algo2 pmd --algoc sc --ntask $ntask --enriched True --tasktype $tasktype --readout $ro --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/
fi

done