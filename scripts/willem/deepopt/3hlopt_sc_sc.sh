#!/bin/bash
#
#SBATCH --job-name=3hl_sc_sc
#SBATCH --output=3hl_sc_sc.txt
#
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-17


# conda activate main
nh_arr=(10 25 50 100 250 500)
task_arr=("1vall" "1vall" "1vall")
ntask_arr=(47 47 47)
ro_arr=("tanh")
bool_arr=(0 1 2)

N1=6
N2=1
NX=$((N1*N2))

for task_id in {0..17}
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

if [ "$bval" -eq "0" ]
then
    nh1=${nh}
    nh2=${nh}
    nh3=100
else
    if [ "$bval" -eq "1" ]
    then
        nh1=100
        nh2=100
        nh3=${nh}
    else
        nh1=${nh}
        nh2=${nh}
        nh3=${nh}
    fi
fi

echo ""
echo "$nh1 $nh2 $nh3"
echo "$ro"
echo "$tasktype"
echo "$ntask"

# if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
if [ "$task_id" -eq "0" ]
then
    srun python3 deepopt3.py --nhidden1 $nh1 --nhidden2 $nh2 --nhidden3 $nh3 --algo1 scd --algo2 sc --algo3 sc --algoc sc --ntask $ntask --tasktype $tasktype --readout $ro --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/
fi

done