#!/bin/bash
#
#SBATCH --job-name=deepopt_scd_bp
#SBATCH --output=deepopt_scd_bp.txt
#
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-54



# conda activate main

nh_arr=(10 25 50 100 250 500)
readout_arr=("tanh" "sigmoid1" "sigmoid10")
bool_arr=(0 1 2)
N1=3
N2=18

for task_id in {0..53}
do

# indices to number of hidden units and method
t_id=$((task_id%N2))
nh_idx=$((t_id/N1))
m_idx=$((t_id%N1))
b_idx=$((task_id/N2))
# get the array elements
nh=${nh_arr[nh_idx]}
readout=${readout_arr[m_idx]}
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

# echo "$readout"
# echo "$nh1"
# echo "$nh2"

if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python3 deepopt.py --nhidden1 $nh1 --nhidden2 $nh2 --algo1 scd --algo2 bp --ntask 47 --tasktype 1vall --readout $readout --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/
fi

done