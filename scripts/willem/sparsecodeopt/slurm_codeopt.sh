#!/bin/bash
#
#SBATCH --job-name=codeopt
#SBATCH --output=codeopt.txt
#
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-71


# conda activate main

nh_arr=(10 25 50 100 250 500)
readout_arr=("tanh" "hardtanh" "linear" "sigmoid1" "sigmoid5" "sigmoid10")

N1=6

for task_id in {0..35}
do

# indices to number of hidden units and method
nh_idx=$((task_id/N1))
ro_idx=$((task_id%N1))
# get the array elements
nh=${nh_arr[nh_idx]}
ro=${readout_arr[ro_idx]}

# echo "$nh"
# echo $ro

t_id1=$((task_id*2))
t_id2=$((task_id*2+1))

if [ "$t_id1" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python3 codeopt.py --algow sc --algoc sc --ntask 47 --tasktype 1vall --nhidden $nh --readout $ro --path /users/wybo/Data/results/biasopt/ --datapath /work/users/wybo/Data/code_matrices/
fi

if [ "$t_id2" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python3 codeopt.py --algow sc --algoc lstsq --ntask 47 --tasktype 1vall --nhidden $nh --readout $ro --path /users/wybo/Data/results/biasopt/ --datapath /work/users/wybo/Data/code_matrices/
fi

done