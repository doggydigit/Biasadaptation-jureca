#!/bin/bash
#
#SBATCH --job-name=biasopt_pmdd
#SBATCH --output=biasopt_pmdd.txt
#
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH -a 0-36


conda activate main

nh_arr=(10 25 50 100 250 500)
readout_arr=("linear" "hardtanh" "tanh" "sigmoid1" "sigmoid5" "sigmoid10")
N1=6

for task_id in {0..35}
do

# indices to number of hidden units and method
nh_idx=$((task_id/N1))
m_idx=$((task_id%N1))
# get the array elements
readout=${readout_arr[m_idx]}
nh=${nh_arr[nh_idx]}

if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python3 biasopt.py --nhidden $nh --algo pmdd --ntask 47 --tasktype 1vall --readout $readout --path /users/wybo/Data/results/biasopt/ --nbfactor 10 --nepoch 100
fi

done