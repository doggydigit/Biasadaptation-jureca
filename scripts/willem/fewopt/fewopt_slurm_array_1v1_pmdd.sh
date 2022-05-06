#!/bin/bash
#
#SBATCH --job-name=fewopt_1v1_pmdd
#SBATCH --output=fewopt_1v1_pmdd.txt
#
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH -a 0-72


conda activate main

nh_arr=(10 25 50 100 250 500)
nsample_arr=(1 2 3 5 10 20 30 50 100 200 300 500)
N1=12
N2=6

for task_id in {0..71}
do

# indices to number of hidden units and method
nh_idx=$((task_id/N1))
m_idx=$((task_id%N1))
# get the array elements
nsample=${nsample_arr[m_idx]}
nh=${nh_arr[nh_idx]}
# echo $nh_idx
# echo $m_idx
# echo "----"
# echo $nh
# echo $method
# echo "----"
# echo $task_id
# echo $task_id2
# echo "----"

if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python3 fewopt.py --methods pmdd --nhidden $nh --nsample $nsample --ntask 50 --tasktype 1v1 --path /users/wybo/Data/results/biasopt/ --nbfactor 10 --nepoch 100
    # srun python3 fewopt.py --methods pmdd --nhidden $nh --nsample $nsample --ntask 2 --tasktype 1v1 --path /users/wybo/Data/results/biasopt/  --nbfactor 1 --nepoch 2 --nperepoch 2
fi

done