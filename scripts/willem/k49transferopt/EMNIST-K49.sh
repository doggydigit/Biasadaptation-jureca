#!/bin/bash
#
#SBATCH --job-name=EMNIST-K49_transfer
#SBATCH --output=EMNIST-K49_transfer.txt
#
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-30


nh_arr=(10 25 50 100 250 500)
npercycle=6

for task_id in {0..30}
do

nh_idx=$((task_id%npercycle))
# get the array elements
nh=${nh_arr[nh_idx]}

if [ "$task_id" -ge "0" ] && [ "$task_id" -lt "6" ] && [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python transferopt.py --algo pmdd --nhidden $nh --ntask 49 --tasktype 1vall --readout tanh --weightdatasets EMNIST --m1datasets K49 --p1datasets K49 --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/ --nbfactor 10 --nepoch 100
fi

if [ "$task_id" -ge "$npercycle" ] && [ "$task_id" -lt "$((2*npercycle))" ] && [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python transferopt.py --algo pmdd --nhidden $nh --ntask 49 --tasktype 1vall --readout tanh --weightdatasets K49 --m1datasets K49 --p1datasets K49 --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/ --nbfactor 10 --nepoch 100
fi

if [ "$task_id" -ge "$((2*npercycle))" ] && [ "$task_id" -lt "$((3*npercycle))" ] && [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python transferopt.py --algo pmdd --nhidden $nh --ntask 49 --tasktype 1vall --readout tanh --weightdatasets EMNIST K49 --m1datasets K49 --p1datasets K49 --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/ --nbfactor 10 --nepoch 100
fi

if [ "$task_id" -ge "$((3*npercycle))" ] && [ "$task_id" -lt "$((4*npercycle))" ] && [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python transferopt.py --algo pmdd --nhidden $nh --ntask 47 --tasktype 1vall --readout tanh --weightdatasets EMNIST K49 --m1datasets EMNIST --p1datasets EMNIST --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/ --nbfactor 10 --nepoch 100
fi

if [ "$task_id" -ge "$((4*npercycle))" ] && [ "$task_id" -lt "$((5*npercycle))" ] && [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python transferopt.py --algo pmdd --nhidden $nh --ntask 47 --tasktype 1vall --readout tanh --weightdatasets K49 --m1datasets EMNIST --p1datasets EMNIST --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/ --nbfactor 10 --nepoch 100
fi


done