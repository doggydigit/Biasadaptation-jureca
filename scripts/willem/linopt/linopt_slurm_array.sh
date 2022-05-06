#!/bin/bash
#
#SBATCH --job-name=linopt
#SBATCH --output=linopt.txt
#
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-2


conda activate main
readout_arr=("tanh" "sigmoid1" "sigmoid10")


for task_id in {0..2}
do

# get the array elements
readout=${readout_arr[task_id]}

if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then
    srun python3 linopt.py --ntask 47 --tasktype 1vall --readout $readout --path /users/wybo/Data/results/biasopt/ --datasetpath /work/users/wybo/Data/ --nbfactor 10 --nepoch 100
fi

done