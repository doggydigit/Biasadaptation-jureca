#!/bin/bash
#
#SBATCH --job-name=metaopt_test
#SBATCH --output=metaopt_test.txt
#
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000M
#SBATCH --array 0-2

datapath="/users/wybo/Data/results/biasopt/"
n_process=2
n_offspring=2
n_gen=1

# readouts
if [ "0" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo scd --readout sigmoid1 --path $datapath
fi

if [ "1" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo scd --readout sigmoid10 --path $datapath
fi