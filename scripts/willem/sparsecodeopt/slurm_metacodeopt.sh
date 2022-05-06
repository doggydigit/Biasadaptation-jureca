#!/bin/bash
#
#SBATCH --job-name=metacodeopt_many_2
#SBATCH --output=metacodeopt_many_2.txt
#
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-5

datapath="/users/wybo/Data/results/biasopt/"
datasetpath="/work/users/wybo/Data/code_matrices/"
n_process=24
n_offspring=48
n_gen=50

# readouts
if [ "0" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metacodeopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --readout sigmoid1 --path $datapath --datasetpath $datasetpath
fi

if [ "1" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metacodeopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --readout sigmoid10 --path $datapath --datasetpath $datasetpath
fi

if [ "2" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metacodeopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 ---readout tanh --path $datapath --datasetpath $datasetpath
fi

# hidden units
if [ "3" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metacodeopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 25 --readout tanh --path $datapath --datasetpath $datasetpath
fi

if [ "4" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metacodeopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --readout tanh --path $datapath --datasetpath $datasetpath
fi

if [ "5" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metacodeopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 250  --readout tanh --path $datapath --datasetpath $datasetpath
fi


