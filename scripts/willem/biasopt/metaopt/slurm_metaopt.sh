#!/bin/bash
#
#SBATCH --job-name=metaopt_many_2
#SBATCH --output=metaopt_many_2.txt
#
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein,hambach
#SBATCH --array 0-1

datapath="/users/wybo/Data/results/biasopt/"
n_process=24
n_offspring=48
n_gen=50

# # readouts
# if [ "0" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo scd --readout sigmoid1 --path $datapath
# fi

# if [ "1" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo scd --readout sigmoid10 --path $datapath
# fi

# if [ "2" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo scd --readout hardtanh --path $datapath
# fi

# if [ "3" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo scd --readout tanh --path $datapath
# fi

# # hidden units
# if [ "4" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 10 --algo scd --readout tanh --path $datapath
# fi

# if [ "5" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 25 --algo scd --readout tanh --path $datapath
# fi

# if [ "6" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 50 --algo scd --readout tanh --path $datapath
# fi

# if [ "7" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo scd --readout tanh --path $datapath
# fi

# if [ "8" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 250 --algo scd --readout tanh --path $datapath
# fi

# if [ "9" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 500 --algo scd --readout tanh --path $datapath
# fi

# algo
# if [ "0" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo pmdd --readout tanh --path $datapath
# fi

# if [ "1" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo rp --readout tanh --path $datapath
# fi

# if [ "2" -eq "$SLURM_ARRAY_TASK_ID" ]
# then
# srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo rpw --readout tanh --path $datapath
# fi

if [ "0" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo br --readout tanh --path $datapath
fi

if [ "1" -eq "$SLURM_ARRAY_TASK_ID" ]
then
srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --algo mr --readout tanh --path $datapath
fi
