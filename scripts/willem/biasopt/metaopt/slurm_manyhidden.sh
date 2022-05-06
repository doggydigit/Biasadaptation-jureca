#!/bin/bash
#
#SBATCH --job-name=metaopt_hidden
#SBATCH --output=metaopt_hidden.txt
#
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=blaustein

datapath="/users/wybo/Data/results/biasopt/"
n_process=48
n_offspring=48
n_gen=50


srun python3 metaopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 10 25 50 100 250 500 --algo scd --readout tanh --path $datapath