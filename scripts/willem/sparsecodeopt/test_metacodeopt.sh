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
#SBATCH --array 0-5

datapath="/users/wybo/Data/results/biasopt/"
datasetpath="/work/users/wybo/Data/code_matrices/"
n_process=1
n_offspring=1
n_gen=1

# readouts
python3 metacodeopt.py --nprocesses $n_process --noffspring $n_offspring --ngen $n_gen --nhidden 100 --readout sigmoid1 --path $datapath --datasetpath $datasetpath
