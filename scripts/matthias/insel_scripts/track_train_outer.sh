#!/bin/bash

for nc in $(seq 0 46); do
    bash sbatch_4_inner.sh train_full_dataset.py track_training_1  $nc track_training_1_"$nc"
done
