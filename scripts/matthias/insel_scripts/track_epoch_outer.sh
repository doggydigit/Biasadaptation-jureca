#!/bin/bash

for s in $(seq 0 24); do
    for nc in $(seq 0 46); do
        bash sbatch_5_inner.sh train_full_dataset.py track_epoch_1  $nc $s track_epoch_1_"$nc"_"$s"
    done
done

for s in $(seq 0 46); do
    bash sbatch_4_inner.sh train_full_dataset.py track_epoch_full  $s track_epoch_full_"$s"
done
