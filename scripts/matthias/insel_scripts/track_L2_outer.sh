#!/bin/bash

declare -A ds
ds["EMNIST_bymerge"]=46
ds["K49"]=48
ds["CIFAR100"]=99

for d in "${!ds[@]}"; do
    for s in $(seq 0 24); do
            bash sbatch_5_inner.sh train_full_dataset.py track_epoch_fullbw  $d $s track_epoch_fullbw_"$d"_"$s"
            bash sbatch_5_inner.sh train_full_dataset.py track_training_fullbw  $d $s track_training_fullbw_"$d"_"$s"
    done
    for nc in $(seq 0 ${ds[$d]}); do
        for s in $(seq 0 2); do
            bash sbatch_6_inner.sh train_full_dataset.py track_epoch_sg  $d $s $nc track_epoch_sg_"d"_"$nc"_"$s"
            bash sbatch_6_inner.sh train_full_dataset.py track_training_sg  $d $s $nc track_training_sg_"d"_"$nc"_"$s"
        done
    done
done
