#!/bin/bash

# Number of classes mst be one less than the actual number
declare -A ds
ds["EMNIST_bymerge"]=46
ds["K49"]=48
ds["CIFAR100"]=99
ns=( "500,500,500" "100,100,100" "25,25,25" "500,500" "100,100" "25,25" "500" "100" "25")
for n in ${ns[@]}
do
    for d in "${!ds[@]}"
    do
        for nc in $(seq 0 ${ds[$d]}); do
            bash sbatch_6_inner.sh train_full_dataset.py train_sg $n $d $nc train_sg_full_["$n"]_"$d"_"$nc"
        done
    done
done
