#!/bin/bash

# Number of classes must be one less than the actual number
declare -A ds
ds["MNIST"]=9
ds["QMNIST"]=9
ds["EMNIST"]=46
ds["EMNIST_letters"]=25
ds["EMNIST_bymerge"]=46
ds["KMNIST"]=9
ns=( "500,500" "250,250" "100,100" "50,50" "25,25" "10,10" "500" "250" "100" "50" "25" "10")
for n in ${ns[@]}
do
    for d in "${!ds[@]}"
    do
        for nc in $(seq 0 ${ds[$d]}); do
            bash sbatch_6_inner.sh train_full_dataset.py train_sg $n $d $nc train_sg_full_["$n"]_"$d"_"$nc"
        done
    done
done

