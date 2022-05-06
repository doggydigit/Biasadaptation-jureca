#!/bin/bash
# Number of classes must be one less than the actual number
declare -A ds
ds["EMNIST_bymerge"]=46
ds["CIFAR100"]=99
ns=( "100,100,100" "100,100" "100")
ps=( "train_b_w" "train_bmr")
for p in ${ps[@]}
do
    for n in ${ns[@]}
    do
        for d in "${!ds[@]}"
        do
            for nc in $(seq 0 ${ds[$d]})
            do
                bash sbatch_6_inner.sh leave1out.py $p $n $d $nc l1o_"$p"_["$n"]_"$d"_"$nc"
            done
        done
    done
done
