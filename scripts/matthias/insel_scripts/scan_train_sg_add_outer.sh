#!/bin/bash

# Number of classes must be one less than the actual number
declare -A ds
ds["EMNIST_bymerge"]=46
ds["K49"]=48
ds["CIFAR100"]=99
ns=( "100,100,100" "100,100" "100")
ps=( "train_sg_lr")
ls=( "0.00002" "0.000006")
for n in ${ns[@]}
do
    for d in "${!ds[@]}"
    do
        for nc in $(seq 0 ${ds[$d]})
        do
            for p in ${ps[@]}
            do
                for l in ${ls[@]}
                do
                    bash sbatch_8_inner.sh scan_params.py $p $n $d early_stopping $l $nc scan_"$p"_["$n"]_"$d"_"$nc"_"$l"
                done
            done
        done
    done
done
