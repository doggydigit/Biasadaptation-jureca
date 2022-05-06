#!/bin/bash

# Number of classes must be one less than the actual number
declare -A ds
ds["EMNIST_bymerge"]=46
ns=( "500,500,500" "100,100,100" "25,25,25" "10,10,10" "500,500" "100,100" "25,25" "10,10" "500" "100" "25" "10")
ps=( "train_sg_lr_bigbatch")
ls=( "0.1" "0.03" "0.02" "0.01" "0.006" "0.003" "0.002" "0.001" "0.0003" "0.0001" "0.00003" "0.00001")
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

