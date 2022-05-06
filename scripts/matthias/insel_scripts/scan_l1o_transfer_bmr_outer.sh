#!/bin/bash

# Number of classes must be one less than the actual number
declare -A ds
ds["MNIST"]=9
ds["QMNIST"]=9
ds["EMNIST"]=46
ds["EMNIST_letters"]=25
ds["EMNIST_bymerge"]=46
ds["KMNIST"]=9
ns=( "500,500" "100,100" "10,10" "500" "100" "10")
ps=( "transfer_bmr_l1o_lr")
ls=( "0.5" "0.4" "0.3" "0.2" "0.1" "0.06" "0.03" "0.02" "0.01" "0.006" "0.003" "0.001" "0.0003" "0.0001" "0.00003")
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
                    bash sbatch_8_inner.sh scan_params.py $p $n $d early_stopping $l $nc "$p"_["$n"]_"$d"_"$nc"_"$l"
                done
            done
        done
    done
done

