#!/bin/bash
# Number of classes must be one less than the actual number
declare -A ds
ds["MNIST"]=9
ds["QMNIST"]=9
ds["EMNIST"]=46
ds["EMNIST_letters"]=25
ds["EMNIST_bymerge"]=46
ds["KMNIST"]=9
ns=( "500,500,500" "250,250,250" "100,100,100" "50,50,50" "25,25,25" "10,10,10")
ps=( "transfer_b" "transfer_bmr")
for p in ${ps[@]}
do
    for n in ${ns[@]}
    do
        for d in "${!ds[@]}"
        do
            for nc in $(seq 0 ${ds[$d]})
            do
                bash sbatch_7_inner.sh leave1out.py $p $n $d early_stopping $nc l1o_"$p"_["$n"]_"$d"_"$nc"_"$l"
            done
        done
    done
done
