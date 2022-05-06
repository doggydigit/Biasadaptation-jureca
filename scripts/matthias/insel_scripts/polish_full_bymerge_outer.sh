#!/bin/bash
declare -A ds
ds["EMNIST_bymerge"]=46
ns=( "500,500,500" "250,250,250" "100,100,100" "50,50,50" "25,25,25" "10,10,10" "500,500" "250,250" "100,100" "50,50" "25,25" "10,10" "500" "250" "100" "50" "25" "10")
for n in ${ns[@]}
do
    for d in "${!ds[@]}"
    do
        for nc in $(seq 0 ${ds[$d]})
        do
            bash sbatch_6_inner.sh train_full_dataset.py polish_b $n $d $nc polish_b_full_["$n"]_"$d"_"$nc"
        done
    done
done
