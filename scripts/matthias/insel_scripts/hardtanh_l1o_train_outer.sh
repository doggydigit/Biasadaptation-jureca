#!/bin/bash
# Number of classes must be one less than the actual number
declare -A ds
ds["EMNIST_willem"]=46
ns=( "500" "250" "100" "50" "25" "10")
for n in ${ns[@]}
do
    for d in "${!ds[@]}"
    do
        for nc in $(seq 0 ${ds[$d]}); do
            bash sbatch_6_inner.sh willem_extra.py train_hardtanh_bw $n $d $nc l1o_train_hardtanh_bw_["$n"]_"$d"_"$nc"
        done
    done
done
