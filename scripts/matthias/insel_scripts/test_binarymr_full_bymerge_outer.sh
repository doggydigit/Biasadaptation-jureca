#!/bin/bash
# Number of classes must be one less than the actual number
declare -A ds
ds["EMNIST_bymerge"]=46
ns=( "500,500,500" "250,250,250" "100,100,100" "50,50,50" "25,25,25" "10,10,10" "500,500" "250,250" "100,100" "50,50" "25,25" "10,10" "500" "250" "100" "50" "25" "10")
ms=( "binarymr")
for m in ${ms[@]}
do
    for n in ${ns[@]}
    do
        for s in $(seq 0 24)
        do
            bash sbatch_6_inner.sh network_evaluator.py $m $n EMNIST_bymerge $s test_bmr_bymerge_l1o_["$n"]_"$s"
        done
    done
done
