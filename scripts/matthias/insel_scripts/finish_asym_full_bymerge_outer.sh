#!/bin/bash
declare -A ds
ds["EMNIST_bymerge"]=46
ns=( "25,25" "25,50" "25,100" "25,250" "25,500")
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
