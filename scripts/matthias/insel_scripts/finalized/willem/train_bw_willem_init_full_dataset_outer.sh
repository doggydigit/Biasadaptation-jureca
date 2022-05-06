#!/bin/bash
ns=( "500" "250" "100" "50" "25" "10")
ds=( "EMNIST_bymerge")
ws=( "pmdd" "scd")
for n in ${ns[@]}
do
    for d in ${ds[@]}
    do
        for w in ${ws[@]}
        do
            bash train_bw_willem_init_full_dataset_inner.sh willeminit_trainbw_full_"$w"["$n"]_"$d" $n $d $w
        done
    done
done

