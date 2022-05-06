#!/bin/bash

# Number of classes must be one less than the actual number
declare -A ds
ds["EMNIST_bymerge"]=46
ns=( "500,500" "250,250" "100,100" "50,50" "25,25" "10,10" "100,500" "100,250" "100,50" "100,25" "100,10" "500,100" "250,100" "50,100" "25,100" "10,100")
ws=( "pmdd" "scd")
for w in ${ws[@]}
do
    for n in ${ns[@]}
    do
        for d in "${!ds[@]}"
        do
            for nc in $(seq 0 ${ds[$d]}); do
                bash willem_deepen_l1o_train_bw_inner.sh willem_deepen_l1o_train_bw_"$w"_["$n"]_"$d"_"$nc" $n $d $w $nc
            done
        done
    done
done
