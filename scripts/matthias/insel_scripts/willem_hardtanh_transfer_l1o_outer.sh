#!/bin/bash
# Number of classes must be one less than the actual number
declare -A ds
ds["EMNIST_bymerge"]=46
ns=( "500" "250" "100" "50" "25" "10")
wts=( "pmdd" "scd")
for n in ${ns[@]}
do
    for d in "${!ds[@]}"
    do
        for nc in $(seq 0 ${ds[$d]}); do
            for wt in ${wts[@]}
            do
                bash sbatch_7_inner.sh willem_extra.py transfer_b_willem_hardtanh $n $d $nc $wt l1o_transfer_b_willem_hardtanh_["$n"]_"$d"_"$nc"_"$wt"
            done
        done
    done
done
