#!/bin/bash
ns=( "10" "10,10" "100" "100,100" "500" "500,500")
ds=( "MNIST" "QMNIST" "EMNIST" "EMNIST_letters" "EMNIST_bymerge")
es=( "early_stopping" "not_early_stopping")
for n in ${ns[@]}
do
    for d in ${ds[@]}
    do
        for e in ${es[@]}
        do
            bash loop_sbatches_outer.sh logfilename_["$n"]_"$d"_"$e" $n $d $e
        done
    done
done
