#!/bin/bash
ns=( "500,500" "250,250" "100,100" "50,50" "25,25" "10,10" "500" "250" "100" "50" "25" "10" "500,100" "250,100" "50,100" "25,100" "10,100" "100,500" "100,250" "100,50" "100,25" "100,10")
ds=( "MNIST" "QMNIST" "EMNIST" "EMNIST_letters" "EMNIST_bymerge" "KMNIST")
for n in ${ns[@]}
do
    for d in ${ds[@]}
    do
        bash sbatch_5_inner.sh train_full_dataset.py train_b_w $n $d train_b_w_full_["$n"]_"$d"
    done
done

ns=( "500,500" "250,250" "100,100" "50,50" "25,25" "10,10" "500" "250" "100" "50" "25" "10" "linear" "500,100" "250,100" "50,100" "25,100" "10,100" "100,500" "100,250" "100,50" "100,25" "100,10")
for n in ${ns[@]}
do
    for d in ${ds[@]}
    do
        bash sbatch_5_inner.sh train_full_dataset.py train_multireadout $n $d train_mr_full_["$n"]_"$d"
        bash sbatch_5_inner.sh train_full_dataset.py train_binarymr $n $d train_bmr_full_["$n"]_"$d"
    done
done
