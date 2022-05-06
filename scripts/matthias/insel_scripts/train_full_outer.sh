#!/bin/bash
ns=( "500,500,500" "100,100,100" "25,25,25" "500,500" "100,100" "25,25" "500" "100" "25")
ds=( "EMNIST_bymerge" "CIFAR100")
for n in ${ns[@]}
do
    for d in ${ds[@]}
    do
        bash sbatch_5_inner.sh train_full_dataset.py train_b_w $n $d train_b_w_full_["$n"]_"$d"
        bash sbatch_5_inner.sh train_full_dataset.py train_g_bw $n $d train_g_bw_full_["$n"]_"$d"
        bash sbatch_5_inner.sh train_full_dataset.py train_bg_w $n $d train_bg_w_full_["$n"]_"$d"
        bash sbatch_5_inner.sh train_full_dataset.py train_binarymr $n $d train_bmr_full_["$n"]_"$d"
    done
done
