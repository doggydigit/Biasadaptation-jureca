#!/bin/bash
ns=( "10")
ds=( "K49")
for n in ${ns[@]}
do
    for d in ${ds[@]}
    do
        bash sbatch_5_inner.sh train_full_dataset.py train_binarymr $n $d train_bmr_full_["$n"]_"$d"
        bash sbatch_4_inner.sh transfer_learn.py transfer_k49_from_emnist $n transfer_bmr_["$n"]_emnist_to_k49
    done
done
