#!/bin/bash
ns=( "500,500,500" "250,250,250" "100,100,100" "50,50,50" "25,25,25" "10,10,10" "500,500" "250,250" "100,100" "50,50" "25,25" "10,10" "500" "250" "100" "50" "25" "10")
ds=( "K49")
for n in ${ns[@]}
do
    for d in ${ds[@]}
    do
        bash sbatch_5_inner.sh train_full_dataset.py train_binarymr $n $d train_bmr_full_["$n"]_"$d"
        bash sbatch_4_inner.sh transfer_learn.py transfer_k49_from_emnist $n transfer_bmr_["$n"]_emnist_to_k49
    done
done
