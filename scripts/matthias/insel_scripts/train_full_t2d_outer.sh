#!/bin/bash
ns=( "500,500,500,500" "250,250,250,250" "100,100,100,100" "50,50,50,50" "25,25,25,25" "500,500,500" "250,250,250" "100,100,100" "50,50,50" "25,25,25" "500,500" "250,250" "100,100" "50,50" "25,25" "500" "250" "100" "50" "25")
ds=( "TASKS2D")
for n in ${ns[@]}
do
    for d in ${ds[@]}
    do
        bash sbatch_5_inner.sh train_full_dataset.py train_b_w $n $d train_b_w_full_["$n"]_"$d"
        bash sbatch_5_inner.sh train_full_dataset.py train_binarymr $n $d train_bmr_full_["$n"]_"$d"
    done
done
