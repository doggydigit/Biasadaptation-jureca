#!/bin/bash
ns=( "500,500,500" "100,100,100" "25,25,25" "10,10,10" "500,500" "100,100" "25,25" "10,10" "500" "100" "25" "10")
ds=( "CIFAR100" "EMNIST_bymerge")
es=( "early_stopping")
bls=( "0.3" "0.2" "0.1" "0.06" "0.03" "0.02" "0.01" "0.006" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0001" "0.00003")
ls=( "0.03" "0.01" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0002" "0.0001" "0.00006" "0.00003" "0.00002" "0.00001")
for n in ${ns[@]}
do
    for bl in ${bls[@]}
    do
        for d in ${ds[@]}
        do
            for e in ${es[@]}
            do
                for l in ${ls[@]}
                do
                    bash sbatch_8_inner.sh scan_params.py train_b_w_lr $n $d $e $l $bl scan_train_b_w_["$n"]_"$d"_"$l"_"$bl"
                    bash sbatch_8_inner.sh scan_params.py train_g_bw_lr $n $d $e $l $bl scan_train_g_bw_["$n"]_"$d"_"$l"_"$bl"
                    bash sbatch_8_inner.sh scan_params.py train_bg_w_lr $n $d $e $l $bl scan_train_bg_w_["$n"]_"$d"_"$l"_"$bl"
                done
            done
        done
    done
done

rls=( "0.1" "0.03" "0.01" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0002" "0.0001" "0.00003")
ls=( "0.1" "0.03" "0.01" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0002" "0.0001" "0.00006" "0.00003")
for n in ${ns[@]}
do
    for rl in ${rls[@]}
    do
        for d in ${ds[@]}
        do
            for e in ${es[@]}
            do
                for l in ${ls[@]}
                do
                    bash sbatch_8_inner.sh scan_params.py train_bmr_lr $n $d $e $l $rl scan_train_bmr_["$n"]_"$d"_"$l"_"$rl"
                done
            done
        done
    done
done

