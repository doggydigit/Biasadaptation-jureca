#!/bin/bash
ns=( "500,500" "100,100" "10,10" "500" "100" "10")
ds=( "MNIST" "QMNIST" "EMNIST" "EMNIST_letters" "EMNIST_bymerge" "KMNIST")
es=( "not_early_stopping" "early_stopping")
ls=( "0.1" "0.03" "0.01" "0.003" "0.001" "0.0003" "0.0001" "0.00003" "0.00001")
ps=( "train_bw_lr" "train_mr_lr")
for n in ${ns[@]}
do
    for p in ${ps[@]}
    do
        for d in ${ds[@]}
        do
            for e in ${es[@]}
            do
                for l in ${ls[@]}
                do
                    bash sbatch_7_inner.sh scan_params.py $p $n $d $e $l scan_"$p"_["$n"]_"$d"_"$e"_"$l"
                done
            done
        done
    done
done

es=( "early_stopping")
ps=( "train_bw_hardtanh_lr")
for n in ${ns[@]}
do
    for p in ${ps[@]}
    do
        for d in ${ds[@]}
        do
            for e in ${es[@]}
            do
                for l in ${ls[@]}
                do
                    bash sbatch_7_inner.sh scan_params.py $p $n $d $e $l scan_"$p"_["$n"]_"$d"_"$e"_"$l"
                done
            done
        done
    done
done

bls=( "0.5" "0.3" "0.2" "0.1" "0.06" "0.03" "0.01" "0.003" "0.001" "0.0003" "0.0001" "0.00003")
ls=( "0.01" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0001" "0.00003")
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
                done
            done
        done
    done
done

es=( "early_stopping")
rls=( "0.3" "0.1" "0.03" "0.01" "0.003" "0.001" "0.0003" "0.0001" "0.00003")
ls=( "0.1" "0.03" "0.01" "0.003" "0.001" "0.0003" "0.0001" "0.00003" "0.00001")
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

ns=( "500,500,500" "100,100,100" "10,10,10")
ds=( "QMNIST" "EMNIST_letters" "EMNIST_bymerge" "KMNIST")
es=( "early_stopping")
bls=( "0.5" "0.3" "0.2" "0.1" "0.06" "0.03" "0.01" "0.003" "0.001")
ls=( "0.01" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0001")
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
                done
            done
        done
    done
done
