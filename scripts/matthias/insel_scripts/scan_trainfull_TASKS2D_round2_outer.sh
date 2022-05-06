
#!/bin/bash
ns=( "100,100,100,100" "50,50,50,50" "25,25,25,25" "100,100,100" "50,50,50" "25,25,25" "100,100" "50,50" "25,25" "100" "50" "25")
ds=( "TASKS2D")
es=( "early_stopping")
bls=( "0.06" "0.03" "0.02" "0.01" "0.006" "0.003" "0.002")
ls=( "0.06" "0.03" "0.02" "0.01" "0.006" "0.003" "0.002")
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

