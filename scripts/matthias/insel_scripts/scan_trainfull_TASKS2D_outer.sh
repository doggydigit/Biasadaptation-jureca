
#!/bin/bash
ns=( "100,100,100,100" "50,50,50,50" "25,25,25,25" "100,100,100" "50,50,50" "25,25,25" "100,100" "50,50" "25,25" "100" "50" "25")
ds=( "TASKS2D")
es=( "early_stopping")
bls=( "0.1" "0.01" "0.001" "0.0001" "0.00001" "0.000001")
ls=( "0.1" "0.01" "0.001" "0.0001" "0.00001" "0.000001")
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

