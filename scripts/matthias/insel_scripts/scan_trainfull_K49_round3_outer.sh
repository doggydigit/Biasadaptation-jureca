
#!/bin/bash
ns=( "500,500,500" "100,100,100" "25,25,25" "500,500" "100,100" "25,25" "500" "100" "25")
ds=( "K49")
es=( "early_stopping")

rls=( "0.01" "0.006" "0.003" "0.002")
ls=( "0.0006" "0.0003" "0.0002" "0.0001" "0.00006" "0.00003" "0.0002")
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

