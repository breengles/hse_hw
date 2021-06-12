#!/usr/bin/env bash

export OMP_NUM_THREADS=1
t=2000000
bu=250000
ba=256

cfg=configs/2v2_1.json
for reg in 0 1e-6
do
    ./train.py -t $t --buffer $bu --batch $ba --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg $reg --env-config $cfg >/dev/null &
    ./train.py -t $t --buffer $bu --batch $ba --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg $reg -tp --env-config $cfg >/dev/null&
    ./train.py -t $t --buffer $bu --batch $ba --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg $reg -tp -ndr --env-config $cfg >/dev/null&
done
