#!/usr/bin/env bash

export OMP_NUM_THREADS=1
t=2000000
bu=250000
ba=256

# ./train.py -t 10000000 --buffer 250000 --batch 256 --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg 1e-6 >/dev/null &
# ./train.py -t 10000000 --buffer 250000 --batch 2048 --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg 1e-6 >/dev/null &
# ./train.py -t 10000000 --buffer 250000 --batch 256 --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg 1e-6 -tp >/dev/null &
# ./train.py -t 10000000 --buffer 250000 --batch 256 --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg 1e-6 -tp -ndr >/dev/null &
# ./train.py -t 10000000 --buffer 250000 --batch 256 --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg 0 -tp >/dev/null &

cfg=configs/1v1.json
for reg in 0 1e-6
do
    ./train.py -t $t --buffer $bu --batch $ba --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg $reg --env-config $cfg >/dev/null &
    ./train.py -t $t --buffer $bu --batch $ba --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg $reg -tp --env-config $cfg >/dev/null &
    ./train.py -t $t --buffer $bu --batch $ba --seed 42 --sigma-max 1 --sigma-min 0.1 --saverate 20000 --actor-lr 3e-4 --critic-lr 3e-4 --actor-reg $reg -tp -ndr --env-config $cfg >/dev/null &
done
