#!/usr/bin/env bash
head -c $(shuf -i 0-65536 -n 1)  < /dev/urandom > rnd.txt
