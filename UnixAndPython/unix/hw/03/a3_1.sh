#!/usr/bin/env bash
head -c $(((1 + RANDOM)*2 % 65536))  < /dev/urandom > rnd.txt
