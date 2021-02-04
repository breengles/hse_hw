#!/usr/bin/env bash
cat ip.txt | xargs -n1 ping -c 2 > res.txt 2> err.txt
