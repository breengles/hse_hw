#!/usr/bin/env bash
head -c $((10485760 / 5))  < /dev/urandom > data 
