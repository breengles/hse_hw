#!/usr/bin/env bash

count_file=0
files=$(find . -maxdepth 1 -type f -name "*.c" -print0 | xargs --null -n 1 basename -z | tr "\000" "\n")
headers=($(find . -maxdepth 1 -type f -name "*.h" -print0 | xargs --null -n 1 basename -z | tr "\000" "\n"))

num_files=${#files[@]}
num_headers=${#headers[@]}
for (( i=1; i<$num_files; i++ ))
do
    file="${files[$i]}"
    count=0
    incl=("$(tr "\n" "\000" < $file | sed -e "s/\/\*.*\*\///g" | tr "\000" "\n" | sed -e "s/\/\/.*//g" -e "s/.*\"#include\s*.*\"//g" -Ee "/^\s*#include\s*(<|\")(.*).h(\"|>)$/!d" -Ee "s/\s*#include\s*<(.*)\.h>/\1/g" -Ee "s/\s*#include\s*\"(.*)\.h\"/\1/g" -Ee "/^\s*#include.*$/d")")
    num_incl=${#incl[@]}
    for (( j=1; j<$num_incl; j++ ))
    do
        hin="${incl[$j]}"
        for (( k=1; k<$num_headers; k++ ))
        do
            head=${headers[$k]}
            if [[ $hin == "$head" ]]
            then
                count++
            fi
        done
    done
    if [ $count -eq ${#incl[@]} ]
    then
        count_file++
    fi
done
if [[ $count_file -eq ${#files[@]} ]]
then
    echo "1"
else
    echo "0"
fi