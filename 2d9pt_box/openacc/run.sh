#!/bin/bash

program="openacc"

if [ ! -d log ]; then
    mkdir log > /dev/null
fi

log_file="$(date "+%Y-%m-%d_%H:%M:%S").txt"

if [ ! -f log/$log_file ]; then
    touch log/$log_file > /dev/null
fi

make clean > /dev/null

bsub -I -b -q q_sw_share -n 1 -cgsp 64 -share_size 2048 ./$program | tee log/$log_file
