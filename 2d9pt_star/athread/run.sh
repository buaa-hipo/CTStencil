#!/bin/bash

# parameters
# DIMT DIMX DIMY NTH
config_0="-1 512 128 -1"
config_1="1 512 128 -1"
# config_1="4 256 256 -1"
config_2="4 256 256 -1"
config_3="4 256 256 -1"

# for POS,AMS comp
config_2="1 512 128 -1"


version=$1

config_name="config_"$version
config=${!config_name}

program="athread_"$version

if [ ! -d log ]; then
    mkdir log > /dev/null
fi

if [ ! -d log/$version ]; then
    mkdir log/$version > /dev/null
fi

log_file="$(date "+%Y-%m-%d_%H:%M:%S").txt"

if [ ! -f log/$version/$log_file ]; then
    touch log/$version/$log_file > /dev/null
fi

make clean > /dev/null

bsub -I -b -q q_sw_expr -n 1 -cgsp 64 -share_size 2048 ./$program $config | tee log/$version/$log_file
