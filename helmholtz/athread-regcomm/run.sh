#!/bin/bash

# parameters
# DIMT DIMX DIMY DIMZ NTH
config_0="-1 64 8 4 -1"
config_1="1 128 4 64 -1"
config_1="2 32 8 128 -1"
config_2="2 32 8 128 -1"
# config_3="2 32 8 128 -1"
config_3="2 32 8 128 -1"
config_4="2 32 8 128 -1"
# config_5="2 32 8 128 2"
config_5="2 32 8 128 2"

# for regcomm exp
# config_3="1 4 2 128 -1"
# config_3="1 4 4 128 -1"
# config_3="1 4 8 128 -1"
# config_3="1 8 2 128 -1"
# config_3="1 8 4 128 -1"
# config_3="1 4 16 128 -1"
# config_3="1 8 8 128 -1"
# config_3="1 16 4 128 -1"
# config_3="1 4 32 128 -1"
# config_3="1 8 16 128 -1"
# config_3="1 16 8 128 -1"
# config_3="1 32 4 128 -1"
# config_3="1 8 32 128 -1"
# config_3="1 16 16 128 -1"
# config_3="1 32 8 128 -1"

# config_5="1 4 2 128 2"
# config_5="1 4 4 128 2"
# config_5="1 4 8 128 2"
# config_5="1 8 2 128 2"
# config_5="1 8 4 128 2"
# config_5="1 4 16 128 2"
# config_5="1 8 8 128 2"
# config_5="1 16 4 128 2"
# config_5="1 4 32 128 2"
# config_5="1 8 16 128 2"
# config_5="1 16 8 128 2"
# config_5="1 32 4 128 2"
# config_5="1 8 32 128 2"
# config_5="1 16 16 128 2"
# config_5="1 32 8 128 2"

# for regcomm exp with DIMT
# config_3="2 32 8 128 -1"

# config_5="2 32 8 128 2"


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
