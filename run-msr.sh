#!/bin/bash

TRACE_DIR=/mnt/disk2/junming/glcache/msr
CACHE=l2cache

# just make sure
(cd prototype && cargo build --release)

for i in {1..3}
do
    echo $i
    # Loop through all traces in the folder
    for file in $TRACE_DIR/*
    do
        if [[ "$file" == *.bin.* ]]; then
            echo $file
            ./prototype/target/release/bench -t oracleGeneral -i "$file" -c 2000 -n 26 -m $CACHE > "$file-lr-$i".out
            mv "$file-lr-$i".out msr-stats/
            sleep 3
        fi
    done
    rm /mnt/pmem1.0/junming/*
    rm /mnt/pmem0.0/junming/*
done
