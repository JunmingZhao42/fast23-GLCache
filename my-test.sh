TRACE_DIR=/mnt/disk2/junming/glcache/msr
TRACE=prn_1.IQI.bin.oracleGeneral
CACHE=segcache

(cd prototype && cargo build --release)
./prototype/target/release/bench -t oracleGeneral -i $TRACE_DIR/$TRACE -c 1000 -m $CACHE

# -t: trace type, oracleGeneral is the only supported type
# -i: trace path
# -c: cache size in MB
# -m: cache type, l2cache or segcache
# -r: how often report stats
# -n: hashpower (the estimated number of objects in the cache is 2^n)
