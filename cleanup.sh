#!/usr/bin/bash

# run with the name of the condor job

cat $1_*.out > $1.dat
gzip $1.dat

for ext in err log out
do
    rm -f $1_*.$ext
done

echo Created $1.dat.gz
