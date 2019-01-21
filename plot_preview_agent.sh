#!/bin/bash

f=`ls -1 nn_wyniki*.csv | tail -n1`
echo "Found '$f'"

while true; do
  cp $f test.csv
  ./plot.py &
  pid=$!
  sleep 5m
  kill $pid
done
