#!/bin/bash

d=`find . -name '*_*_*_*' -type d | head -n1`
echo "Found '$d'"

while true; do
  ./plot.py -v 3 $d/learning_results.csv &
  pid=$!
  sleep 5m
  kill $pid
done
