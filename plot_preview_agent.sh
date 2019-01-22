#!/bin/bash

d=`find . -name '*_*_*_*' -type d | head -n1`
echo "Found '$d'"

while true; do
  cp $d/learning_results.csv test.csv
  ./plot.py -v 2 &
  pid=$!
  sleep 5m
  kill $pid
done
