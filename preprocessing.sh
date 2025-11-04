#!/bin/bash

for i in {00..03}; do
  python preprocessing.py \
    --lcp "data/input/data_${i}/lcp" \
    --rcp "data/input/data_${i}/rcp" \
    --save "data/input/data_${i}/csv/"
done
