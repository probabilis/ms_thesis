#!/bin/bash

for i in {00..01}; do
  python preprocessing.py \
    --lcp "data/expdata/data_${i}/lcp" \
    --rcp "data/expdata/data_${i}/rcp" \
    --save "data/expdata/data_${i}/csv/"
done
