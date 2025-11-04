#!/bin/bash
DATASET="data_01"

% comment: 

for i in {5..6}; do
  python evaluation_add.py \
    --dataset $DATASET \
    --recording "00${i}"
done