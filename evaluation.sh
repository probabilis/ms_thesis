#!/bin/bash
DATASET="data_00"

echo "Running GD Nesterov optimization method for experimental image data for $DATASET."

#comment: data_0x for dataset nr + looping over different files in.

for i in {1..5}; do
  python evaluation_add.py \
    --dataset $DATASET \
    --recording "00${i}"
done