#!/bin/bash

NOPFS_ROOT=$HOME/NoPFS/
OUTPUT_DIR=$HOME/NoPFS/runs/logs-$(date +%Y-%m-%d-%H-%M-%S)

DATA_DIR=/home/DATA/ImageNet_raw
SEED=42

# should be on local disk or high performance /scratch
CACHE_DIR=/home/SSD/nopfs_cache

mpirun -np 2  \
python ${NOPFS_ROOT}/benchmark/resnet50.py \
  --output-dir=${OUTPUT_DIR} \
  --seed ${SEED} \
  --no-eval  \
  --save-stats \
  --data-dir ${DATA_DIR} \
  --cache-dir ${CACHE_DIR} \
  --dataset imagenet \
  --drop-last \
  --batch-size 64 \
  --epochs 1 \
  --dist
