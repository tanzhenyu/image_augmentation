#!/bin/bash

JOB_DIR=/tmp/reduced_cifar10_wrn_40_2
EPOCHS=15
BATCH_SIZE=128
WRN_DEPTH=40
WRN_K=2
DATASET=reduced_cifar10
DATA_DIR=~/tensorflow_datasets/

python3 -m image_augmentation.ai_platform.train_wrn --job-dir=${JOB_DIR} \
    --epochs=${EPOCHS} --batch-size=${BATCH_SIZE} --wrn-depth=${WRN_DEPTH} \
    --wrn-k=${WRN_K} --auto-augment --dataset=${DATASET} --data-dir=${DATA_DIR} \
