#!/bin/bash

JOB_DIR=/tmp/demo_experiment

mkdir ${JOB_DIR}

EPOCHS=5
BATCH_SIZE=128
WRN_DEPTH=28
WRN_K=10
DATASET=reduced_cifar10

PADDING_MODE=reflect
NORMALIZATION=rgb_normalization
OPTIMIZER=sgd
INIT_LR=0.1
DROP_LR_BY=0.2
L2_REG=0.0005

python3 -m image_augmentation.ai_platform.train_wrn --job-dir=${JOB_DIR} \
    --epochs=${EPOCHS} --batch-size=${BATCH_SIZE} --wrn-depth=${WRN_DEPTH} \
    --wrn-k=${WRN_K} --no-cutout --dataset=${DATASET} --no-cutout \
    --padding-mode ${PADDING_MODE} --normalization ${NORMALIZATION} \
    --optimizer ${OPTIMIZER} --init-lr ${INIT_LR} --l2-reg ${L2_REG} \
    --drop-lr-by ${DROP_LR_BY} --drop-lr-every 60 --drop-lr-every 120 --drop-lr-every 160 \
    --sgd-nesterov

