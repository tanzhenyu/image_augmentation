#!/bin/bash
set -v

DATE=$(date '+%Y%m%d_%H%M%S')
SRC_VERSION=v`git log -1 --pretty=format:%h`
JOB_NAME=cifar10_wrn_28_10_${SRC_VERSION}_$(date +%Y%m%d_%H%M%S)
JOB_DIR=gs://shrill-anstett-us/${JOB_NAME}

DATA_DIR=gs://shrill-anstett-us/tensorflow_datasets
EPOCHS=200
DATASET=cifar10

OPT=sgd
INIT_LR=0.1
DECAY=0.0005
DROP_LR_BY=0.2
DROP_LR_EVERY=60

WRN_DEPTH=28
WRN_K=10

REGION=us-central1
SCALE_TIER=basic-gpu

# PYTHON_VERSION=3.7
# RUNTIME_VERSION=2.1

# gcloud ai-platform jobs submit training "${JOB_NAME}" \
#   --package-path image_augmentation \
#   --module-name image_augmentation.ai_platform.train_wrn \
#   --region ${REGION} \
#   --python-version ${PYTHON_VERSION} \
#   --runtime-version ${RUNTIME_VERSION} \
#   --scale-tier ${SCALE_TIER} \
#   --job-dir "${JOB_DIR}" \
#   --stream-logs -- \
#   --data-dir "${DATA_DIR}" \
#   --epochs ${EPOCHS} \
#   --dataset ${DATASET}

IMAGE_URI=gcr.io/ml-dl-tfrc-tpu/image_augmentation/wrn_trainer

gcloud ai-platform jobs submit training "${JOB_NAME}" \
    --master-image-uri ${IMAGE_URI} \
    --region ${REGION} \
    --scale-tier custom \
    --master-machine-type n1-highmem-2 \
    --master-accelerator count=1,type=nvidia-tesla-t4 \
    --job-dir "${JOB_DIR}" \
    --stream-logs -- \
    --data-dir ${DATA_DIR} --epochs ${EPOCHS} --dataset ${DATASET} \
    --optimizer ${OPT} --init-lr ${INIT_LR} --drop-lr-by ${DROP_LR_BY} \
    --weight-decay ${DECAY} --drop-lr-every ${DROP_LR_EVERY} \
    --wrn-depth ${WRN_DEPTH} --wrn-k ${WRN_K} --no-cutout
