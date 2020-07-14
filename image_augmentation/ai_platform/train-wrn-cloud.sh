#!/bin/bash
set -v

DATE=$(date '+%Y%m%d_%H%M%S')
JOB_NAME=cifar10_wrn_40_2_$(date +%Y%m%d_%H%M%S)
JOB_DIR=gs://shrill-anstett-us/${JOB_NAME}

DATA_DIR=gs://shrill-anstett-us/tensorflow_datasets
EPOCHS=50
DATASET=cifar10

REGION=us-central1
SCALE_TIER=basic-gpu

PYTHON_VERSION=3.7
RUNTIME_VERSION=2.1

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
    --scale-tier ${SCALE_TIER} \
    --job-dir "${JOB_DIR}" \
    --stream-logs -- \
    --data-dir "${DATA_DIR}" \
    --epochs ${EPOCHS} \
    --dataset ${DATASET}
