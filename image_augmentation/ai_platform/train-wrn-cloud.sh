#!/bin/bash
set -v

echo "Training Cloud ML model"

DATE=$(date '+%Y%m%d_%H%M%S')
JOB_NAME=cifar10_wrn_40_2_$(date +%Y%m%d_%H%M%S)
JOB_DIR=gs://shrill-anstett-us/${JOB_NAME}

DATA_DIR=gs://shrill-anstett-us/tensorflow_datasets
EPOCHS=50
DATASET=cifar10

REGION=us-central1
PYTHON_VERSION=3.7
RUNTIME_VERSION=2.1
SCALE_TIER=basic-gpu

gcloud ai-platform jobs submit training "${JOB_NAME}" \
  --package-path image_augmentation \
  --module-name image_augmentation.ai_platform.train_wrn \
  --region ${REGION} \
  --python-version ${PYTHON_VERSION} \
  --runtime-version ${RUNTIME_VERSION} \
  --scale-tier ${SCALE_TIER} \
  --job-dir "${JOB_DIR}" \
  --stream-logs -- \
  --data-dir "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --dataset ${DATASET}
