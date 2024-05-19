#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../checkpoints

pip install gdown
pip install --upgrade gdown

wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt $SCRIPT_DIR/../checkpoints/v2-1_512-ema-pruned.ckpt

gdown https://storage.googleapis.com/sfr-hive-data-research/checkpoints/hive_rw_condition.ckpt -O $SCRIPT_DIR/../checkpoints/hive_rw_condition.ckpt
gdown https://storage.googleapis.com/sfr-hive-data-research/checkpoints/hive_v2_rw_condition.ckpt -O $SCRIPT_DIR/../checkpoints/hive_v2_rw_condition.ckpt
gdown https://storage.googleapis.com/sfr-hive-data-research/checkpoints/hive_rw.ckpt -O $SCRIPT_DIR/../checkpoints/hive_rw.ckpt
gdown https://storage.googleapis.com/sfr-hive-data-research/checkpoints/hive_v2_rw.ckpt -O $SCRIPT_DIR/../checkpoints/hive_v2_rw.ckpt


#gcloud storage cp gs://sfr-hive-data-research/checkpoints/hive_rw_condition.ckpt $SCRIPT_DIR/../checkpoints/hive_rw_condition.ckpt
#gcloud storage cp gs://sfr-hive-data-research/checkpoints/hive_v2_rw_condition.ckpt $SCRIPT_DIR/../checkpoints/hive_v2_rw_condition.ckpt
#gcloud storage cp gs://sfr-hive-data-research/checkpoints/hive_rw.ckpt $SCRIPT_DIR/../checkpoints/hive_rw.ckpt
#gcloud storage cp gs://sfr-hive-data-research/checkpoints/hive_v2_rw.ckpt $SCRIPT_DIR/../checkpoints/hive_v2_rw.ckpt
