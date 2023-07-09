#!/bin/bash

# Make data folder relative to script location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../data

# Download HIVE training data
gsutil cp -r gs://sfr-hive-data-research/data/training $SCRIPT_DIR/../data/training

# Download HIVE evaluation data

gcloud storage cp gs://sfr-hive-data-research/data/test.jsonl $SCRIPT_DIR/../data/test.jsonl
gsutil cp -r gs://sfr-hive-data-research/data/evaluation $SCRIPT_DIR/../data/