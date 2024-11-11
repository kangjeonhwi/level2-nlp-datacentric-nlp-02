#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python ./Relabeling/clustering.py \
    --data_path "./data/train.csv" \
    --model_name "team-lucid/deberta-v3-xlarge-korean" \
    --output_dir "./output/clustering"