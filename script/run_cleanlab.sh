#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python ./Relabeling/cleanlab.py \
    --embedding_model_type 'klue/roberta-large' \
    --predict_model_type 'klue/roberta-large' \
    --clf_type "LGBM" \
    --data_path "../data/train.csv" \
    --output_dir "./output/cleanlab"


