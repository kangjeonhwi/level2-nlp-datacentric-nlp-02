#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python ./Relabeling/reduce_embedding_tuning.py \
    --study_name "100 times margin 1~3" \
    --storage_name ""sqlite:///optuna_study.db"" \
    --pretrained_path "team-lucid/deberta-v3-xlarge-korean" \
    --data_path "../../data/using_reduce_dim.csv" \
    --n_trials 100