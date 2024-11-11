#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export WANDB_API_KEY=

python ./Relabeling/reduce_embedding.py \
    --project_name "Level2_Data-Centric_Reduce Dim" \
    --project_sub_name "No header with ContrastiveLoss modify" \
    --pretrained_path "team-lucid/deberta-v3-xlarge-korean" \
    --model_save_path "./output/reduce_embedding/best_model.pth" \
    --data_path "../../data/using_reduce_dim.csv" 