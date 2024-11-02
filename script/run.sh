#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export WANDB_API_KEY=

python ./run.py \
    --project_name "Level2_Data-Centric_Baseline" \
    --data_dir "../../data" \
    --output_dir "./output/baseline"