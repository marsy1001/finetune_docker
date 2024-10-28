#!/bin/bash
# -*- coding: utf-8 -*-
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES="0,1"

prog='train_SFT_QLoRA.py'

# INPUT-DATA PATH
training_data='../data/train_data.csv'
eval_data='../data/valid_data.csv'
num_train_data=$(wc -l < $training_data)
cache_dir='..data/cache'

# INPUT-LLM PATH
base_modelname=$1

# OUTPUT-LoRA PATH
save_dirname='../models'

# LoRA-MODEL PARAM
lora_r=8 
lora_alpha=16 
epochs=3 
target_modules="Wqkv" 

# TRAIN PARAM
per_device_batch_size=1
total_batch_size=4 
lr=5e-5 
max_seq_len=2048 
save_steps=10
eval_steps=5
n_worker=0
dtype=bf16 

#accelerate launch $prog \
python3 $prog \
    --model_name $1 \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --train_data_path $training_data \
    --valid_data_path $eval_data \
    --output_dir $save_dirname \
    --num_train_epochs 3 \
    --per_device_train_batch_size $per_device_batch_size \
    --per_device_eval_batch_size $per_device_batch_size \
    --evaluation_strategy 'steps' \
    --eval_steps $eval_steps \
    --save_strategy 'steps' \
    --save_steps $save_steps \
    --learning_rate $lr \
    --save_strategy steps \
    --group_by_length True \
    --logging_strategy steps \
    --logging_steps 50 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.3 \
    --lr_scheduler_type 'cosine' \
    --gradient_accumulation_steps 1 \
    --report_to 'mlflow'