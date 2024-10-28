#!/bin/bash
# -*- coding: utf-8 -*-
export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3

prog='train_SFT.py'

# INPUT-DATA PATH
training_data='../data/train_data.csv'
eval_data='../data/valid_data.csv'
num_train_data=$(wc -l < $training_data)
cache_dir='../data/cache'

# INPUT-LLM PATH
base_modelname=$1

# OUTPUT-LoRA PATH
save_dirname='../models'

# LoRA-MODEL PARAM
lora_r=8 
lora_alpha=16 

target_modules="Wqkv"

# TRAIN PARAM
per_device_batch_size=4 
total_batch_size=16
lr=5e-5 
epochs=1 
max_seq_len=2048 
save_steps=5000
eval_steps=1000
n_worker=0
dtype=bf16

#python $prog \
#accelerate launch $prog \
python $prog \
    --model_name_or_path $1 \
    --train_data_path $training_data \
    --valid_data_path $eval_data \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --output_dir $save_dirname \
    --num_train_epochs $epochs \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy 'steps' \
    --save_strategy 'steps' \
    --save_steps $save_steps \
    --eval_steps $eval_steps \
    --save_total_limit 100 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type 'cosine' \
    --logging_steps 50 \
    --fsdp 'shard_grad_op auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --report_to 'mlflow' 
