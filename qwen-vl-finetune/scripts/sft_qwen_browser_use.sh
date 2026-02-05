#!/bin/bash

# ============================================================
# Environment Setup
# ============================================================
export QUANTIZATION_BITS=0

# 🚀 ZeRO-3 使用 CPU offload，显存占用更小，LIGER_KERNEL 可选
export USE_LIGER_KERNEL=1

# ⚠️ 注意：Gradient Checkpointing 必须开启，否则 30B 模型会 OOM
# 已通过 data_flatten=False 解决与 Flash Attention 的兼容性问题 

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

NUM_GPUS=2

# 👇 使用 ZeRO-3 配置（CPU offload）
deepspeed_config=/root/autodl-tmp/Qwen3-VL/qwen-vl-finetune/scripts/zero3_offload.json

llm=./qwen3-vl-30b-a3b-local

# ============================================================
# Training Hyperparameters
# ============================================================
lr=1e-5
batch_size=2          
grad_accum_steps=16   

entry_file=qwen-vl-finetune/qwenvl/train/train_qwen_browser_use.py
csv_path=claude_df_converted.csv
image_dir=./images
output_dir=./output
run_name="qwen3vl-browser-use-zero3-offload"

echo "============================================================"
echo "🚀 Starting Qwen-30B Training with ZeRO-3 (CPU Offload) + Liger Kernel"
echo "============================================================"

deepspeed --num_gpus=${NUM_GPUS} --master_port=29502 \
    ${entry_file} \
    --deepspeed ${deepspeed_config} \
    --model_name_or_path ${llm} \
    --csv_path ${csv_path} \
    --image_dir ${image_dir} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --lora_enable True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 16000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb