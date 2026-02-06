#!/bin/bash

# ============================================================
# UI-TARS-1.5-7B Fine-tuning with ZeRO-2
# ============================================================
# Model: ByteDance-Seed/UI-TARS-1.5-7B (based on Qwen2.5-VL)
# Config: ZeRO-2 + CPU Offload (full SFT, no LoRA)
# Data: CSV + Images (multimodal only)
# Recommended: 2x GPU (A100/H100/4090)
# ============================================================

# ============================================================
# Environment Setup
# ============================================================
export QUANTIZATION_BITS=0

# UI-TARS-1.5 is Qwen2.5-VL based; Liger Kernel support is optional
export USE_LIGER_KERNEL=0

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

NUM_GPUS=2

# ZeRO-2 with CPU Offload (sufficient for 7B full SFT)
# Switch to zero2.json (no offload) if you have enough GPU memory
deepspeed_config=/root/autodl-tmp/Qwen3-VL/qwen-vl-finetune/scripts/zero2_offload.json

# Model path (local on autoDL)
llm=./models/UI-TARS-1.5-7B

# ============================================================
# Training Hyperparameters
# ============================================================
lr=1e-5
batch_size=2
grad_accum_steps=8    # Effective batch = 2 * 8 * 2 = 32

# Training entry point (uses CSV multimodal data pipeline)
entry_file=qwen-vl-finetune/qwenvl/train/train_ui_tars.py

# Dataset configuration (CSV + images)
# csv_path: CSV file with columns: messages (JSON), image_payload_id
# image_dir: Directory containing images (filename = image_payload_id.jpg/png)
csv_path=claude_df_converted.csv
image_dir=./images

# Output configuration
output_dir=./output/ui-tars-1.5-7b-sft-zero2
run_name="ui-tars-1.5-7b-sft-zero2"

echo "============================================================"
echo "Starting UI-TARS-1.5-7B Training with ZeRO-2 (CPU Offload)"
echo "============================================================"
echo "Model: ${llm} (Qwen2.5-VL based)"
echo "Data CSV: ${csv_path}"
echo "Image Dir: ${image_dir}"
echo "Output: ${output_dir}"
echo "DeepSpeed: ${deepspeed_config}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${batch_size}"
echo "Gradient accumulation: ${grad_accum_steps}"
echo "Effective batch size: $((batch_size * grad_accum_steps * NUM_GPUS))"
echo "Learning rate: ${lr}"
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
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
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
