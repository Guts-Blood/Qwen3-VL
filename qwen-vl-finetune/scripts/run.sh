#!/bin/bash

# ============================================================
# 显存优化黑科技
# ============================================================
export USE_LIGER_KERNEL=1  # 必须开，把Logits显存从18G降到0.5G
export QUANTIZATION_BITS=0  # 不使用量化，使用 BF16

# ============================================================
# Pipeline Parallelism 配置（不使用 DeepSpeed）
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1

# 优化 NCCL 通信（虽然是单进程，但 device_map="auto" 会用到）
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# 确保不会意外启用 DeepSpeed
unset ACCELERATE_USE_DEEPSPEED

echo "============================================================"
echo "🚀 Starting Qwen-30B Training with Pipeline Parallelism"
echo "   - 2x H100 80GB = 160GB 总显存"
echo "   - device_map='auto': 自动切分模型到 2 个 GPU"
echo "   - 单进程启动（直接 python）"
echo "   - NO DeepSpeed, NO ZeRO-3"
echo "   - Memory optimization: Liger Kernel"
echo "   - Flash Attention: Enabled (data_flatten=False)"
echo "============================================================"

python qwen-vl-finetune/qwenvl/train/train_qwen_browser_use.py \
    --model_name_or_path ./qwen3-vl-30b-a3b-local \
    --csv_path claude_df_converted.csv \
    --image_dir ./images \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir ./output_pp \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --model_max_length 20000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name qwen3vl-pipeline-parallelism
