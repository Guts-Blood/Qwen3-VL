#!/bin/bash

# ============================================================================
# UI-TARS-7B-SFT Fine-tuning Script
# ============================================================================
# Model: ByteDance-Seed/UI-TARS-7B-SFT (7B parameters)
# Recommended: 2x H100 80GB with ZeRO-3
# Supports: 30k token context length
# ============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}  # Number of GPUs per node

# DeepSpeed configuration
# Use ZeRO-3 for better memory efficiency with long context
deepspeed=./scripts/zero3.json

# Model configuration
# Change this to your local path if you've downloaded the model
llm=./models/UI-TARS-7B-SFT
# Or use HuggingFace hub directly:
# llm=ByteDance-Seed/UI-TARS-7B-SFT

# Training hyperparameters
lr=1e-5                    # Learning rate for full fine-tuning
batch_size=1               # Per device batch size (1 for 30k tokens)
grad_accum_steps=16        # Gradient accumulation (effective batch=1*16*2=32)

# Training entry point
entry_file=qwenvl/train/train_ui_tars.py

# Dataset configuration
# Replace with your dataset names (comma-separated)
datasets=your_dataset1,your_dataset2

# Output configuration
run_name="ui-tars-7b-sft"
output_dir=./output/${run_name}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 30000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Print configuration
echo "============================================================================"
echo "🚀 UI-TARS-7B-SFT Training Configuration"
echo "============================================================================"
echo "Model: ${llm}"
echo "Datasets: ${datasets}"
echo "Output: ${output_dir}"
echo "DeepSpeed: ${deepspeed}"
echo "GPUs per node: ${NPROC_PER_NODE}"
echo "Batch size per GPU: ${batch_size}"
echo "Gradient accumulation: ${grad_accum_steps}"
echo "Effective batch size: $((batch_size * grad_accum_steps * NPROC_PER_NODE))"
echo "Learning rate: ${lr}"
echo "Max sequence length: 30000 tokens"
echo "============================================================================"
echo ""

# Check if model exists locally
if [ ! -d "${llm}" ] && [[ ! "${llm}" =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$ ]]; then
    echo "⚠️  Warning: Model directory not found: ${llm}"
    echo "💡 To download the model, run:"
    echo "   python download_ui_tars.py --local_dir ${llm}"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p ${output_dir}

# Launch training with torchrun
echo "🚀 Launching training..."
echo ""

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    ${entry_file} ${args}

# Check training status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ Training completed successfully!"
    echo "📁 Model saved to: ${output_dir}"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "❌ Training failed. Check the logs above for errors."
    echo "============================================================================"
    exit 1
fi
