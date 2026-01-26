#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# Quantization configuration
export QUANTIZATION_BITS=8

# DeepSpeed configuration
# MoE model only supports zero2
# Adjust based on your model size and GPU configuration
deepspeed=./qwen-vl-finetune/scripts/zero2.json

# Model configuration
llm=./qwen3-vl-30b-a3b-local

# Training hyperparameters
lr=1e-5
batch_size=2
grad_accum_steps=4

# Training entry point
entry_file=qwen-vl-finetune/qwenvl/train/train_qwen_browser_use.py

# CSV dataset configuration
csv_path=claude_df_converted.csv
image_dir=./images
image_ids_file=""
output_jsonl=""

# Output configuration
run_name="qwen3vl-browser-use"
output_dir=./output

# Training arguments
args="
    --deepspeed ${deepspeed}
    --model_name_or_path ${llm}
    --csv_path ${csv_path}
    --image_dir ${image_dir}
    --data_flatten True
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --bf16
    --lora_enable True
    --output_dir ${output_dir}
    --num_train_epochs 0.5
    --per_device_train_batch_size ${batch_size}
    --per_device_eval_batch_size $((batch_size*2))
    --gradient_accumulation_steps ${grad_accum_steps}
    --max_pixels 50176
    --min_pixels 784
    --eval_strategy no
    --save_strategy steps
    --save_steps 1000
    --save_total_limit 1
    --learning_rate ${lr}
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 8192
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name ${run_name}
    --report_to wandb"

# Add optional arguments if provided
if [ -n "${image_ids_file}" ]; then
    args="${args} --image_ids_file ${image_ids_file}"
fi

if [ -n "${output_jsonl}" ]; then
    args="${args} --output_jsonl ${output_jsonl}"
fi

echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Num Nodes: $NNODES"
echo "Procs per Node: $NPROC_PER_NODE"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
