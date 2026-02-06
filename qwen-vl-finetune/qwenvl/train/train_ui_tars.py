#!/usr/bin/env python3
"""
Training script for UI-TARS-1.5-7B model.

UI-TARS-1.5-7B is based on Qwen2.5-VL architecture.
Uses the same CSV-based multimodal data processing pipeline as train_qwen_browser_use.py:
  - Reads training data from CSV (with messages + image_payload_id columns)
  - Loads images from a local directory
  - Only trains on multimodal samples (must have valid images)
  - Sorts by token length to avoid OOM

Usage:
    deepspeed --num_gpus=2 qwen-vl-finetune/qwenvl/train/train_ui_tars.py \
        --deepspeed <zero2_config.json> \
        --model_name_or_path ./models/UI-TARS-1.5-7B \
        --csv_path <your_data.csv> \
        --image_dir <your_image_dir> \
        ...
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the training function from browser_use which handles CSV + image multimodal data
from qwenvl.train.train_qwen_browser_use import train


if __name__ == "__main__":
    print("=" * 80)
    print("Starting UI-TARS-1.5-7B Fine-tuning")
    print("=" * 80)
    print("Model: UI-TARS-1.5-7B (Based on Qwen2.5-VL)")
    print("Data: CSV + Images (multimodal)")
    print("Using Flash Attention 2 for efficient training")
    print("=" * 80)
    print()

    train(attn_implementation="flash_attention_2")
