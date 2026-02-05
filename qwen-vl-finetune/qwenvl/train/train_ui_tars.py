#!/usr/bin/env python3
"""
Training script for UI-TARS-7B-SFT model
This is a wrapper around train_qwen.py for UI-TARS specific configurations
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the main training function
from qwenvl.train.train_qwen import train

if __name__ == "__main__":
    # UI-TARS uses the same architecture as Qwen2.5-VL
    # So we can directly use the train_qwen.py with flash_attention_2
    print("=" * 80)
    print("🚀 Starting UI-TARS-7B-SFT Fine-tuning")
    print("=" * 80)
    print("📝 Model: UI-TARS-7B-SFT (Based on Qwen2.5-VL-7B)")
    print("🔧 Using Flash Attention 2 for efficient training")
    print("=" * 80)
    print()
    
    train(attn_implementation="flash_attention_2")
