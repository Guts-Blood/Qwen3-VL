import os
import json
import logging
import pathlib
import torch
import transformers
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import BitsAndBytesConfig
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Try to import Liger Kernel for memory-efficient operations
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2vl  # type: ignore
    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False

from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import (
    LazySupervisedDataset,
    update_processor_pixels
)
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

import gc


def format_bytes(bytes_val):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def print_gpu_memory_details(model=None, stage="", print_model_details=True):
    """
    打印详细的GPU内存使用情况，包括模型权重、激活值、梯度等各部分。
    
    Args:
        model: PyTorch模型，如果提供则计算模型参数内存
        stage: 当前阶段描述（如"After model loading"）
        print_model_details: 是否打印模型参数详细信息
    """
    if not torch.cuda.is_available():
        rank0_print("CUDA not available, skipping GPU memory details")
        return
    
    rank0_print("\n" + "="*80)
    rank0_print(f"GPU Memory Details - {stage}")
    rank0_print("="*80)
    
    # 基本内存信息
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    
    rank0_print(f"\n[Basic Memory Info]")
    rank0_print(f"  Total GPU Memory:     {format_bytes(total_memory)}")
    rank0_print(f"  Currently Allocated:  {format_bytes(allocated)} ({allocated/total_memory*100:.2f}%)")
    rank0_print(f"  Currently Reserved:   {format_bytes(reserved)} ({reserved/total_memory*100:.2f}%)")
    rank0_print(f"  Max Allocated:        {format_bytes(max_allocated)} ({max_allocated/total_memory*100:.2f}%)")
    rank0_print(f"  Max Reserved:          {format_bytes(max_reserved)} ({max_reserved/total_memory*100:.2f}%)")
    rank0_print(f"  Free Memory:           {format_bytes(total_memory - reserved)}")
    
    # 详细内存统计
    stats = torch.cuda.memory_stats(device)
    rank0_print(f"\n[Detailed Memory Breakdown]")
    rank0_print(f"  Active Bytes:         {format_bytes(stats.get('active_bytes.all.current', 0))}")
    rank0_print(f"  Inactive Split Bytes: {format_bytes(stats.get('inactive_split_bytes.all.current', 0))}")
    rank0_print(f"  Allocated Bytes:      {format_bytes(stats.get('allocated_bytes.all.current', 0))}")
    rank0_print(f"  Reserved Bytes:       {format_bytes(stats.get('reserved_bytes.all.current', 0))}")
    rank0_print(f"  Active All:           {format_bytes(stats.get('active_bytes.all.peak', 0))}")
    
    # 模型参数内存
    if model is not None and print_model_details:
        rank0_print(f"\n[Model Parameters Memory]")
        total_params = 0
        trainable_params = 0
        param_memory = 0
        buffer_memory = 0
        
        trainable_param_memory = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            param_memory += param.numel() * param.element_size()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_param_memory += param.numel() * param.element_size()
        
        for name, buffer in model.named_buffers():
            buffer_memory += buffer.numel() * buffer.element_size()
        
        rank0_print(f"  Total Parameters:     {total_params:,} ({format_bytes(param_memory)})")
        rank0_print(f"  Trainable Parameters: {trainable_params:,} ({format_bytes(trainable_param_memory)})")
        rank0_print(f"  Non-trainable Params: {total_params - trainable_params:,} ({format_bytes(param_memory - trainable_param_memory)})")
        rank0_print(f"  Buffer Memory:        {format_bytes(buffer_memory)}")
        
        # 按模块分组统计
        module_memory = {}
        for name, param in model.named_parameters():
            module_name = name.split('.')[0] if '.' in name else name
            if module_name not in module_memory:
                module_memory[module_name] = 0
            module_memory[module_name] += param.numel() * param.element_size()
        
        rank0_print(f"\n[Memory by Module (Top 10)]")
        sorted_modules = sorted(module_memory.items(), key=lambda x: x[1], reverse=True)
        for module_name, mem in sorted_modules[:10]:
            rank0_print(f"  {module_name:30s}: {format_bytes(mem)}")
    
    # 梯度内存（如果在训练中）
    if model is not None:
        gradient_memory = 0
        
        for param in model.parameters():
            if param.grad is not None:
                gradient_memory += param.grad.numel() * param.grad.element_size()
        
        rank0_print(f"\n[Gradients]")
        rank0_print(f"  Gradient Memory:      {format_bytes(gradient_memory)}")
        if gradient_memory == 0:
            rank0_print(f"  Note: Optimizer states are managed by Trainer and not directly accessible here")
    
    # 内存碎片信息
    rank0_print(f"\n[Memory Fragmentation]")
    fragmentation = stats.get('inactive_split_bytes.all.current', 0) / max(reserved, 1) * 100
    rank0_print(f"  Fragmentation:        {fragmentation:.2f}%")
    
    rank0_print("="*80 + "\n")


# ⚡️ 1. 定义一个自定义回调类 ⚡️
class MemoryClearCallback(transformers.TrainerCallback):
    """
    在训练过程中定期执行垃圾回收和CUDA缓存清理，
    以缓解训练中的内存碎片化问题。
    同时打印详细的内存使用情况。
    """
    def __init__(self, print_interval=10, cleanup_interval=50):
        """
        Args:
            print_interval: 每隔多少步打印一次详细内存信息
            cleanup_interval: 每隔多少步执行一次内存清理（减少频率以提高性能）
        """
        self.print_interval = print_interval
        self.cleanup_interval = cleanup_interval
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        
        # 每隔N步打印一次详细内存信息
        if self.step_count % self.print_interval == 0:
            model = kwargs.get('model', None)
            print_gpu_memory_details(
                model=model,
                stage=f"Step {state.global_step}",
                print_model_details=False  # 训练中不打印模型详情以节省时间
            )
        
        # 定期清理内存，而不是每个步骤都清理（提高性能）
        if self.step_count % self.cleanup_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 返回 control，确保训练继续进行
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时打印内存信息"""
        model = kwargs.get('model', None)
        print_gpu_memory_details(
            model=model,
            stage="Training Start",
            print_model_details=True
        )
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时打印内存信息"""
        model = kwargs.get('model', None)
        print_gpu_memory_details(
            model=model,
            stage="Training End",
            print_model_details=True
        )
        return control

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def parse_messages(messages_str: str) -> List[Dict[str, str]]:
    """
    Parse messages field from CSV (can be JSON string or already parsed).
    
    Args:
        messages_str: String representation of messages (JSON or list)
    
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    if pd.isna(messages_str) or messages_str == "":
        return []
    
    # If it's already a string representation of JSON
    if isinstance(messages_str, str):
        try:
            messages = json.loads(messages_str)
        except json.JSONDecodeError:
            # Try to handle it as a Python literal (with single quotes)
            import ast
            try:
                messages = ast.literal_eval(messages_str)
            except:
                rank0_print(f"Warning: Could not parse messages as JSON: {messages_str[:100]}")
                return []
    
    # Handle case where messages is not a list
    if not isinstance(messages, list):
        rank0_print(f"Warning: messages is not a list: {type(messages)}")
        return []
    
    # Convert to standard format
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role in ["user", "assistant", "system"] and content:
                result.append({
                    "role": role,
                    "content": str(content)
                })
    
    return result


def download_images_if_needed(
    image_ids_file: str,
    image_dir: str,
) -> Dict[str, str]:
    """
    Collect local images that already exist on disk.
    
    Returns:
        Dictionary mapping image_payload_id to image path
    """
    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)
    
    mapping_file = image_dir_path / "image_mapping.json"
    
    # Check which images we need
    rank0_print(f"Loading image IDs from: {image_ids_file}")
    with open(image_ids_file, "r", encoding="utf-8") as f:
        needed_image_ids = set(json.load(f))
    
    # Check which images are already downloaded (support multiple formats)
    existing_images = {}
    # Supported image formats
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    for ext in image_extensions:
        for img_file in image_dir_path.glob(ext):
            image_id = img_file.stem
            if image_id in needed_image_ids:
                existing_images[image_id] = str(img_file.absolute())
    
    rank0_print(f"Found {len(existing_images)} existing images out of {len(needed_image_ids)} needed")
    
    missing_ids = needed_image_ids - set(existing_images.keys())
    if missing_ids:
        rank0_print(
            f"Skipping {len(missing_ids)} images that are listed but missing locally."
        )
    
    # Persist current mapping for next run
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(existing_images, f, ensure_ascii=False, indent=2)
    rank0_print(f"Saved {len(existing_images)} image mappings to {mapping_file}")
    
    return existing_images


def convert_csv_to_training_data(
    csv_path: str,
    image_mapping: Dict[str, str],
    tokenizer,
    output_jsonl: Optional[str] = None,
    chunk_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Convert CSV data to training format.
    
    Args:
        csv_path: Path to CSV file
        image_mapping: Dictionary mapping image_payload_id to image path
        tokenizer: Tokenizer for calculating token lengths
        output_jsonl: Optional path to save JSONL file
        chunk_size: Number of rows to process at a time
    
    Returns:
        List of training data items with num_tokens field for sorting
    """
    training_data = []
    
    rank0_print(f"Reading CSV from: {csv_path}")
    
    # Process CSV in chunks
    total_processed = 0
    total_valid = 0
    skipped_no_image_id = 0
    skipped_image_not_found = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        for idx, row in chunk.iterrows():
            total_processed += 1
            try:
                # Parse messages
                messages = parse_messages(row.get("messages", ""))
                if not messages:
                    continue
                
                # Get image path - 只训练多模态数据，必须有图片
                image_payload_id = row.get("image_payload_id", "")
                
                # 强制要求必须有 image_payload_id
                if not pd.notna(image_payload_id) or not str(image_payload_id).strip():
                    # Skip samples without image_payload_id (只训练多模态数据)
                    skipped_no_image_id += 1
                    continue
                
                image_id = str(image_payload_id).strip()
                
                # 强制要求图片必须存在于 image_mapping 中
                if image_id not in image_mapping:
                    # Skip samples whose images are not found locally
                    skipped_image_not_found += 1
                    continue
                
                image_path = image_mapping[image_id]
                
                # Convert messages to conversations format (data is already converted)
                # Messages are in Qwen format: system, user, assistant
                # We convert to conversations format: from: human/gpt, value: content
                conversations = []
                last_user_idx = None
                
                # Find last user message index for image insertion
                for i, msg in enumerate(messages):
                    if msg["role"] == "user":
                        last_user_idx = i
                
                # Process messages and convert to conversations format
                # System messages are merged into the first user message
                system_content = None
                has_user_message = False
                for i, msg in enumerate(messages):
                    role = msg["role"]
                    content = msg["content"]
                    
                    if role == "system":
                        # Collect system message to merge into first user message
                        if system_content is None:
                            system_content = content
                        else:
                            system_content += "\n" + content
                    elif role == "user":
                        has_user_message = True
                        # Merge system message into first user message if exists
                        if system_content is not None:
                            content = f"{system_content}\n{content}"
                            system_content = None  # Only merge once
                        
                        # Add image placeholder if this is the last user message and we have an image
                        if image_path and i == last_user_idx:
                            # Check if content already has <image> tag
                            if "<image>" not in content:
                                content = f"{content}\n<image>"
                        conversations.append({
                            "from": "human",
                            "value": content
                        })
                    elif role == "assistant":
                        conversations.append({
                            "from": "gpt",
                            "value": content
                        })
                
                # If system message exists but no user message, skip this sample
                if system_content is not None and not has_user_message:
                    continue
                
                # Build training data item
                if not conversations:
                    continue
                
                result = {
                    "conversations": conversations
                }
                
                # Store image path - _build_messages will convert it to absolute path using data_path
                # Store just the filename since images are in image_dir which will be set as data_path
                # image_path 必须存在（因为前面的逻辑已经保证）
                image_path_obj = Path(image_path)
                # Store just the filename - it will be resolved relative to data_path (image_dir)
                result["image"] = image_path_obj.name
                
                # Calculate token length for sorting (use CSV total_tokens if available, otherwise estimate)
                # Use the messages format for tokenization (Qwen format with system/user/assistant)
                # image_path 必须存在（因为前面的逻辑已经保证），所以总是添加 128 tokens for image
                total_tokens = row.get("total_tokens", None)
                if pd.notna(total_tokens):
                    try:
                        # Use CSV total_tokens as it's more accurate
                        csv_tokens = int(total_tokens)
                        # Add image tokens (typically 128 tokens per image)
                        result["num_tokens"] = csv_tokens + 128
                    except (ValueError, TypeError):
                        # If total_tokens is invalid, estimate from messages
                        formatted = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        result["num_tokens"] = len(tokenizer.encode(formatted, add_special_tokens=False))
                        # Add image tokens (typically 128 tokens per image)
                        result["num_tokens"] += 128
                else:
                    # Estimate token length if not available in CSV
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    result["num_tokens"] = len(tokenizer.encode(formatted, add_special_tokens=False))
                    # Add image tokens (typically 128 tokens per image)
                    result["num_tokens"] += 128
                
                training_data.append(result)
                total_valid += 1
                
            except Exception as e:
                rank0_print(f"Error processing row {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            if total_processed % 100 == 0:
                rank0_print(f"Processed: {total_processed}, Valid: {total_valid}, "
                          f"Skipped (no image_id): {skipped_no_image_id}, "
                          f"Skipped (image not found): {skipped_image_not_found}")
    
    rank0_print(f"\n[Data Processing Summary]")
    rank0_print(f"Total processed: {total_processed}")
    rank0_print(f"Valid (with images): {total_valid}")
    rank0_print(f"Skipped (no image_payload_id): {skipped_no_image_id}")
    rank0_print(f"Skipped (image not found locally): {skipped_image_not_found}")
    rank0_print(f"Note: Only multimodal data (with images) will be trained")
    
    # Use print() to ensure this message always appears
    print(f"\n{'='*80}")
    print(f"[DATA PROCESSING SUMMARY]")
    print(f"Total valid samples before filtering: {len(training_data)}")
    print(f"{'='*80}\n")
    
    if not training_data:
        print("=" * 80)
        print("ERROR: No valid training data found!")
        print("=" * 80)
        rank0_print("Warning: No valid training data found!")
        return training_data
    
    # Filter out samples that exceed max token length to prevent OOM
    max_token_length = 20000  # Should match model_max_length in training args
    original_count = len(training_data)
    training_data = [item for item in training_data if item.get("num_tokens", 0) <= max_token_length]
    filtered_count = original_count - len(training_data)
    
    print(f"\n{'='*80}")
    print(f"[TOKEN LENGTH FILTERING]")
    print(f"Max token length: {max_token_length}")
    print(f"Original count: {original_count}")
    print(f"Filtered out: {filtered_count}")
    print(f"Remaining samples: {len(training_data)}")
    print(f"{'='*80}\n")
    
    if filtered_count > 0:
        rank0_print(f"Filtered out {filtered_count} samples exceeding {max_token_length} tokens")
        rank0_print(f"Remaining samples: {len(training_data)}")
    
    if not training_data:
        print("=" * 80)
        print("ERROR: All samples were filtered out due to token length!")
        print("=" * 80)
        return training_data
    
    # Sort by token length (ascending) to avoid OOM
    rank0_print("Sorting data by token length (ascending)...")
    training_data.sort(key=lambda x: x.get("num_tokens", 0))
    min_tokens = training_data[0].get('num_tokens', 0)
    max_tokens = training_data[-1].get('num_tokens', 0)
    
    print(f"\n{'='*80}")
    print(f"[TOKEN LENGTH RANGE AFTER SORTING]")
    print(f"Min tokens: {min_tokens}")
    print(f"Max tokens: {max_tokens}")
    print(f"{'='*80}\n")
    
    rank0_print(f"Token length range: {min_tokens} - {max_tokens} tokens")
    
    # Save to JSONL if requested
    if output_jsonl:
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        rank0_print(f"Saved {len(training_data)} items to {output_jsonl}")
    
    return training_data


class BrowserUseDataset(LazySupervisedDataset):
    """Custom dataset for browser-use data from CSV."""
    
    def __init__(self, processor, data_args, training_data: List[Dict[str, Any]], image_dir: str = "./images"):
        # Initialize parent Dataset class (but skip LazySupervisedDataset.__init__ 
        # since we're providing our own data)
        from torch.utils.data import Dataset
        Dataset.__init__(self)
        
        # Set model type for rope index
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            from qwenvl.data.rope2d import get_rope_index_3
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            from qwenvl.data.rope2d import get_rope_index_25
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            from qwenvl.data.rope2d import get_rope_index_2
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")
        
        # Add data_path to each item (required by preprocess_qwen_visual)
        image_dir_path = Path(image_dir).resolve()
        for item in training_data:
            item["data_path"] = str(image_dir_path)
        
        # Store training data (already sorted by token length ascending)
        self.list_data_dict = training_data
        
        # Use print() to ensure this critical information is visible
        print(f"\n{'='*80}")
        print(f"[DATASET INITIALIZATION]")
        print(f"Total training samples in dataset: {len(self.list_data_dict)}")
        if len(self.list_data_dict) > 0:
            min_tokens = self.list_data_dict[0].get("num_tokens", 0)
            max_tokens = self.list_data_dict[-1].get("num_tokens", 0)
            print(f"Token length range: {min_tokens} - {max_tokens} tokens (sorted ascending)")
        print(f"{'='*80}\n")
        
        rank0_print(f"Total training samples: {len(self.list_data_dict)}")
        if len(self.list_data_dict) > 0:
            min_tokens = self.list_data_dict[0].get("num_tokens", 0)
            max_tokens = self.list_data_dict[-1].get("num_tokens", 0)
            rank0_print(f"Token length range: {min_tokens} - {max_tokens} tokens (sorted ascending)")
        
        # Initialize processor and other attributes
        # Note: processor should already be updated via update_processor_pixels before this
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        
        # Set item function (inherited from LazySupervisedDataset)
        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Get CSV dataset arguments
    csv_path = data_args.csv_path
    image_ids_file = data_args.image_ids_file
    image_dir = data_args.image_dir or './images'
    output_jsonl = data_args.output_jsonl
    
    if not csv_path:
        raise ValueError("csv_path must be provided for browser-use dataset training")

    # Default to Qwen3-VL-30B-A3B-Instruct if not specified
    if not hasattr(model_args, 'model_name_or_path') or not model_args.model_name_or_path:
        model_args.model_name_or_path = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        rank0_print(f"Using default model: {model_args.model_name_or_path}")
    
    # Apply Liger Kernel optimizations BEFORE loading model
    # This monkey-patches the model classes to use memory-efficient fused operations
    use_liger_kernel = int(os.environ.get("USE_LIGER_KERNEL", "1")) == 1
    if LIGER_KERNEL_AVAILABLE and use_liger_kernel:
        rank0_print("=" * 80)
        rank0_print("🚀 Applying Liger Kernel optimizations (before model loading)...")
        rank0_print("   - Fused RMSNorm (saves activation memory)")
        rank0_print("   - Fused RoPE (saves intermediate memory)")
        rank0_print("   - Fused SwiGLU (saves FFN activation memory)")
        rank0_print("   - Fused Cross Entropy (saves ~85% logits memory!)")
        rank0_print("   Expected memory savings: 10-15GB for 30B model")
        rank0_print("=" * 80)
        try:
            # Apply Liger Kernel to Qwen2VL/Qwen3VL model classes
            # This must be done BEFORE model instantiation
            apply_liger_kernel_to_qwen2vl(
                rope=True,              # Fused RoPE
                rms_norm=True,          # Fused RMSNorm
                swiglu=True,            # Fused SwiGLU
                cross_entropy=True,     # Fused Cross Entropy (biggest saving!)
                fused_linear_cross_entropy=True,  # Extra optimization
            )
            rank0_print("✓ Liger Kernel applied successfully!")
        except Exception as e:
            rank0_print(f"⚠ Warning: Failed to apply Liger Kernel: {e}")
            rank0_print("   Continuing with standard operations...")
    elif not LIGER_KERNEL_AVAILABLE:
        rank0_print("=" * 80)
        rank0_print("ℹ️  Liger Kernel not available (install with: pip install liger-kernel)")
        rank0_print("   Running with standard operations")
        rank0_print("=" * 80)
    else:
        rank0_print("ℹ️  Liger Kernel disabled by USE_LIGER_KERNEL=0")
    
    # 判断模型类型
    model_path_lower = model_args.model_name_or_path.lower()
    is_moe_model = "qwen3" in model_path_lower and ("a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower() or "a3b" in model_path_lower)
    # ⚠️ 注意：FP8 模型不支持训练，只加载原始 FP16/BF16 模型
    # 如果路径中包含 fp8，给出警告但继续加载原始模型
    is_fp8_model = "fp8" in model_path_lower
    if is_fp8_model:
        rank0_print("=" * 80)
        rank0_print("⚠️  WARNING: FP8 models cannot be trained!")
        rank0_print("   Loading original FP16/BF16 model instead.")
        rank0_print("   Please use the non-FP8 model path for training.")
        rank0_print("=" * 80)
    # 检查模型大小：30B, 32B等大模型需要量化
    is_large_model = any(size in model_path_lower for size in ["30b", "32b", "70b", "235b"])
    # 使用 Zero3 CPU Offload 时，不需要量化，直接使用 BF16
    use_quantization = False  # Zero3 CPU Offload 模式下禁用量化，使用 BF16 + CPU Offload
    
    # 量化配置：Zero3 CPU Offload 模式下使用 BF16，不需要量化
    # 从环境变量读取量化位数，默认为 0（禁用量化，使用 Zero3 CPU Offload）
    quantization_bits = int(os.environ.get("QUANTIZATION_BITS", "0"))
    rank0_print(f"Quantization bits from env: {quantization_bits}")
    
    # 如果量化位数为 0，禁用量化，使用 BF16 + DeepSpeed Zero3 CPU Offload
    if quantization_bits == 0:
        rank0_print("Quantization disabled (QUANTIZATION_BITS=0), using BF16 + DeepSpeed Zero3 CPU Offload")
        use_quantization = False
        quantization_config = None
    elif use_quantization:
        if quantization_bits == 4:
            rank0_print("Using 4-bit quantization (NF4)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 
            )
        else:
            # 默认使用 8-bit 量化
            rank0_print("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
    else:
        quantization_config = None
    
    # Load model
    # ⚠️ 注意：FP8 模型不支持训练，始终加载原始 FP16/BF16 模型
    if is_moe_model:
        if quantization_config is not None:
            # MoE模型（如qwen3-vl-30B-A3B）使用量化配置以节省内存
            rank0_print(f"Loading MoE model with {quantization_bits}-bit quantization: {model_args.model_name_or_path}")
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            # 不使用量化，使用 BF16 + DeepSpeed Zero3 CPU Offload
            rank0_print(f"Loading MoE model with BF16 (no quantization, Zero3 CPU Offload): {model_args.model_name_or_path}")
            # 如果路径包含 fp8，尝试加载对应的原始模型路径
            model_path = model_args.model_name_or_path
            if "fp8" in model_path.lower():
                # 尝试将 FP8 路径转换为原始模型路径
                model_path = model_path.replace("-FP8", "").replace("-fp8", "").replace("_FP8", "").replace("_fp8", "")
                rank0_print(f"   Converted FP8 path to original model path: {model_path}")
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_path_lower:
        # 普通qwen3模型：使用 BF16/FP16，配合 Zero3 CPU Offload
        if use_quantization:
            rank0_print(f"Loading large Qwen3 model with {quantization_bits}-bit quantization: {model_args.model_name_or_path}")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            rank0_print(f"Loading Qwen3 model with BF16/FP16 (no quantization, Zero3 CPU Offload): {model_args.model_name_or_path}")
            # 如果路径包含 fp8，尝试加载对应的原始模型路径
            model_path = model_args.model_name_or_path
            if "fp8" in model_path.lower():
                # 尝试将 FP8 路径转换为原始模型路径
                model_path = model_path.replace("-FP8", "").replace("-fp8", "").replace("_FP8", "").replace("_fp8", "")
                rank0_print(f"   Converted FP8 path to original model path: {model_path}")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
                device_map="auto",
            )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    rank0_print(f'the initialized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    
    # 打印模型的 device_map（如果使用了 device_map="auto"）
    if hasattr(model, "hf_device_map"):
        rank0_print("=" * 80)
        rank0_print("📍 Model Device Map (Pipeline Parallelism):")
        device_distribution = {}
        for module_name, device in model.hf_device_map.items():
            device_str = str(device)
            if device_str not in device_distribution:
                device_distribution[device_str] = []
            device_distribution[device_str].append(module_name)
        
        for device, modules in sorted(device_distribution.items()):
            rank0_print(f"   {device}: {len(modules)} modules")
            if len(modules) <= 10:
                for mod in modules[:10]:
                    rank0_print(f"      - {mod}")
            else:
                for mod in modules[:5]:
                    rank0_print(f"      - {mod}")
                rank0_print(f"      ... and {len(modules)-5} more modules")
        rank0_print("=" * 80)
    else:
        rank0_print("⚠️  No hf_device_map found - model may not be using device_map='auto'")
    
    # 打印每个 GPU 的内存使用情况
    if torch.cuda.is_available():
        rank0_print("=" * 80)
        rank0_print("📊 GPU Memory Usage (After Model Loading):")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            rank0_print(f"   GPU {i}: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total")
        rank0_print("=" * 80)
    
    # 验证模型配置：打印模型总参数数（从 config 获取，不受 ZeRO-3 影响）
    if hasattr(model.config, 'num_parameters'):
        total_params_from_config = model.config.num_parameters
    else:
        # 尝试从 hidden_size 等配置估算（对于 MoE 模型可能不准确）
        total_params_from_config = None
    
    # 检查是否使用 DeepSpeed ZeRO-3
    is_zero3 = False
    if training_args.deepspeed:
        try:
            with open(training_args.deepspeed, 'r') as f:
                ds_config = json.load(f)
                if ds_config.get('zero_optimization', {}).get('stage') == 3:
                    is_zero3 = True
                    has_cpu_offload = (
                        ds_config.get('zero_optimization', {}).get('offload_param', {}).get('device') == 'cpu' or
                        ds_config.get('zero_optimization', {}).get('offload_optimizer', {}).get('device') == 'cpu'
                    )
        except:
            pass
    
    if is_zero3:
        rank0_print("=" * 80)
        rank0_print("ℹ️  ZeRO-3 + CPU Offload Mode Detected")
        rank0_print("   This is NORMAL behavior:")
        rank0_print("   - Model parameters are sharded across GPUs and offloaded to CPU")
        rank0_print("   - Only parameters needed for current computation are on GPU")
        rank0_print("   - GPU memory usage will be much lower than full model size")
        if total_params_from_config:
            rank0_print(f"   - Model has ~{total_params_from_config/1e9:.2f}B total parameters")
        rank0_print("   - Parameters will be loaded to GPU on-demand during training")
        rank0_print("=" * 80)
    
    # ⚠️ 检查并清理 quantization_config（如果存在）
    # 原始模型不应该有 quantization_config，但为了安全起见还是检查一下
    if hasattr(model.config, 'quantization_config') and model.config.quantization_config is not None:
        rank0_print("=" * 80)
        rank0_print("⚠️  WARNING: Found quantization_config in model.config (unexpected for original model)")
        rank0_print("   Deleting quantization_config attribute to allow training...")
        delattr(model.config, 'quantization_config')
        rank0_print("✓ quantization_config attribute deleted from model.config")
        rank0_print("=" * 80)
    
    # Print memory after model loading
    print_gpu_memory_details(model=model, stage="After Model Loading (with Liger Kernel if enabled)", print_model_details=True)
    
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    # NOTE: DO NOT enable gradient checkpointing here!
    # It must be enabled AFTER LoRA layers are added, otherwise it won't work for LoRA parameters
    # See gradient_checkpointing_enable() call after get_peft_model()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        rank0_print("LoRA enabled")

        # 检查模型是否真的被量化了（通过检查参数的 dtype）
        sample_param = next(model.parameters())
        is_actually_quantized = hasattr(sample_param, 'quant_state') or sample_param.dtype == torch.int8
        rank0_print(f"Sample parameter dtype: {sample_param.dtype}, is_actually_quantized: {is_actually_quantized}")
        
        # First freeze all parameters before adding LoRA
        for p in model.parameters():
            p.requires_grad = False
        
        # 对于量化模型（4-bit 或 8-bit），需要先调用 prepare_model_for_kbit_training
        # 但如果量化没有真正生效，跳过这一步以避免 OOM
        # NOTE: Do NOT enable gradient checkpointing here - will be done after LoRA
        if use_quantization and is_actually_quantized:
            rank0_print("Preparing quantized model for k-bit training...")
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=False,  # Will enable after LoRA
            )
        elif use_quantization and not is_actually_quantized:
            rank0_print("WARNING: Quantization was requested but model is not actually quantized!")
            rank0_print("Skipping prepare_model_for_kbit_training to avoid OOM.")
            rank0_print("Model will be trained with BF16 weights + LoRA.")

        # For MoE models, include expert FFN layers in addition to attention layers
        # - q_proj, k_proj, v_proj, o_proj: attention projections
        # - gate_proj, up_proj, down_proj: MoE expert FFN layers
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj",      # MoE expert FFN layers
        ]
        rank0_print(f"LoRA target modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # ⚠️ Critical: Ensure quantization_config is deleted before PEFT processing
        # PEFT's get_peft_model() calls config.to_dict() which tries to serialize quantization_config
        if hasattr(model.config, 'quantization_config'):
            rank0_print("=" * 80)
            rank0_print("⚠️  Final cleanup: Deleting quantization_config before PEFT processing...")
            delattr(model.config, 'quantization_config')
            rank0_print("✓ quantization_config deleted (PEFT-safe)")
            rank0_print("=" * 80)
        
        model = get_peft_model(model, lora_config)
        
        # Check if gradient checkpointing should be disabled (for Zero3 compatibility)
        disable_grad_ckpt = int(os.environ.get("DISABLE_GRADIENT_CHECKPOINTING", "0")) == 1
        use_grad_ckpt = training_args.gradient_checkpointing and not disable_grad_ckpt
        
        # CRITICAL: For DeepSpeed Zero Stage 3, we must enable gradient checkpointing
        # BEFORE calling enable_input_require_grads() to ensure consistent module counting
        # across all ranks. This is a known DeepSpeed + PEFT compatibility issue.
        if use_grad_ckpt:
            rank0_print("=" * 80)
            rank0_print("Enabling gradient checkpointing (Zero3-compatible mode)...")
            
            # For Zero3, we need to enable checkpointing on the base model first
            # before any hooks are registered
            if hasattr(model, "base_model"):
                model.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            rank0_print("✓ Gradient checkpointing enabled (before input gradients)")
            rank0_print("=" * 80)
        elif disable_grad_ckpt:
            rank0_print("=" * 80)
            rank0_print("⚠️  Gradient checkpointing DISABLED by environment variable")
            rank0_print("   This may help resolve Zero3 + LoRA module count mismatch issues")
            rank0_print("   Memory usage will be higher, but Zero3 still saves significant memory")
            rank0_print("=" * 80)
        
        # THEN enable input gradients (after gradient checkpointing)
        # This order is critical for Zero3 + LoRA
        model.enable_input_require_grads()
        
        # Additional hook to ensure inputs have requires_grad=True for gradient checkpointing
        # This is CRITICAL for LoRA + Gradient Checkpointing + Flash Attention + DeepSpeed ZeRO-3
        def make_inputs_require_grad(module, input, output):
            """
            Force output tensors to have requires_grad=True.
            This is essential for gradient checkpointing with LoRA.
            """
            if isinstance(output, torch.Tensor):
                if not output.requires_grad:
                    output.requires_grad_(True)
            elif isinstance(output, (tuple, list)):
                # Convert to list to modify
                new_output = []
                for item in output:
                    if isinstance(item, torch.Tensor) and not item.requires_grad:
                        item.requires_grad_(True)
                    new_output.append(item)
                # Preserve original type (tuple or list)
                output = type(output)(new_output)
            return output
        
        # Register hooks on ALL possible embedding layers to ensure coverage
        # For DeepSpeed ZeRO-3 + PEFT models, the structure can be complex
        hooks_registered = 0
        
        rank0_print("=" * 80)
        rank0_print("Searching for embedding layers to register gradient hooks...")
        
        # Strategy 1: Try base_model.model hierarchy (PEFT + DeepSpeed)
        if hasattr(model, "base_model"):
            rank0_print(f"Found model.base_model: {type(model.base_model)}")
            if hasattr(model.base_model, "model"):
                base_model = model.base_model.model
                rank0_print(f"Found model.base_model.model: {type(base_model)}")
                
                # Try language_model.embed_tokens (Qwen3VL MoE)
                if hasattr(base_model, "language_model") and hasattr(base_model.language_model, "embed_tokens"):
                    base_model.language_model.embed_tokens.register_forward_hook(make_inputs_require_grad)
                    rank0_print("✓ Registered gradient hook on base_model.model.language_model.embed_tokens")
                    hooks_registered += 1
                
                # Try model.embed_tokens (standard)
                if hasattr(base_model, "embed_tokens"):
                    base_model.embed_tokens.register_forward_hook(make_inputs_require_grad)
                    rank0_print("✓ Registered gradient hook on base_model.model.embed_tokens")
                    hooks_registered += 1
            
            # Try get_input_embeddings
            if hasattr(model.base_model, "get_input_embeddings"):
                try:
                    embeddings = model.base_model.get_input_embeddings()
                    if embeddings is not None:
                        embeddings.register_forward_hook(make_inputs_require_grad)
                        rank0_print(f"✓ Registered gradient hook on base_model.get_input_embeddings() → {type(embeddings)}")
                        hooks_registered += 1
                except Exception as e:
                    rank0_print(f"   Failed to get base_model.get_input_embeddings(): {e}")
        
        # Strategy 2: Top-level model.get_input_embeddings (fallback)
        if hooks_registered == 0:
            if hasattr(model, "get_input_embeddings"):
                try:
                    embeddings = model.get_input_embeddings()
                    if embeddings is not None:
                        embeddings.register_forward_hook(make_inputs_require_grad)
                        rank0_print(f"✓ Registered gradient hook on model.get_input_embeddings() → {type(embeddings)}")
                        hooks_registered += 1
                except Exception as e:
                    rank0_print(f"   Failed to get model.get_input_embeddings(): {e}")
        
        # Strategy 3: Manual search for Embedding layer (last resort)
        if hooks_registered == 0:
            rank0_print("⚠️  Attempting manual search for Embedding layers...")
            for name, module in model.named_modules():
                if "embed_tokens" in name or "wte" in name or isinstance(module, torch.nn.Embedding):
                    try:
                        module.register_forward_hook(make_inputs_require_grad)
                        rank0_print(f"✓ Registered gradient hook on {name} ({type(module)})")
                        hooks_registered += 1
                        break  # Only hook the first embedding layer found
                    except Exception as e:
                        rank0_print(f"   Failed to hook {name}: {e}")
        
        rank0_print("=" * 80)
        if hooks_registered == 0:
            rank0_print("❌ CRITICAL: Could not register any gradient hooks!")
            rank0_print("   Training WILL fail with 'None of the inputs have requires_grad=True'")
            rank0_print("   Please check model structure manually")
        else:
            rank0_print(f"✅ Input gradients enabled ({hooks_registered} hook(s) registered successfully)")
        
        # 验证一下是否开启成功
        if hasattr(model, "is_gradient_checkpointing"):
            rank0_print(f"Gradient Checkpointing Enabled: {model.is_gradient_checkpointing}")
        elif hasattr(model, "base_model"):
            if hasattr(model.base_model, "is_gradient_checkpointing"):
                rank0_print(f"Gradient Checkpointing Enabled: {model.base_model.is_gradient_checkpointing}")
            elif hasattr(model.base_model, "model") and hasattr(model.base_model.model, "is_gradient_checkpointing"):
                rank0_print(f"Gradient Checkpointing Enabled: {model.base_model.model.is_gradient_checkpointing}")
        else:
            rank0_print("⚠️  Could not verify gradient checkpointing status (may be in base model)")
        
        # CRITICAL: Multiple synchronization barriers for DeepSpeed Zero3 + LoRA + Gradient Checkpointing
        # Ensure all ranks have the same model structure at each step
        if torch.distributed.is_initialized():
            # Sync after LoRA application
            torch.distributed.barrier()
            rank0_print("✓ All ranks synchronized after LoRA setup")
            
            # Sync after gradient checkpointing (if enabled)
            if use_grad_ckpt:
                torch.distributed.barrier()
                rank0_print("✓ All ranks synchronized after gradient checkpointing")
            
            # Sync after input gradients enabled
            torch.distributed.barrier()
            rank0_print("✓ All ranks synchronized after input gradients enabled")
            
            # Final verification: Check module count consistency
            try:
                module_count = len(list(model.named_modules()))
                module_counts = [0] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(module_counts, module_count)
                
                if len(set(module_counts)) > 1:
                    rank0_print("=" * 80)
                    rank0_print("⚠️  WARNING: Module count mismatch detected!")
                    rank0_print(f"   Module counts across ranks: {module_counts}")
                    rank0_print("   This may cause ZeRO-3 initialization issues.")
                    rank0_print("   Attempting additional synchronization...")
                    rank0_print("=" * 80)
                    
                    # Additional synchronization attempts
                    for i in range(3):
                        torch.distributed.barrier()
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                    
                    # Re-check module count
                    module_count_after = len(list(model.named_modules()))
                    module_counts_after = [0] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(module_counts_after, module_count_after)
                    
                    if len(set(module_counts_after)) > 1:
                        rank0_print(f"⚠️  Module count still inconsistent: {module_counts_after}")
                        rank0_print("   Training may fail, but attempting to continue...")
                    else:
                        rank0_print(f"✓ Module count now consistent: {module_counts_after}")
                else:
                    rank0_print(f"✓ Module count consistent across all ranks: {module_counts[0]}")
            except Exception as e:
                rank0_print(f"⚠️  Could not verify module consistency: {e}")
        
        # Print trainable parameters to verify LoRA is working
        model.print_trainable_parameters()
        
        # Print memory after LoRA application
        print_gpu_memory_details(model=model, stage="After LoRA Application", print_model_details=True)
    else:
        set_model(model_args, model)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
        elif not torch.distributed.is_initialized():
            # Single GPU or CPU training
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
        
        # =================================================================
        # 强制开启 Gradient Checkpointing（适用于 Zero3 CPU Offload）
        # =================================================================
        # Check if gradient checkpointing should be disabled (for Zero3 compatibility)
        disable_grad_ckpt = int(os.environ.get("DISABLE_GRADIENT_CHECKPOINTING", "0")) == 1
        use_grad_ckpt = training_args.gradient_checkpointing and not disable_grad_ckpt
        
        if use_grad_ckpt:
            rank0_print("=" * 80)
            rank0_print("Enabling gradient checkpointing for Zero3 CPU Offload...")
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            rank0_print("✓ Gradient checkpointing enabled")
            rank0_print("=" * 80)
        elif disable_grad_ckpt:
            rank0_print("=" * 80)
            rank0_print("⚠️  Gradient checkpointing DISABLED by environment variable")
            rank0_print("=" * 80)
        
        # =================================================================
        # 这一步至关重要：告诉模型，即使分布在不同卡上，也要传递梯度
        # =================================================================
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            rank0_print("✓ Input gradients enabled via enable_input_require_grads()")
        else:
            def make_inputs_require_grad(module, input, output):
                if isinstance(output, torch.Tensor):
                    if not output.requires_grad:
                        output.requires_grad_(True)
                elif isinstance(output, (tuple, list)):
                    for item in output:
                        if isinstance(item, torch.Tensor) and not item.requires_grad:
                            item.requires_grad_(True)
                return output
            
            if hasattr(model, "get_input_embeddings"):
                embeddings = model.get_input_embeddings()
                if embeddings is not None:
                    embeddings.register_forward_hook(make_inputs_require_grad)
                    rank0_print("✓ Input gradients enabled via forward hook")
        
        # Additional hook to ensure inputs have requires_grad=True for gradient checkpointing
        # This is needed because enable_input_require_grads() might not work correctly with gradient checkpointing
        if use_grad_ckpt:
            def make_inputs_require_grad_robust(module, input, output):
                if isinstance(output, torch.Tensor):
                    if not output.requires_grad:
                        output.requires_grad_(True)
                elif isinstance(output, (tuple, list)):
                    for item in output:
                        if isinstance(item, torch.Tensor) and not item.requires_grad:
                            item.requires_grad_(True)
                return output
            
            # Try multiple embedding locations
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                model.model.embed_tokens.register_forward_hook(make_inputs_require_grad_robust)
                rank0_print("✓ Additional gradient hook on model.model.embed_tokens")
            elif hasattr(model, "language_model") and hasattr(model.language_model, "embed_tokens"):
                model.language_model.embed_tokens.register_forward_hook(make_inputs_require_grad_robust)
                rank0_print("✓ Additional gradient hook on language_model.embed_tokens")
            elif hasattr(model, "get_input_embeddings"):
                embeddings = model.get_input_embeddings()
                if embeddings is not None:
                    embeddings.register_forward_hook(make_inputs_require_grad_robust)
                    rank0_print("✓ Additional input gradient hook registered for gradient checkpointing")
        
        # 验证一下是否开启成功
        if hasattr(model, "is_gradient_checkpointing"):
            rank0_print(f"Gradient Checkpointing Enabled: {model.is_gradient_checkpointing}")
        else:
            # For some model architectures, check the language model
            if hasattr(model, "language_model") and hasattr(model.language_model, "is_gradient_checkpointing"):
                rank0_print(f"Gradient Checkpointing Enabled: {model.language_model.is_gradient_checkpointing}")
            else:
                rank0_print("⚠️  Could not verify gradient checkpointing status")
        
        # Print memory after setting trainable parameters
        print_gpu_memory_details(model=model, stage="After Setting Trainable Parameters", print_model_details=True)
    
    # Collect images - 直接从 CSV 提取并检查哪些图片实际存在
    # 只训练多模态数据，因此只保留能找到图片的条目
    rank0_print("Collecting image IDs from CSV and checking which images exist locally...")
    
    if image_ids_file and os.path.exists(image_ids_file):
        # 如果提供了 image_ids_file，使用它（但仍需要检查图片是否实际存在）
        image_mapping = download_images_if_needed(
            image_ids_file=image_ids_file,
            image_dir=image_dir,
        )
    else:
        # 直接从 CSV 提取 image IDs（更简单直接，不需要临时文件）
        rank0_print("Extracting image IDs from CSV...")
        image_ids = set()
        for chunk in pd.read_csv(csv_path, chunksize=1000):
            # 只提取非空的 image_payload_id
            valid_image_ids = chunk[chunk['image_payload_id'].notna()]['image_payload_id'].astype(str).str.strip()
            # 过滤空字符串
            valid_image_ids = valid_image_ids[valid_image_ids != '']
            image_ids.update(valid_image_ids.tolist())
        
        rank0_print(f"Found {len(image_ids)} unique image IDs in CSV")
        
        # 直接检查哪些图片存在，不创建临时文件
        image_dir_path = Path(image_dir)
        image_dir_path.mkdir(parents=True, exist_ok=True)
        
        image_mapping = {}
        # Supported image formats
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        for ext in image_extensions:
            for img_file in image_dir_path.glob(ext):
                image_id = img_file.stem
                if image_id in image_ids:
                    image_mapping[image_id] = str(img_file.absolute())
        
        rank0_print(f"Found {len(image_mapping)} existing images out of {len(image_ids)} needed in CSV")
        
        missing_ids = image_ids - set(image_mapping.keys())
        if missing_ids:
            rank0_print(
                f"Warning: {len(missing_ids)} images from CSV are missing locally. These rows will be skipped (multimodal only)."
            )
        
        # Persist mapping for next run
        mapping_file = image_dir_path / "image_mapping.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(image_mapping, f, ensure_ascii=False, indent=2)
        rank0_print(f"Saved {len(image_mapping)} image mappings to {mapping_file}")
    
    # Convert CSV to training data (with token length calculation for sorting)
    rank0_print("Converting CSV to training format...")
    training_data = convert_csv_to_training_data(
        csv_path=csv_path,
        image_mapping=image_mapping,
        tokenizer=tokenizer,
        output_jsonl=output_jsonl,
        chunk_size=1000
    )
    
    # Update processor pixels based on data args
    processor = update_processor_pixels(processor, data_args)
    train_dataset = BrowserUseDataset(processor, data_args, training_data, image_dir=image_dir)
    
    # Print memory after dataset creation
    print_gpu_memory_details(model=model, stage="After Dataset Creation", print_model_details=False)
    
    # Create data collator
    if data_args.data_flatten or data_args.data_packing:
        from qwenvl.data.data_processor import FlattenedDataCollatorForSupervisedDataset
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
    else:
        from qwenvl.data.data_processor import DataCollatorForSupervisedDataset
        data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    
    data_module = {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator
    }
    
    # Critical check: Verify dataset size before training
    print(f"\n{'='*80}")
    print(f"[PRE-TRAINING VERIFICATION]")
    print(f"Dataset length: {len(train_dataset)}")
    print(f"Expected steps per epoch: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)}")
    print(f"Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"World size (num GPUs): {training_args.world_size}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Total expected steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size) * training_args.num_train_epochs}")
    print(f"{'='*80}\n")
    
    # ⚡️ 2. 实例化自定义回调 ⚡️
    # 设置每隔10步打印一次详细内存信息，每隔50步清理一次内存
    memory_callback = MemoryClearCallback(print_interval=10, cleanup_interval=50)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Reach training code")
    print_gpu_memory_details(model=model, stage="Before Trainer Creation", print_model_details=False)
    
    # ⚠️ Final check: Ensure quantization_config is removed before Trainer creation
    if hasattr(model.config, 'quantization_config'):
        if model.config.quantization_config is not None:
            rank0_print("=" * 80)
            rank0_print("⚠️  WARNING: quantization_config still exists! Deleting now...")
            delattr(model.config, 'quantization_config')
            rank0_print("✓ quantization_config attribute deleted")
            rank0_print("=" * 80)
        else:
            # If it exists but is None, delete it anyway to be safe
            delattr(model.config, 'quantization_config')
            rank0_print("✓ Final check: quantization_config attribute deleted (was None)")
    else:
        rank0_print("✓ Final check: quantization_config attribute does not exist (OK for training)")
    
    # CRITICAL: Final synchronization before Trainer creation for ZeRO-3 + LoRA
    # This is the last chance to ensure model structure consistency before DeepSpeed initialization
    if torch.distributed.is_initialized() and training_args.lora_enable and training_args.deepspeed:
        rank0_print("=" * 80)
        rank0_print("Final synchronization before Trainer creation (ZeRO-3 + LoRA)...")
        
        # Multiple synchronization barriers
        for i in range(2):
            torch.distributed.barrier()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Verify module count one more time
        try:
            module_count = len(list(model.named_modules()))
            module_counts = [0] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(module_counts, module_count)
            
            if len(set(module_counts)) > 1:
                rank0_print("=" * 80)
                rank0_print("⚠️  CRITICAL: Module count mismatch before Trainer creation!")
                rank0_print(f"   Module counts: {module_counts}")
                rank0_print("   This will likely cause ZeRO-3 initialization to fail.")
                rank0_print("   Attempting one more synchronization...")
                rank0_print("=" * 80)
                
                # Force all ranks to wait and sync
                torch.distributed.barrier()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.distributed.barrier()
            else:
                rank0_print(f"✓ Module count verified: {module_counts[0]} modules on all ranks")
        except Exception as e:
            rank0_print(f"⚠️  Could not verify module count: {e}")
        
        torch.distributed.barrier()
        rank0_print("✓ All ranks synchronized - proceeding to Trainer creation")
        rank0_print("=" * 80)
    
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, callbacks=[memory_callback], **data_module
    )
    
    # Print memory after trainer creation (optimizer states are created here)
    print_gpu_memory_details(model=model, stage="After Trainer Creation (Optimizer States Created)", print_model_details=False)

    # Smart checkpoint resumption logic
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    should_resume = False
    resume_checkpoint = None
    
    if checkpoint_dirs:
        # Find the latest checkpoint
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.name.split("-")[-1]))
        latest_checkpoint = checkpoint_dirs[-1]
        
        rank0_print(f"\n{'='*80}")
        rank0_print(f"Found existing checkpoint: {latest_checkpoint}")
        
        # Check if checkpoint is valid and compatible with current DeepSpeed config
        # For Zero Stage 3, we need specific files
        if training_args.deepspeed:
            # Check if it's a valid DeepSpeed Zero3 checkpoint
            zero_checkpoint_dir = latest_checkpoint / "zero"
            if zero_checkpoint_dir.exists():
                # Check for rank-specific files
                rank_files = list(zero_checkpoint_dir.glob("*_optim_states.pt"))
                if len(rank_files) > 0:
                    rank0_print(f"✓ Valid DeepSpeed checkpoint found with {len(rank_files)} rank file(s)")
                    should_resume = True
                    resume_checkpoint = str(latest_checkpoint)
                else:
                    rank0_print(f"⚠ Warning: Checkpoint exists but missing optimizer state files")
                    rank0_print(f"   This may be an incomplete or incompatible checkpoint (e.g., Zero2 → Zero3)")
                    rank0_print(f"   Starting training from scratch...")
            else:
                rank0_print(f"⚠ Warning: Checkpoint missing 'zero' directory (incompatible format)")
                rank0_print(f"   This is likely a Zero2 checkpoint, but we're using Zero3")
                rank0_print(f"   Starting training from scratch...")
        else:
            # Non-DeepSpeed checkpoint
            if (latest_checkpoint / "pytorch_model.bin").exists() or \
               (latest_checkpoint / "model.safetensors").exists():
                should_resume = True
                resume_checkpoint = str(latest_checkpoint)
            else:
                rank0_print(f"⚠ Warning: Checkpoint incomplete, starting from scratch...")
        
        rank0_print(f"{'='*80}\n")
    
    if should_resume:
        rank0_print(f"🔄 Resuming training from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        rank0_print("🚀 Starting training from scratch")
        trainer.train()
    
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")


