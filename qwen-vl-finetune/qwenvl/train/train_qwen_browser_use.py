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
    
    if not training_data:
        rank0_print("Warning: No valid training data found!")
        return training_data
    
    # Sort by token length (ascending) to avoid OOM
    rank0_print("Sorting data by token length (ascending)...")
    training_data.sort(key=lambda x: x.get("num_tokens", 0))
    rank0_print(f"Token length range: {training_data[0].get('num_tokens', 0)} - {training_data[-1].get('num_tokens', 0)} tokens")
    
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
    
    # 判断模型类型
    model_path_lower = model_args.model_name_or_path.lower()
    is_moe_model = "qwen3" in model_path_lower and ("a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower() or "a3b" in model_path_lower)
    is_fp8_model = "fp8" in model_path_lower
    # 检查模型大小：30B, 32B等大模型需要量化（但FP8模型已经是预量化的，不需要）
    is_large_model = any(size in model_path_lower for size in ["30b", "32b", "70b", "235b"])
    use_quantization = (is_moe_model or is_large_model) and not is_fp8_model
    
    # 量化配置：仅对非FP8的大模型使用，小模型（如4B）使用BF16/FP16
    # 从环境变量读取量化位数，默认为 8-bit
    quantization_bits = int(os.environ.get("QUANTIZATION_BITS", "8"))
    rank0_print(f"Quantization bits from env: {quantization_bits}")
    
    if use_quantization:
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
    
    # Load model
    if is_moe_model:
        if is_fp8_model:
            # FP8预量化MoE模型（如Qwen3-VL-30B-A3B-Instruct-FP8）直接加载，不需要额外量化
            rank0_print(f"Loading FP8 pre-quantized MoE model: {model_args.model_name_or_path}")
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        else:
            # MoE模型（如qwen3-vl-30B-A3B）使用量化配置以节省内存
            rank0_print(f"Loading MoE model with {quantization_bits}-bit quantization: {model_args.model_name_or_path}")
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
            )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_path_lower:
        # 普通qwen3模型：FP8模型直接加载，大模型使用4-bit量化，小模型（如4B）使用BF16/FP16
        if is_fp8_model:
            rank0_print(f"Loading FP8 pre-quantized Qwen3 model: {model_args.model_name_or_path}")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        elif use_quantization:
            rank0_print(f"Loading large Qwen3 model with {quantization_bits}-bit quantization: {model_args.model_name_or_path}")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
            )
        else:
            rank0_print(f"Loading Qwen3 model with BF16/FP16 (no quantization): {model_args.model_name_or_path}")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    rank0_print(f'the initialized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    
    # Print memory after model loading
    print_gpu_memory_details(model=model, stage="After Model Loading", print_model_details=True)
    
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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

        # 对于量化模型（4-bit 或 8-bit），需要先调用 prepare_model_for_kbit_training
        if use_quantization:
            rank0_print("Preparing quantized model for k-bit training...")
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_args.gradient_checkpointing,
            )

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        
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
    
    # ⚡️ 2. 实例化自定义回调 ⚡️
    # 设置每隔10步打印一次详细内存信息，每隔50步清理一次内存
    memory_callback = MemoryClearCallback(print_interval=10, cleanup_interval=50)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Reach training code")
    print_gpu_memory_details(model=model, stage="Before Trainer Creation", print_model_details=False)
    
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, callbacks=[memory_callback], **data_module
    )
    
    # Print memory after trainer creation (optimizer states are created here)
    print_gpu_memory_details(model=model, stage="After Trainer Creation (Optimizer States Created)", print_model_details=False)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

