#!/usr/bin/env python3
"""
Download UI-TARS-7B-SFT model from HuggingFace
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
from tqdm import tqdm


def download_model(
    repo_id: str = "ByteDance-Seed/UI-TARS-7B-SFT",
    local_dir: str = "./models/UI-TARS-7B-SFT",
    use_auth_token: bool = False,
):
    """
    Download model from HuggingFace Hub
    
    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to save the model
        use_auth_token: Whether to use HuggingFace token for authentication
    """
    
    print(f"🚀 Starting download of {repo_id}")
    print(f"📁 Target directory: {local_dir}")
    
    # Create directory if not exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Login if token is needed (for gated models)
    if use_auth_token:
        print("🔐 Authenticating with HuggingFace...")
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token)
        else:
            print("⚠️  HF_TOKEN not found in environment variables")
            print("💡 Run: huggingface-cli login")
            response = input("Do you want to login now? (y/n): ")
            if response.lower() == 'y':
                login()
            else:
                print("❌ Exiting. Please set HF_TOKEN or login first.")
                sys.exit(1)
    
    try:
        # Download the model
        print("\n📥 Downloading model files...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=True if use_auth_token else None,
            ignore_patterns=[
                "*.git*",
                "*.md",
                "*.txt",
                "*.json.bak"
            ],
            max_workers=8,
        )
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"📁 Location: {Path(local_dir).absolute()}")
        print(f"\n💡 You can now use this model with:")
        print(f"   model_name_or_path=\"{local_dir}\"")
        
        return local_dir
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Verify the repository exists: https://huggingface.co/ByteDance-Seed/UI-TARS-7B-SFT")
        print("   3. If it's a gated model, run with --auth flag and login")
        print("   4. Try: huggingface-cli login")
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download UI-TARS-7B-SFT model from HuggingFace"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ByteDance-Seed/UI-TARS-7B-SFT",
        help="HuggingFace repository ID (default: ByteDance-Seed/UI-TARS-7B-SFT)"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./models/UI-TARS-7B-SFT",
        help="Local directory to save the model (default: ./models/UI-TARS-7B-SFT)"
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Use HuggingFace authentication (for gated/private models)"
    )
    
    args = parser.parse_args()
    
    download_model(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        use_auth_token=args.auth
    )


if __name__ == "__main__":
    main()
