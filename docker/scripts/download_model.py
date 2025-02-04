#!/usr/bin/env python3
"""Download and prepare Kokoro model for Docker build."""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from loguru import logger


def download_model(version: str, output_dir: str) -> None:
    """Download model files from HuggingFace.
    
    Args:
        version: Model version to download
        output_dir: Directory to save model files
    """
    try:
        logger.info(f"Downloading Kokoro model version {version}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download model files
        model_file = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M",
            filename=f"kokoro-{version}.pth"
        )
        config_file = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M",
            filename="config.json"
        )
        
        # Copy to output directory
        shutil.copy2(model_file, os.path.join(output_dir, "model.pt"))
        shutil.copy2(config_file, os.path.join(output_dir, "config.json"))
        
        # Verify files
        model_path = os.path.join(output_dir, "model.pt")
        config_path = os.path.join(output_dir, "config.json")
        
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise RuntimeError(f"Config file not found: {config_path}")
            
        # Load and verify model
        logger.info("Verifying model files...")
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"Loaded config: {config}")
        
        model = torch.load(model_path, map_location="cpu")
        logger.info(f"Loaded model with keys: {model.keys()}")
        
        logger.info(f"✓ Model files prepared in {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download Kokoro model for Docker build")
    parser.add_argument(
        "--version",
        default="v1_0",
        help="Model version to download"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for model files"
    )
    
    args = parser.parse_args()
    download_model(args.version, args.output)


if __name__ == "__main__":
    main()