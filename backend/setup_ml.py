
import argparse
import json
import os
import torch
from tqdm import tqdm
import requests

# Configuration for model downloads
MODELS = {
    "yolov5s": {
        "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
        "dest": "models/yolov5s.pt"
    },
    # Add other models here if needed
}

DEFAULT_CONFIG = {
    "model_path": "models/yolov5s.pt",
    "confidence_threshold": 0.4,
    "iou_threshold": 0.5,
    "device": "auto"  # auto, cpu, cuda:0, etc.
}

def download_file(url, dest_path):
    """Downloads a file with a progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

def setup_config(config_path="config.json"):
    """Creates a default config.json if it doesn't exist."""
    if os.path.exists(config_path):
        print(f"'{config_path}' already exists. Skipping creation.")
        return
    
    print(f"Creating default config at '{config_path}'...")
    with open(config_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)

def test_install():
    """Tests core library imports and checks for a CUDA device."""
    print("--- Running Installation Test ---")
    try:
        import cv2
        import torch
        import torchvision
        print("✅ OpenCV, PyTorch, and Torchvision imported successfully.")
    except ImportError as e:
        print(f"❌ Failed to import a core library: {e}")
        return

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    if cuda_available:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print("---------------------------------")


def main():
    parser = argparse.ArgumentParser(description="ML Model Setup for Industrial Monitor")
    parser.add_argument("--download", type=str, choices=MODELS.keys(), help="Download a specific model.")
    parser.add_argument("--config", action="store_true", help="Generate the default config.json.")
    parser.add_argument("--test", action="store_true", help="Run installation and environment tests.")
    
    args = parser.parse_args()

    if args.config:
        setup_config()
        
    if args.download:
        model_name = args.download
        if model_name in MODELS:
            print(f"Downloading model: {model_name}...")
            download_file(MODELS[model_name]['url'], MODELS[model_name]['dest'])
            print("Download complete.")
        else:
            print(f"Error: Model '{model_name}' not found in configuration.")

    if args.test:
        test_install()

    if not any(vars(args).values()):
        parser.print_help()
        print("\nNo arguments provided. Use --help to see options.")

if __name__ == "__main__":
    main()
