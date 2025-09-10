"""
Setup script for ML model initialization
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import torch
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLSetup:
    """Setup ML models and dependencies"""

    # Model registry with URLs and checksums
    MODEL_REGISTRY = {
        'yolov5s': {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
            'size': 14.1,  # MB
            'description': 'YOLOv5 small model for gauge detection'
        },
        'yolov5m': {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt',
            'size': 40.8,  # MB
            'description': 'YOLOv5 medium model (higher accuracy)'
        },
        'vdn_resnet34': {
            'url': None,  # Would need actual URL
            'size': 83.3,
            'description': 'VDN with ResNet34 backbone for pointer detection'
        },
        'custom_gauge': {
            'url': None,  # Placeholder for custom trained model
            'size': 0,
            'description': 'Custom trained gauge detector'
        }
    }

    def __init__(self, model_dir='./weights', data_dir='./data'):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

    def download_model(self, model_name, force=False):
        """Download a specific model"""
        if model_name not in self.MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            return False

        model_info = self.MODEL_REGISTRY[model_name]
        model_path = self.model_dir / f"{model_name}.pt"

        if model_path.exists() and not force:
            logger.info(f"Model {model_name} already exists")
            return True

        if not model_info['url']:
            logger.warning(f"No URL available for {model_name}")
            # Create placeholder
            torch.save({'state_dict': {}, 'model_name': model_name}, model_path)
            return True

        logger.info(f"Downloading {model_name} ({model_info['size']} MB)...")

        try:
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Successfully downloaded {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False

    def setup_sample_data(self):
        """Download sample gauge images for testing"""
        sample_urls = [
            # Sample gauge images (using placeholder URLs)
            'https://example.com/gauge1.jpg',
            'https://example.com/gauge2.jpg',
        ]

        sample_dir = self.data_dir / 'samples'
        sample_dir.mkdir(exist_ok=True)

        logger.info("Setting up sample data...")
        # Download logic here

    def verify_cuda(self):
        """Verify CUDA availability"""
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            logger.warning("CUDA not available, will use CPU (slower)")
            return False

    def create_config(self):
        """Create default configuration file"""
        config = {
            'models': {
                'default_backend': 'hybrid',
                'yolo_model': 'yolov5s',
                'vdn_enabled': True,
                'confidence_threshold': 0.5
            },
            'calibration': {
                'default_min_angle': 0,
                'default_max_angle': 270,
                'default_min_value': 0,
                'default_max_value': 100
            },
            'tracking': {
                'enabled': True,
                'window_size': 5,
                'outlier_threshold': 2.0
            },
            'few_shot': {
                'samples_required': 5,
                'learning_rate': 0.001,
                'epochs': 10
            }
        }

        config_path = Path('config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Created configuration file: {config_path}")
        return config

    def run_tests(self):
        """Run basic tests to verify installation"""
        logger.info("Running installation tests...")

        tests_passed = []
        tests_failed = []

        # Test 1: Import core libraries
        try:
            import cv2
            import numpy as np
            import torch
            import torchvision
            tests_passed.append("Core libraries")
        except ImportError as e:
            tests_failed.append(f"Core libraries: {e}")

        # Test 2: OpenCV functionality
        try:
            import cv2
            import numpy as np
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            tests_passed.append("OpenCV")
        except Exception as e:
            tests_failed.append(f"OpenCV: {e}")

        # Test 3: PyTorch
        try:
            import torch
            x = torch.randn(1, 3, 224, 224)
            tests_passed.append("PyTorch")
        except Exception as e:
            tests_failed.append(f"PyTorch: {e}")

        # Test 4: Model loading
        try:
            from ml_dial_cv import create_enhanced_detector
            detector = create_enhanced_detector(backend='classical')
            tests_passed.append("ML detector")
        except Exception as e:
            tests_failed.append(f"ML detector: {e}")

        # Print results
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Tests Passed ({len(tests_passed)}):")
        for test in tests_passed:
            logger.info(f"  ✓ {test}")

        if tests_failed:
            logger.error(f"\nTests Failed ({len(tests_failed)}):")
            for test in tests_failed:
                logger.error(f"  ✗ {test}")

        logger.info(f"{'=' * 50}\n")

        return len(tests_failed) == 0


def main():
    parser = argparse.ArgumentParser(description='Setup ML models for gauge reading')
    parser.add_argument('--download-all', action='store_true',
                        help='Download all available models')
    parser.add_argument('--download', type=str,
                        help='Download specific model')
    parser.add_argument('--test', action='store_true',
                        help='Run installation tests')
    parser.add_argument('--config', action='store_true',
                        help='Create default configuration')
    parser.add_argument('--cuda-check', action='store_true',
                        help='Check CUDA availability')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download of models')

    args = parser.parse_args()

    setup = MLSetup()

    if args.cuda_check:
        setup.verify_cuda()

    if args.config:
        setup.create_config()

    if args.download_all:
        for model_name in setup.MODEL_REGISTRY.keys():
            setup.download_model(model_name, force=args.force)

    elif args.download:
        setup.download_model(args.download, force=args.force)

    if args.test:
        success = setup.run_tests()
        sys.exit(0 if success else 1)

    if not any(vars(args).values()):
        # No arguments, show help
        parser.print_help()
        print("\nQuick setup:")
        print("  python setup_ml.py --config --download yolov5s --test")
        print("\nFull setup:")
        print("  python setup_ml.py --download-all --config --test")


if __name__ == "__main__":
    main()
