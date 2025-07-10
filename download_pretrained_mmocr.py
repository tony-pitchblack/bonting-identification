#!/usr/bin/env python3

import argparse
import os
import urllib.request
from pathlib import Path
import sys

CKPT = {
    'ABINet-Vision': 'https://download.openmmlab.com/mmocr/textrecog/abinet/abinet-vision_20e_st-an_mj/abinet-vision_20e_st-an_mj_20220915_152445-85cfb03d.pth',
}

def get_filename_from_url(url):
    return url.split('/')[-1]

def get_model_path(model_name):
    url = CKPT[model_name]
    filename = get_filename_from_url(url)
    return Path('ckpt/pretrained_mmocr') / filename

def download_model(model_name):
    if model_name not in CKPT:
        print(f"Error: Model '{model_name}' not available")
        return False
    
    url = CKPT[model_name]
    model_path = get_model_path(model_name)
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_path.exists():
        print(f"Model '{model_name}' already downloaded at {model_path}")
        return True
    
    print(f"Downloading {model_name}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"Successfully downloaded {model_name} to {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return False

def list_available():
    print("Available models:")
    for model_name in CKPT.keys():
        print(f"  - {model_name}")

def list_downloaded():
    print("Downloaded models:")
    downloaded = []
    for model_name in CKPT.keys():
        model_path = get_model_path(model_name)
        if model_path.exists():
            downloaded.append(model_name)
            print(f"  - {model_name} ({model_path})")
    
    if not downloaded:
        print("  No models downloaded yet")

def main():
    parser = argparse.ArgumentParser(description='Download MMOCR pretrained models')
    parser.add_argument('--list-available', '--list_available', action='store_true', 
                       help='List all available models')
    parser.add_argument('--list-downloaded', '--list_downloaded', action='store_true',
                       help='List downloaded models')
    parser.add_argument('model', nargs='?', 
                       help='Model name to download')
    
    args = parser.parse_args()
    
    if args.list_available:
        list_available()
        return
    
    if args.list_downloaded:
        list_downloaded()
        return
    
    if args.model:
        download_model(args.model)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 