#!/usr/bin/env python3

import json
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image  # type: ignore

def parse_label_line(line):
    """Parse a single line from CEGD-R label file"""
    parts = line.strip().split(',')
    if len(parts) < 9:
        return None
    
    # Extract coordinates (8 values for 4 points)
    coords = [int(x) for x in parts[:8]]
    # Text is everything after the 8th comma
    text = ','.join(parts[8:]) if len(parts) > 8 else parts[8]
    
    return coords, text

def convert_cegdr_to_mmocr(cegdr_root='data/CEGD-R', output_dir='data/CEGD-R_MMOCR', split_ratio=0.8, last_recog_only=False):
    """Convert CEGD-R dataset to MMOCR format"""
    cegdr_root = Path(cegdr_root)
    output_dir = Path(output_dir)
    
    # Create output directories
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # Find all label files
    label_files = sorted(list((cegdr_root / 'Labels').glob('gt_eartags*.txt')))
    print(f"Found {len(label_files)} label files")
    
    records = []
    
    for label_file in tqdm(label_files, desc="Converting files"):
        # Get corresponding image file
        img_name = label_file.name.replace('gt_', '').replace('.txt', '.jpg')
        img_path = cegdr_root / 'Image' / img_name
        
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found, skipping")
            continue
        
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception as e:
            print(f"Error: cannot open {img_path}: {e}")
            continue
        
        # Copy image to output directory
        dst_img_path = output_dir / 'images' / img_name
        if not dst_img_path.exists():
            try:
                dst_img_path.symlink_to(img_path.resolve())
            except:
                # If symlink fails, copy the file
                import shutil
                shutil.copy2(img_path, dst_img_path)
        
        # Parse annotations
        annotations = []
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        result = parse_label_line(line)
                        if result is None:
                            print(f"Warning: Could not parse line {line_num} in {label_file}")
                            continue
                        
                        coords, text = result
                        
                        # Convert to polygon format (list of x,y coordinates)
                        polygon = coords  # Already in the correct format
                        
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]
                        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        annotations.append({
                            "polygon": polygon,
                            "bbox": bbox,
                            "bbox_label": 0,
                            "text": text,
                            "ignore": False
                        })
                        
                    except Exception as e:
                        print(f"Error parsing line {line_num} in {label_file}: {e}")
                        continue
        
        # Filter to last annotation only if requested
        if last_recog_only and annotations:
            annotations = [annotations[-1]]

        records.append({
            "img_path": f"images/{img_name}",
            "height": height,
            "width": width,
            "instances": annotations
        })
    
    # Split into train and test
    split_idx = int(len(records) * split_ratio)
    train_records = records[:split_idx]
    test_records = records[split_idx:]
    
    # Save annotations
    train_file = output_dir / 'annotations' / 'train.json'
    test_file = output_dir / 'annotations' / 'test.json'
    
    meta = {
        "dataset_type": "TextDetDataset",
        "task_name": "textdet",
        "category": [{"id": 0, "name": "text"}]
    }
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump({"metainfo": meta, "data_list": train_records}, f, indent=2, ensure_ascii=False)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump({"metainfo": meta, "data_list": test_records}, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed!")
    print(f"Training samples: {len(train_records)}")
    print(f"Test samples: {len(test_records)}")
    print(f"Output directory: {output_dir}")
    print(f"Training annotations: {train_file}")
    print(f"Test annotations: {test_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert CEGD-R dataset to MMOCR format')
    parser.add_argument('--input', '-i', required=False, 
                       help='Path to CEGD-R dataset root (contains Images and Labels folders)', default='data/CEGD-R')
    parser.add_argument('--output', '-o', required=False,
                       help='Output directory for converted dataset', default='data/CEGD-R_MMOCR')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                       help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--last-recog-only', action='store_true',
                       help='Keep only the last recognition instance per image')
    
    args = parser.parse_args()
    
    convert_cegdr_to_mmocr(args.input, args.output, args.split_ratio, args.last_recog_only)

if __name__ == '__main__':
    main() 