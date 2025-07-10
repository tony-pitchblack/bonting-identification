#!/usr/bin/env python3

from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image  # type: ignore
import shutil
import re

# Parsing no longer required for conversion; keep utility for potential future use
def parse_label_line(line):
    parts = line.strip().split(',')
    if len(parts) < 9:
        return None
    try:
        coords = [int(round(float(x))) for x in parts[:8]]
    except ValueError:
        return None
    text = ','.join(parts[8:]) if len(parts) > 8 else parts[8]
    return coords, text

def split_CEGR_dataset(cegdr_root: Path, output_dir: Path, split_ratio: float = 0.8):
    # Ensure we are working with Path objects
    cegdr_root = Path(cegdr_root)
    output_dir = Path(output_dir)
    (output_dir / 'Image' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'Image' / 'test').mkdir(parents=True, exist_ok=True)
    (output_dir / 'Labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'Labels' / 'test').mkdir(parents=True, exist_ok=True)

    label_files = sorted((cegdr_root / 'Labels').glob('gt_eartags*.txt'))
    print(f"Found {len(label_files)} label files")
    processed_train = 0
    processed_test = 0
    for idx, label_file in enumerate(tqdm(label_files, desc="Converting files")):
        img_name = label_file.name.replace('gt_', '').replace('.txt', '.jpg')
        img_path = cegdr_root / 'Image' / img_name
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found, skipping")
            continue

        split = 'train' if idx < int(len(label_files) * split_ratio) else 'test'

        # Handle image
        src_img = img_path
        dst_img = output_dir / 'Image' / split / img_name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(src_img.resolve())
            except Exception:
                shutil.copy2(src_img, dst_img)

        # Handle annotation file
        dst_ann = output_dir / 'Labels' / split / label_file.name
        if not dst_ann.exists():
                try:
                    dst_ann.symlink_to(label_file.resolve())
                except Exception:
                    shutil.copy2(label_file, dst_ann)

        if split == 'train':
            processed_train += 1
        else:
            processed_test += 1

    print("Conversion completed!")
    print(f"Training samples: {processed_train}")
    print(f"Test samples: {processed_test}")
    print(f"Output directory: {output_dir}")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / 'data' / 'CEGD-R'
    default_output = repo_root / 'data' / 'CEGD-R_train_test'

    parser = argparse.ArgumentParser(description='Split CEGD-R dataset into train and test')
    parser.add_argument('--input', '-i', required=False,
                        help='Path to CEGD-R dataset root (contains Image and Labels folders)',
                        default=str(default_input))
    parser.add_argument('--output', '-o', required=False,
                        help='Output directory for converted dataset',
                        default=str(default_output))
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Train/test split ratio (default: 0.8)')
    args = parser.parse_args()

    split_CEGR_dataset(Path(args.input), Path(args.output), args.split_ratio)

if __name__ == '__main__':
    main() 