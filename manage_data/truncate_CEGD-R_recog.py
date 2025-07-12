#!/usr/bin/env python3
"""
Truncate CEGD-R text recognition dataset so that all annotations only
contain characters present in the alignment dictionary located at
`../mmocr/dicts/lower_english_digits_space.txt`.

Expected existing structure (already split & prepared via MMOCR):
    data/CEGD-R_train_test/
        textrecog_train.json
        textrecog_test.json
        textrecog_imgs/               # directory with images

The script creates filtered counterparts in the same directory:
    textrecog_train_truncated.json
    textrecog_test_truncated.json
    textrecog_imgs_truncated/        # images referenced by new JSONs

Usage:
    python truncate_CEGD-R_recog.py
"""

import json
import shutil
import os
from pathlib import Path

ALIGNMENT_DICT = "../mmocr/dicts/lower_english_digits_space.txt"
DATA_DIR = Path("data/CEGD-R_train_test")


def load_allowed_chars(dict_path: Path) -> set[str]:
    """Return a set of allowed characters from the dictionary file."""
    with dict_path.open("r", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.rstrip("\n")}


def text_allowed(text: str, allowed: set[str]) -> bool:
    """Check if *all* characters of *text* (lower-cased) are in *allowed*."""
    return all((ch.lower() in allowed) for ch in text)


def filter_data_list(data_list: list, allowed: set[str], imgs_src_root: Path, imgs_dst_root: Path):
    """Yield filtered entries and copy corresponding images.

    The *img_path* string inside each kept entry is rewritten to point to
    the new *imgs_dst_root* directory (keeping the rest of the sub-path).
    """
    kept_entries = []
    for item in data_list:
        texts_ok = True
        for inst in item.get("instances", []):
            if not text_allowed(inst.get("text", ""), allowed):
                texts_ok = False
                break
        if not texts_ok:
            continue

        # Copy image to new directory, preserving sub-folders
        relative_img_path = Path(item["img_path"])  # e.g. textrecog_imgs/train/xxx.jpg

        # Destination img_path: replace leading "textrecog_imgs" with "textrecog_imgs_truncated"
        new_relative = Path(imgs_dst_root.name) / relative_img_path.relative_to("textrecog_imgs")

        src = imgs_src_root / relative_img_path.relative_to("textrecog_imgs")
        dst = imgs_dst_root / relative_img_path.relative_to("textrecog_imgs")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        # Update item img_path and add to list
        item_copy = json.loads(json.dumps(item))  # deep copy via JSON
        item_copy["img_path"] = str(new_relative).replace(os.sep, "/")
        kept_entries.append(item_copy)
    return kept_entries


def process_split(json_path: Path, allowed: set[str]):
    """Process a single split JSON file and write its truncated version."""
    with json_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    imgs_src_root = DATA_DIR / "textrecog_imgs"
    imgs_dst_root = DATA_DIR / "textrecog_imgs_truncated"

    filtered_data = filter_data_list(dataset["data_list"], allowed, imgs_src_root, imgs_dst_root)

    out_path = json_path.with_stem(json_path.stem + "_truncated")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"metainfo": dataset.get("metainfo", {}), "data_list": filtered_data}, f, ensure_ascii=False)

    print(f"{json_path.name}: kept {len(filtered_data)} / {len(dataset['data_list'])} entries → {out_path.name}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    allowed_chars = load_allowed_chars((script_dir / ALIGNMENT_DICT).resolve())

    for split in ("textrecog_train.json", "textrecog_test.json"):
        process_split(DATA_DIR / split, allowed_chars) 