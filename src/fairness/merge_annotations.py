# -*- coding: utf-8 -*-
"""
merge_annotations.py
====================

Purpose
-------
Merge the original and supplementary gold standard annotations into a single file.

Usage
-----
python -m src.fairness.merge_annotations
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config

def main():
    # Load config for paths
    CONFIG = load_config()
    OUT_DIR = Path(CONFIG['paths']['outputs'])
    GOLD_DIR = OUT_DIR / "data" / "gold"
    
    # File paths
    original_file = GOLD_DIR / "gold_labels_annotator_louise.csv"
    supplement_file = GOLD_DIR / "gold_supplement_20250909.csv"
    
    print("--- Merging Gold Standard Annotations ---")
    
    # Load both files
    print(f"Loading original annotations: {original_file}")
    original_df = pd.read_csv(original_file)
    print(f"  → Loaded {len(original_df)} videos")
    
    print(f"Loading supplementary annotations: {supplement_file}")
    supplement_df = pd.read_csv(supplement_file)
    print(f"  → Loaded {len(supplement_df)} videos")
    
    # Combine the dataframes
    combined_df = pd.concat([original_df, supplement_df], axis=0, ignore_index=True)
    
    # Check for any duplicate video_ids (shouldn't be any)
    duplicates = combined_df['video_id'].duplicated().sum()
    if duplicates > 0:
        print(f"⚠ Warning: Found {duplicates} duplicate video_ids, removing...")
        combined_df = combined_df.drop_duplicates('video_id', keep='first')
    
    # Save the combined file
    output_file = GOLD_DIR / "gold_labels_annotator_louise_complete.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Combined annotations saved: {output_file}")
    print(f"  Total videos: {len(combined_df)}")
    
    # Show distribution statistics
    print("\n--- Annotation Statistics ---")
    print(f"Amateur distribution:")
    print(combined_df['is_amateur_gold'].value_counts())
    print(f"\nGroup distribution:")
    print(combined_df['group_gold'].value_counts())
    
    print("\n✓ Merge complete! Next step: Run finalization")
    print(f"  python -m src.fairness.27_ground_truth --finalize \"{GOLD_DIR}/gold_labels_annotator_louise_complete.csv\"")

if __name__ == "__main__":
    main()