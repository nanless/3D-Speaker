#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Check for missing embedding files.')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing audio files')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing individual embedding files')
    parser.add_argument('--output_file', type=str, default='missing_files.json',
                        help='Output file for missing files list')
    parser.add_argument('--audio_extensions', nargs='+', default=['.wav', '.flac', '.mp3'],
                        help='Audio file extensions to check')
    
    return parser.parse_args()

def scan_audio_files(data_root, audio_extensions):
    """Scan for all audio files in the data directory."""
    audio_files = []
    
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            
            for audio_file in speaker_dir.iterdir():
                if audio_file.suffix.lower() in audio_extensions:
                    audio_files.append({
                        'path': str(audio_file),
                        'dataset': dataset_name,
                        'speaker_id': speaker_id,
                        'utterance_id': audio_file.stem,
                        'relative_path': str(audio_file.relative_to(data_root))
                    })
    
    return audio_files

def scan_embedding_files(embeddings_dir):
    """Scan for all existing embedding files."""
    embedding_files = set()
    
    utterances_dir = os.path.join(embeddings_dir, 'utterances')
    if os.path.exists(utterances_dir):
        for pkl_file in Path(utterances_dir).rglob('*.pkl'):
            # Extract info from path: utterances/dataset/speaker_id/utterance_id.pkl
            parts = pkl_file.relative_to(utterances_dir).parts
            if len(parts) == 3:
                dataset, speaker_id, filename = parts
                utterance_id = Path(filename).stem
                key = f"{dataset}_{speaker_id}_{utterance_id}"
                embedding_files.add(key)
    
    return embedding_files

def find_missing_files(audio_files, embedding_files):
    """Find audio files that don't have corresponding embedding files."""
    missing_files = []
    
    for audio_info in audio_files:
        expected_key = f"{audio_info['dataset']}_{audio_info['speaker_id']}_{audio_info['utterance_id']}"
        if expected_key not in embedding_files:
            missing_files.append(audio_info)
    
    return missing_files

def analyze_missing_patterns(missing_files):
    """Analyze patterns in missing files."""
    patterns = {
        'by_dataset': defaultdict(int),
        'by_speaker': defaultdict(int),
        'by_extension': defaultdict(int),
        'by_file_size': defaultdict(int)
    }
    
    for file_info in missing_files:
        patterns['by_dataset'][file_info['dataset']] += 1
        patterns['by_speaker'][f"{file_info['dataset']}_{file_info['speaker_id']}"] += 1
        
        # Get file extension and size
        file_path = Path(file_info['path'])
        if file_path.exists():
            patterns['by_extension'][file_path.suffix.lower()] += 1
            file_size = file_path.stat().st_size
            if file_size == 0:
                patterns['by_file_size']['empty'] += 1
            elif file_size < 1024:  # < 1KB
                patterns['by_file_size']['very_small'] += 1
            elif file_size < 10240:  # < 10KB
                patterns['by_file_size']['small'] += 1
            else:
                patterns['by_file_size']['normal'] += 1
        else:
            patterns['by_file_size']['not_found'] += 1
    
    return patterns

def main():
    args = parse_args()
    
    print("Checking for missing embedding files...")
    print(f"Data root: {args.data_root}")
    print(f"Embeddings dir: {args.embeddings_dir}")
    
    # Scan audio files
    print("Scanning audio files...")
    audio_files = scan_audio_files(args.data_root, args.audio_extensions)
    print(f"Found {len(audio_files)} audio files")
    
    # Scan embedding files
    print("Scanning embedding files...")
    embedding_files = scan_embedding_files(args.embeddings_dir)
    print(f"Found {len(embedding_files)} embedding files")
    
    # Find missing files
    print("Finding missing files...")
    missing_files = find_missing_files(audio_files, embedding_files)
    print(f"Found {len(missing_files)} missing embedding files")
    
    if len(missing_files) == 0:
        print("All audio files have corresponding embedding files!")
        return
    
    # Analyze patterns
    print("Analyzing missing file patterns...")
    patterns = analyze_missing_patterns(missing_files)
    
    # Report results
    print(f"\n=== MISSING FILES ANALYSIS ===")
    print(f"Total missing files: {len(missing_files)}")
    print(f"Success rate: {(len(audio_files) - len(missing_files)) / len(audio_files) * 100:.2f}%")
    
    print(f"\nMissing files by dataset:")
    for dataset, count in sorted(patterns['by_dataset'].items()):
        print(f"  {dataset}: {count}")
    
    print(f"\nMissing files by file size:")
    for size_cat, count in sorted(patterns['by_file_size'].items()):
        print(f"  {size_cat}: {count}")
    
    print(f"\nMissing files by extension:")
    for ext, count in sorted(patterns['by_extension'].items()):
        print(f"  {ext}: {count}")
    
    # Show top speakers with missing files
    speaker_counts = patterns['by_speaker']
    if speaker_counts:
        print(f"\nTop speakers with missing files:")
        sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)
        for speaker, count in sorted_speakers[:10]:
            print(f"  {speaker}: {count}")
    
    # Show sample missing files
    print(f"\nSample missing files:")
    for i, file_info in enumerate(missing_files[:10]):
        file_path = Path(file_info['path'])
        size_info = ""
        if file_path.exists():
            size = file_path.stat().st_size
            size_info = f" (size: {size} bytes)"
        else:
            size_info = " (FILE NOT FOUND)"
        
        print(f"  {i+1}. {file_info['relative_path']}{size_info}")
    
    if len(missing_files) > 10:
        print(f"  ... and {len(missing_files) - 10} more")
    
    # Save detailed results
    results = {
        'summary': {
            'total_audio_files': len(audio_files),
            'total_embedding_files': len(embedding_files),
            'missing_files': len(missing_files),
            'success_rate': (len(audio_files) - len(missing_files)) / len(audio_files)
        },
        'missing_files': missing_files,
        'patterns': {
            'by_dataset': dict(patterns['by_dataset']),
            'by_speaker': dict(patterns['by_speaker']),
            'by_extension': dict(patterns['by_extension']),
            'by_file_size': dict(patterns['by_file_size'])
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output_file}")
    
    # Create script for re-extraction
    script_content = f"""#!/bin/bash
# Script to re-extract missing embeddings

DATA_ROOT="{args.data_root}"
MODEL_PATH="/root/workspace/speaker_verification/mix_adult_kid/exp/eres2netv2_lm/models/CKPT-EPOCH-9-00/embedding_model.ckpt"
CONFIG_FILE="conf/eres2netv2_lm.yaml"
OUTPUT_DIR="{os.path.dirname(args.embeddings_dir)}"

echo "Re-extracting {len(missing_files)} missing embeddings..."

python extract_missing_embeddings.py \\
    --missing_files_json "{args.output_file}" \\
    --model_path "$MODEL_PATH" \\
    --config_file "$CONFIG_FILE" \\
    --output_dir "$OUTPUT_DIR" \\
    --use_gpu \\
    --gpu 0

echo "Re-extraction completed."
"""
    
    script_file = 'reextract_missing.sh'
    with open(script_file, 'w') as f:
        f.write(script_content)
    os.chmod(script_file, 0o755)
    
    print(f"Re-extraction script created: {script_file}")
    print(f"Run: ./{script_file}")

if __name__ == "__main__":
    main() 