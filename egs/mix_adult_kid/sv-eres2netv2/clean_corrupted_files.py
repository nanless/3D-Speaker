#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Clean corrupted audio files.')
    parser.add_argument('--missing_files_json', type=str, required=True,
                        help='JSON file containing missing files information')
    parser.add_argument('--min_file_size', type=int, default=1024,
                        help='Minimum file size in bytes to consider valid')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    
    return parser.parse_args()

def load_missing_files(json_file):
    """Load missing files from JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['missing_files']

def clean_corrupted_files(missing_files, min_file_size, dry_run=False):
    """Clean corrupted audio files."""
    deleted_files = []
    skipped_files = []
    
    for file_info in missing_files:
        file_path = Path(file_info['path'])
        
        if not file_path.exists():
            skipped_files.append({
                'path': str(file_path),
                'reason': 'File not found'
            })
            continue
        
        file_size = file_path.stat().st_size
        
        if file_size < min_file_size:
            if dry_run:
                print(f"[DRY RUN] Would delete: {file_path} (size: {file_size} bytes)")
                deleted_files.append({
                    'path': str(file_path),
                    'size': file_size,
                    'reason': f'File too small (<{min_file_size} bytes)'
                })
            else:
                try:
                    file_path.unlink()
                    print(f"Deleted: {file_path} (size: {file_size} bytes)")
                    deleted_files.append({
                        'path': str(file_path),
                        'size': file_size,
                        'reason': f'File too small (<{min_file_size} bytes)'
                    })
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                    skipped_files.append({
                        'path': str(file_path),
                        'reason': f'Deletion error: {e}'
                    })
        else:
            skipped_files.append({
                'path': str(file_path),
                'reason': f'File size OK ({file_size} bytes)'
            })
    
    return deleted_files, skipped_files

def main():
    args = parse_args()
    
    print("Starting corrupted file cleanup...")
    print(f"Missing files JSON: {args.missing_files_json}")
    print(f"Minimum file size: {args.min_file_size} bytes")
    print(f"Dry run: {args.dry_run}")
    
    # Load missing files
    missing_files = load_missing_files(args.missing_files_json)
    print(f"Found {len(missing_files)} missing files to check")
    
    # Clean corrupted files
    deleted_files, skipped_files = clean_corrupted_files(
        missing_files, args.min_file_size, args.dry_run
    )
    
    # Report results
    print(f"\n=== CLEANUP RESULTS ===")
    print(f"Files processed: {len(missing_files)}")
    print(f"Files deleted: {len(deleted_files)}")
    print(f"Files skipped: {len(skipped_files)}")
    
    if deleted_files:
        print(f"\nDeleted files:")
        for file_info in deleted_files:
            print(f"  {file_info['path']} - {file_info['reason']}")
    
    if skipped_files:
        print(f"\nSkipped files:")
        for file_info in skipped_files[:10]:  # Show first 10
            print(f"  {file_info['path']} - {file_info['reason']}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
    
    # Save results
    results = {
        'summary': {
            'total_files': len(missing_files),
            'deleted_files': len(deleted_files),
            'skipped_files': len(skipped_files),
            'dry_run': args.dry_run
        },
        'deleted_files': deleted_files,
        'skipped_files': skipped_files
    }
    
    results_file = 'cleanup_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    if args.dry_run:
        print("\nThis was a dry run. No files were actually deleted.")
        print("Run without --dry_run to actually delete the files.")

if __name__ == "__main__":
    main() 