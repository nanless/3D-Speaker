#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm

from speakerlab.utils.builder import build
from speakerlab.utils.utils import get_logger
from speakerlab.utils.config import build_config

def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings for missing files.')
    parser.add_argument('--missing_files_json', type=str, required=True,
                        help='JSON file containing missing files information')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the embedding model checkpoint')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for embeddings')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for extraction')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--skip_corrupted', action='store_true', help='Skip corrupted audio files')
    
    return parser.parse_args()

def load_missing_files(json_file):
    """Load missing files from JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['missing_files']

def load_model_and_config(config_file, model_path, device):
    """Load the embedding model and configuration."""
    config = build_config(config_file)
    
    # Build the embedding model
    embedding_model = build('embedding_model', config)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        embedding_model.load_state_dict(checkpoint['model'])
    else:
        embedding_model.load_state_dict(checkpoint)
    
    embedding_model.to(device)
    embedding_model.eval()
    
    # Build feature extractor
    feature_extractor = build('feature_extractor', config)
    
    return embedding_model, feature_extractor, config

def save_individual_embedding(embedding, file_info, output_dir):
    """Save individual embedding file maintaining directory structure."""
    dataset = file_info['dataset']
    speaker_id = file_info['speaker_id']
    utterance_id = file_info['utterance_id']
    
    # Create directory structure
    base_dir = os.path.join(output_dir, 'embeddings_individual', 'utterances', dataset, speaker_id)
    os.makedirs(base_dir, exist_ok=True)
    
    # Save embedding
    filename = f"{utterance_id}.pkl"
    save_path = os.path.join(base_dir, filename)
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'embedding': embedding,
            'info': file_info
        }, f)
    
    return save_path

def extract_single_embedding(audio_path, embedding_model, feature_extractor, config, device):
    """Extract embedding for a single audio file."""
    try:
        # Load audio
        wav, fs = torchaudio.load(audio_path)
        
        # Ensure sample rate matches
        if fs != config.sample_rate:
            wav = torchaudio.functional.resample(wav, fs, config.sample_rate)
        
        # Extract features
        feat = feature_extractor(wav)
        feat = feat.unsqueeze(0).to(device)
        
        # Extract embedding
        with torch.no_grad():
            emb = embedding_model(feat).detach().cpu().numpy()
        
        return emb.flatten(), None
        
    except Exception as e:
        return None, str(e)

def validate_audio_file(audio_path, min_duration=0.1, max_duration=30.0):
    """Validate audio file before processing."""
    try:
        file_path = Path(audio_path)
        if not file_path.exists():
            return False, "File not found"
        
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "Empty file"
        
        if file_size < 1024:  # Less than 1KB
            return False, f"File too small ({file_size} bytes)"
        
        # Try to load and check duration
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            if duration < min_duration:
                return False, f"Duration too short ({duration:.2f}s)"
            
            if duration > max_duration:
                return False, f"Duration too long ({duration:.2f}s)"
            
            return True, None
            
        except Exception as e:
            return False, f"Audio loading error: {str(e)}"
            
    except Exception as e:
        return False, f"File validation error: {str(e)}"

def main():
    args = parse_args()
    logger = get_logger()
    
    # Setup device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load missing files
    logger.info(f"Loading missing files from: {args.missing_files_json}")
    missing_files = load_missing_files(args.missing_files_json)
    logger.info(f"Found {len(missing_files)} missing files to process")
    
    # Load model and config
    logger.info("Loading model and configuration...")
    embedding_model, feature_extractor, config = load_model_and_config(
        args.config_file, args.model_path, device
    )
    logger.info("Model loaded successfully")
    
    # Process missing files
    successful_extractions = 0
    failed_extractions = []
    skipped_files = []
    
    progress_bar = tqdm(missing_files, desc="Processing missing files")
    
    for file_info in progress_bar:
        audio_path = file_info['path']
        
        # Validate audio file
        is_valid, error_msg = validate_audio_file(audio_path)
        if not is_valid:
            skipped_files.append({
                'file_info': file_info,
                'reason': error_msg
            })
            logger.warning(f"Skipping {audio_path}: {error_msg}")
            continue
        
        # Extract embedding
        embedding, error = extract_single_embedding(
            audio_path, embedding_model, feature_extractor, config, device
        )
        
        if embedding is not None:
            # Save embedding
            save_path = save_individual_embedding(embedding, file_info, args.output_dir)
            successful_extractions += 1
            
            progress_bar.set_postfix({
                'success': successful_extractions,
                'failed': len(failed_extractions),
                'skipped': len(skipped_files)
            })
        else:
            failed_extractions.append({
                'file_info': file_info,
                'error': error
            })
            logger.error(f"Failed to extract embedding for {audio_path}: {error}")
    
    # Report results
    logger.info(f"\n=== EXTRACTION RESULTS ===")
    logger.info(f"Total files processed: {len(missing_files)}")
    logger.info(f"Successful extractions: {successful_extractions}")
    logger.info(f"Failed extractions: {len(failed_extractions)}")
    logger.info(f"Skipped files: {len(skipped_files)}")
    logger.info(f"Success rate: {successful_extractions / len(missing_files) * 100:.2f}%")
    
    # Save detailed results
    results = {
        'summary': {
            'total_files': len(missing_files),
            'successful_extractions': successful_extractions,
            'failed_extractions': len(failed_extractions),
            'skipped_files': len(skipped_files),
            'success_rate': successful_extractions / len(missing_files)
        },
        'failed_extractions': failed_extractions,
        'skipped_files': skipped_files
    }
    
    results_file = 'missing_extraction_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Show samples of failed/skipped files
    if failed_extractions:
        logger.info(f"\nSample failed extractions:")
        for i, item in enumerate(failed_extractions[:5]):
            logger.info(f"  {i+1}. {item['file_info']['relative_path']}")
            logger.info(f"     Error: {item['error']}")
    
    if skipped_files:
        logger.info(f"\nSample skipped files:")
        for i, item in enumerate(skipped_files[:5]):
            logger.info(f"  {i+1}. {item['file_info']['relative_path']}")
            logger.info(f"     Reason: {item['reason']}")
    
    logger.info("Processing completed!")

if __name__ == "__main__":
    main() 