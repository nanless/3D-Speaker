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
from pathlib import Path
from collections import defaultdict
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

from speakerlab.utils.builder import build
from speakerlab.utils.utils import get_logger
from speakerlab.utils.config import build_config

def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings for all utterances and compute speaker-level averages.')
    parser.add_argument('--data_root', type=str, 
                        default='/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments',
                        help='Root directory containing dataset_names/speaker_ids/utts structure')
    parser.add_argument('--model_path', type=str,
                        default='/root/workspace/speaker_verification/mix_adult_kid/exp/eres2netv2_lm/models/CKPT-EPOCH-9-00/embedding_model.ckpt',
                        help='Path to the embedding model checkpoint')
    parser.add_argument('--config_file', type=str,
                        default='conf/eres2netv2_lm.yaml',
                        help='Path to the config file')
    parser.add_argument('--output_dir', type=str, default='embeddings_output',
                        help='Output directory for embeddings and metadata')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for extraction')
    parser.add_argument('--gpu', nargs='+', default=['0', '1', '2', '3'], help='GPU ids to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--compute_similarity', action='store_true', help='Compute speaker embedding similarity matrix')
    parser.add_argument('--similarity_threshold', type=float, default=0.8, help='Threshold for identifying similar speakers')
    parser.add_argument('--save_individual', action='store_true', default=True, help='Save individual embedding files')
    
    return parser.parse_args()

def scan_audio_files(data_root):
    """Scan the directory structure to find all audio files."""
    audio_files = []
    dataset_names = []
    
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        dataset_names.append(dataset_name)
        
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            
            for audio_file in speaker_dir.iterdir():
                if audio_file.suffix.lower() in ['.wav', '.flac', '.mp3']:
                    audio_files.append({
                        'path': str(audio_file),
                        'dataset': dataset_name,
                        'speaker_id': speaker_id,
                        'utterance_id': audio_file.stem
                    })
    
    return audio_files, dataset_names

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

def save_individual_embedding(embedding, file_info, output_dir, embedding_type='utterance'):
    """Save individual embedding file maintaining directory structure."""
    dataset = file_info['dataset']
    speaker_id = file_info['speaker_id']
    
    # Create directory structure: output_dir/embeddings_individual/dataset/speaker_id/
    if embedding_type == 'utterance':
        base_dir = os.path.join(output_dir, 'embeddings_individual', 'utterances', dataset, speaker_id)
        filename = f"{file_info['utterance_id']}.pkl"
    else:  # speaker
        base_dir = os.path.join(output_dir, 'embeddings_individual', 'speakers', dataset)
        filename = f"{speaker_id}.pkl"
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Save embedding
    save_path = os.path.join(base_dir, filename)
    with open(save_path, 'wb') as f:
        pickle.dump({
            'embedding': embedding,
            'info': file_info
        }, f)
    
    return save_path

def extract_embeddings_batch(audio_files_batch, embedding_model, feature_extractor, config, device, output_dir, save_individual, rank, logger):
    """Extract embeddings for a batch of audio files."""
    embeddings = []
    file_info = []
    
    with torch.no_grad():
        for file_info_item in audio_files_batch:
            try:
                wav_path = file_info_item['path']
                wav, fs = torchaudio.load(wav_path)
                
                # Ensure sample rate matches
                if fs != config.sample_rate:
                    wav = torchaudio.functional.resample(wav, fs, config.sample_rate)
                
                # Extract features
                feat = feature_extractor(wav)
                feat = feat.unsqueeze(0).to(device)
                
                # Extract embedding
                emb = embedding_model(feat).detach().cpu().numpy()
                embeddings.append(emb)
                file_info.append(file_info_item)
                
                # Save individual utterance embedding if requested
                if save_individual:
                    save_individual_embedding(emb.flatten(), file_info_item, output_dir, 'utterance')
                
            except Exception as e:
                logger.error(f"Rank {rank}: Error processing {file_info_item['path']}: {e}")
                continue
    
    return embeddings, file_info

def compute_speaker_similarity(speaker_embeddings, output_dir, threshold=0.8, logger=None):
    """Compute pairwise similarity between speaker embeddings."""
    if logger is None:
        logger = get_logger()
    
    if len(speaker_embeddings) < 2:
        logger.warning("Need at least 2 speakers to compute similarity.")
        return
        
    logger.info(f"Computing similarity matrix for {len(speaker_embeddings)} speakers...")
    
    # Prepare data
    speaker_keys = list(speaker_embeddings.keys())
    embeddings_matrix = np.array([speaker_embeddings[key]['embedding'] for key in speaker_keys])
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    # Create detailed similarity results
    similarity_results = {
        'similarity_matrix': similarity_matrix.tolist(),
        'speaker_keys': speaker_keys,
        'high_similarity_pairs': [],
        'statistics': {}
    }
    
    # Find high similarity pairs (excluding self-similarity)
    high_sim_pairs = []
    for i in range(len(speaker_keys)):
        for j in range(i+1, len(speaker_keys)):
            sim_score = similarity_matrix[i, j]
            if sim_score >= threshold:
                pair_info = {
                    'speaker1': speaker_keys[i],
                    'speaker2': speaker_keys[j],
                    'similarity': float(sim_score),
                    'speaker1_info': {
                        'dataset': speaker_embeddings[speaker_keys[i]]['dataset'],
                        'speaker_id': speaker_embeddings[speaker_keys[i]]['speaker_id'],
                        'num_utterances': speaker_embeddings[speaker_keys[i]]['num_utterances']
                    },
                    'speaker2_info': {
                        'dataset': speaker_embeddings[speaker_keys[j]]['dataset'],
                        'speaker_id': speaker_embeddings[speaker_keys[j]]['speaker_id'],
                        'num_utterances': speaker_embeddings[speaker_keys[j]]['num_utterances']
                    }
                }
                high_sim_pairs.append(pair_info)
    
    similarity_results['high_similarity_pairs'] = high_sim_pairs
    
    # Compute statistics
    # Get upper triangle of similarity matrix (excluding diagonal)
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    similarity_results['statistics'] = {
        'mean_similarity': float(np.mean(upper_triangle)),
        'std_similarity': float(np.std(upper_triangle)),
        'min_similarity': float(np.min(upper_triangle)),
        'max_similarity': float(np.max(upper_triangle)),
        'median_similarity': float(np.median(upper_triangle)),
        'num_high_similarity_pairs': len(high_sim_pairs),
        'total_pairs': len(upper_triangle),
        'threshold_used': threshold
    }
    
    # Save similarity results
    similarity_file = os.path.join(output_dir, 'speaker_similarity.json')
    with open(similarity_file, 'w') as f:
        json.dump(similarity_results, f, indent=2)
    
    # Save similarity matrix as numpy array for easy loading
    similarity_matrix_file = os.path.join(output_dir, 'similarity_matrix.npy')
    np.save(similarity_matrix_file, similarity_matrix)
    
    # Save speaker keys mapping
    speaker_mapping_file = os.path.join(output_dir, 'speaker_keys_mapping.json')
    speaker_mapping = {i: key for i, key in enumerate(speaker_keys)}
    with open(speaker_mapping_file, 'w') as f:
        json.dump(speaker_mapping, f, indent=2)
    
    # Create visualization if possible
    try:
        create_similarity_heatmap(similarity_matrix, speaker_keys, output_dir, logger)
    except Exception as e:
        logger.warning(f"Could not create similarity heatmap: {e}")
    
    # Print summary
    logger.info(f"Similarity computation completed:")
    logger.info(f"  - Mean similarity: {similarity_results['statistics']['mean_similarity']:.4f}")
    logger.info(f"  - Std similarity: {similarity_results['statistics']['std_similarity']:.4f}")
    logger.info(f"  - High similarity pairs (>={threshold}): {len(high_sim_pairs)}")
    logger.info(f"  - Results saved to: {similarity_file}")
    
    return similarity_results

def create_similarity_heatmap(similarity_matrix, speaker_keys, output_dir, logger):
    """Create and save similarity heatmap visualization."""
    try:
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(similarity_matrix, 
                   annot=False,  # Don't annotate all cells as it would be too crowded
                   cmap='viridis',
                   vmin=0, vmax=1,
                   square=True,
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title(f'Speaker Embedding Similarity Matrix ({len(speaker_keys)} speakers)')
        plt.xlabel('Speaker Index')
        plt.ylabel('Speaker Index')
        
        # Save the plot
        heatmap_file = os.path.join(output_dir, 'similarity_heatmap.png')
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity heatmap saved to: {heatmap_file}")
        
        # Create a smaller heatmap for the first 50 speakers if there are many speakers
        if len(speaker_keys) > 50:
            plt.figure(figsize=(12, 10))
            subset_matrix = similarity_matrix[:50, :50]
            subset_keys = speaker_keys[:50]
            
            sns.heatmap(subset_matrix,
                       annot=False,
                       cmap='viridis',
                       vmin=0, vmax=1,
                       square=True,
                       cbar_kws={'label': 'Cosine Similarity'})
            
            plt.title(f'Speaker Embedding Similarity Matrix (First 50 speakers)')
            plt.xlabel('Speaker Index')
            plt.ylabel('Speaker Index')
            
            subset_heatmap_file = os.path.join(output_dir, 'similarity_heatmap_subset.png')
            plt.savefig(subset_heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Subset similarity heatmap saved to: {subset_heatmap_file}")
            
    except ImportError:
        logger.warning("matplotlib or seaborn not available, skipping heatmap generation")
    except Exception as e:
        logger.warning(f"Error creating heatmap: {e}")

def main():
    args = parse_args()
    
    # Setup device and distributed training
    if args.use_gpu and torch.cuda.is_available():
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        gpu_id = int(args.gpu[rank % len(args.gpu)])
        device = torch.device(f'cuda:{gpu_id}')
    else:
        rank = 0
        world_size = 1
        device = torch.device('cpu')
    
    logger = get_logger()
    
    # All ranks should show their status
    logger.info(f"Rank {rank}/{world_size}: Starting on device {device}")
    
    # Load model and config
    logger.info(f"Rank {rank}: Loading model and configuration...")
    
    embedding_model, feature_extractor, config = load_model_and_config(
        args.config_file, args.model_path, device
    )
    
    logger.info(f"Rank {rank}: Model loaded successfully")
    
    # Scan audio files (all ranks need to do this for proper distribution)
    logger.info(f"Rank {rank}: Scanning audio files...")
    audio_files, dataset_names = scan_audio_files(args.data_root)
    logger.info(f"Rank {rank}: Found {len(audio_files)} audio files across {len(dataset_names)} datasets")
    
    # For multi-GPU, distribute the work
    if world_size > 1:
        # Initialize distributed training
        if torch.cuda.is_available():
            torch.distributed.init_process_group(backend='nccl')
        else:
            torch.distributed.init_process_group(backend='gloo')
        
        # Split files across processes
        files_per_process = len(audio_files) // world_size
        start_idx = rank * files_per_process
        if rank == world_size - 1:
            end_idx = len(audio_files)
        else:
            end_idx = (rank + 1) * files_per_process
        
        local_audio_files = audio_files[start_idx:end_idx]
        logger.info(f"Rank {rank}: Processing {len(local_audio_files)} files (indices {start_idx}-{end_idx-1})")
    else:
        local_audio_files = audio_files
        logger.info(f"Rank {rank}: Processing {len(local_audio_files)} files")
    
    # Create output directory for this rank
    rank_output_dir = os.path.join(args.output_dir, f'rank_{rank:02d}')
    os.makedirs(rank_output_dir, exist_ok=True)
    
    # Save metadata first (only rank 0)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump({
                'total_files': len(audio_files),
                'datasets': dataset_names,
                'num_datasets': len(dataset_names)
            }, f, indent=2)
    
    # Extract embeddings for local files
    utterance_embeddings = {}
    speaker_embeddings_dict = defaultdict(list)
    
    # Process files in batches
    batch_size = args.batch_size
    num_batches = (len(local_audio_files) + batch_size - 1) // batch_size
    
    # All ranks show progress bars
    logger.info(f"Rank {rank}: Starting embedding extraction with {num_batches} batches")
    progress_bar = tqdm(range(num_batches), 
                       desc=f"Rank {rank}: Extracting embeddings",
                       position=rank,
                       leave=True,
                       ncols=100)
    
    for batch_idx in progress_bar:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(local_audio_files))
        batch_files = local_audio_files[start_idx:end_idx]
        
        embeddings, file_info = extract_embeddings_batch(
            batch_files, embedding_model, feature_extractor, config, device, 
            args.output_dir, args.save_individual, rank, logger
        )
        
        for emb, info in zip(embeddings, file_info):
            # Store utterance embedding
            utt_key = f"{info['dataset']}_{info['speaker_id']}_{info['utterance_id']}"
            utterance_embeddings[utt_key] = {
                'embedding': emb.flatten(),
                'dataset': info['dataset'],
                'speaker_id': info['speaker_id'],
                'utterance_id': info['utterance_id'],
                'path': info['path']
            }
            
            # Collect for speaker-level averaging
            speaker_key = f"{info['dataset']}_{info['speaker_id']}"
            speaker_embeddings_dict[speaker_key].append(emb.flatten())
        
        # Update progress bar with current status
        progress_bar.set_postfix({
            'utterances': len(utterance_embeddings),
            'speakers': len(speaker_embeddings_dict)
        })
    
    logger.info(f"Rank {rank}: Finished extraction. Computing speaker averages...")
    
    # Compute speaker-level averages
    speaker_embeddings = {}
    for speaker_key, emb_list in speaker_embeddings_dict.items():
        if emb_list:
            avg_embedding = np.mean(emb_list, axis=0)
            dataset, speaker_id = speaker_key.split('_', 1)
            speaker_info = {
                'embedding': avg_embedding,
                'dataset': dataset,
                'speaker_id': speaker_id,
                'num_utterances': len(emb_list)
            }
            speaker_embeddings[speaker_key] = speaker_info
            
            # Save individual speaker embedding if requested
            if args.save_individual:
                save_individual_embedding(avg_embedding, {
                    'dataset': dataset,
                    'speaker_id': speaker_id,
                    'num_utterances': len(emb_list)
                }, args.output_dir, 'speaker')
    
    logger.info(f"Rank {rank}: Computed {len(speaker_embeddings)} speaker embeddings")
    
    # Save results for this rank
    logger.info(f"Rank {rank}: Saving results to {rank_output_dir}")
    
    # Save utterance embeddings
    with open(os.path.join(rank_output_dir, 'utterance_embeddings.pkl'), 'wb') as f:
        pickle.dump(utterance_embeddings, f)
    
    # Save speaker embeddings
    with open(os.path.join(rank_output_dir, 'speaker_embeddings.pkl'), 'wb') as f:
        pickle.dump(speaker_embeddings, f)
    
    # Compute similarity if requested (only for rank 0 to avoid duplication)
    if args.compute_similarity and rank == 0:
        logger.info(f"Rank {rank}: Computing speaker embedding similarity...")
        compute_speaker_similarity(
            speaker_embeddings, 
            rank_output_dir, 
            threshold=args.similarity_threshold,
            logger=logger
        )
    
    # Save metadata
    metadata = {
        'rank': rank,
        'world_size': world_size,
        'device': str(device),
        'num_utterances': len(utterance_embeddings),
        'num_speakers': len(speaker_embeddings),
        'utterance_keys': list(utterance_embeddings.keys()),
        'speaker_keys': list(speaker_embeddings.keys()),
        'similarity_computed': args.compute_similarity and rank == 0,
        'individual_files_saved': args.save_individual
    }
    
    with open(os.path.join(rank_output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Rank {rank}: Extracted {len(utterance_embeddings)} utterance embeddings")
    logger.info(f"Rank {rank}: Computed {len(speaker_embeddings)} speaker embeddings")
    logger.info(f"Rank {rank}: Results saved to {rank_output_dir}")
    
    if args.save_individual:
        logger.info(f"Rank {rank}: Individual embedding files saved in directory structure")
    
    # Synchronize all processes before finishing
    if world_size > 1:
        torch.distributed.barrier()
        logger.info(f"Rank {rank}: Finished and synchronized with other ranks")

if __name__ == "__main__":
    main() 