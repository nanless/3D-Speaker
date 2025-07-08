#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from speakerlab.utils.utils import get_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Merge embeddings from multiple GPU ranks and compute similarity.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing rank_XX subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for merged results')
    parser.add_argument('--compute_similarity', action='store_true', 
                        help='Compute speaker embedding similarity matrix')
    parser.add_argument('--similarity_threshold', type=float, default=0.8,
                        help='Threshold for identifying similar speakers')
    parser.add_argument('--merge_individual', action='store_true', default=True,
                        help='Merge individual embedding files into organized structure')
    
    return parser.parse_args()

def merge_individual_embeddings(input_dir, output_dir, logger):
    """Merge individual embedding files from all ranks."""
    logger.info("Merging individual embedding files...")
    
    # Create output directories
    individual_output_dir = os.path.join(output_dir, 'embeddings_individual')
    os.makedirs(individual_output_dir, exist_ok=True)
    
    # Find all rank directories
    rank_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir() and d.name.startswith('rank_')]
    
    merged_count = {'utterances': 0, 'speakers': 0}
    
    for rank_dir in sorted(rank_dirs):
        logger.info(f"Processing individual files from {rank_dir.name}...")
        
        # Look for individual embeddings in the main input directory structure
        individual_source = os.path.join(input_dir, 'embeddings_individual')
        
        if os.path.exists(individual_source):
            # Copy utterance embeddings
            utt_source = os.path.join(individual_source, 'utterances')
            if os.path.exists(utt_source):
                utt_target = os.path.join(individual_output_dir, 'utterances')
                count = copy_individual_files(utt_source, utt_target, logger)
                merged_count['utterances'] += count
            
            # Copy speaker embeddings
            spk_source = os.path.join(individual_source, 'speakers')
            if os.path.exists(spk_source):
                spk_target = os.path.join(individual_output_dir, 'speakers')
                count = copy_individual_files(spk_source, spk_target, logger)
                merged_count['speakers'] += count
    
    logger.info(f"Merged individual files: {merged_count['utterances']} utterances, {merged_count['speakers']} speakers")
    
    return merged_count

def copy_individual_files(source_dir, target_dir, logger):
    """Copy individual embedding files maintaining directory structure."""
    if not os.path.exists(source_dir):
        return 0
    
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    
    # Recursively copy all .pkl files
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.pkl'):
                source_file = os.path.join(root, file)
                # Maintain relative directory structure
                rel_path = os.path.relpath(root, source_dir)
                target_subdir = os.path.join(target_dir, rel_path)
                os.makedirs(target_subdir, exist_ok=True)
                
                target_file = os.path.join(target_subdir, file)
                
                # Copy if not already exists or if source is newer
                if not os.path.exists(target_file) or os.path.getmtime(source_file) > os.path.getmtime(target_file):
                    import shutil
                    shutil.copy2(source_file, target_file)
                    count += 1
    
    return count

def load_individual_embedding(file_path):
    """Load a single individual embedding file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data['embedding'], data['info']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def create_individual_file_index(individual_dir, logger):
    """Create an index of all individual embedding files."""
    logger.info("Creating index of individual embedding files...")
    
    index = {
        'utterances': {},
        'speakers': {}
    }
    
    # Index utterance files
    utt_dir = os.path.join(individual_dir, 'utterances')
    if os.path.exists(utt_dir):
        for dataset_dir in Path(utt_dir).iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                for speaker_dir in dataset_dir.iterdir():
                    if speaker_dir.is_dir():
                        speaker_id = speaker_dir.name
                        for utt_file in speaker_dir.iterdir():
                            if utt_file.suffix == '.pkl':
                                utt_id = utt_file.stem
                                key = f"{dataset_name}_{speaker_id}_{utt_id}"
                                index['utterances'][key] = str(utt_file)
    
    # Index speaker files
    spk_dir = os.path.join(individual_dir, 'speakers')
    if os.path.exists(spk_dir):
        for dataset_dir in Path(spk_dir).iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                for spk_file in dataset_dir.iterdir():
                    if spk_file.suffix == '.pkl':
                        speaker_id = spk_file.stem
                        key = f"{dataset_name}_{speaker_id}"
                        index['speakers'][key] = str(spk_file)
    
    logger.info(f"Indexed {len(index['utterances'])} utterance files and {len(index['speakers'])} speaker files")
    
    # Save index
    index_file = os.path.join(individual_dir, 'file_index.json')
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    return index

def merge_embeddings(input_dir, logger):
    """Merge embeddings from all rank directories."""
    merged_utterance_embeddings = {}
    merged_speaker_embeddings = {}
    
    rank_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir() and d.name.startswith('rank_')]
    
    if not rank_dirs:
        logger.error(f"No rank directories found in {input_dir}")
        return None, None
    
    logger.info(f"Found {len(rank_dirs)} rank directories to merge")
    
    for rank_dir in sorted(rank_dirs):
        rank_name = rank_dir.name
        logger.info(f"Processing {rank_name}...")
        
        # Load utterance embeddings
        utt_emb_file = rank_dir / 'utterance_embeddings.pkl'
        if utt_emb_file.exists():
            with open(utt_emb_file, 'rb') as f:
                rank_utt_embeddings = pickle.load(f)
            merged_utterance_embeddings.update(rank_utt_embeddings)
            logger.info(f"  Loaded {len(rank_utt_embeddings)} utterance embeddings from {rank_name}")
        
        # Load speaker embeddings
        spk_emb_file = rank_dir / 'speaker_embeddings.pkl'
        if spk_emb_file.exists():
            with open(spk_emb_file, 'rb') as f:
                rank_spk_embeddings = pickle.load(f)
            merged_speaker_embeddings.update(rank_spk_embeddings)
            logger.info(f"  Loaded {len(rank_spk_embeddings)} speaker embeddings from {rank_name}")
    
    logger.info(f"Merged results:")
    logger.info(f"  Total utterance embeddings: {len(merged_utterance_embeddings)}")
    logger.info(f"  Total speaker embeddings: {len(merged_speaker_embeddings)}")
    
    return merged_utterance_embeddings, merged_speaker_embeddings

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

def analyze_similarity_distribution(similarity_results, output_dir, logger):
    """Analyze and visualize similarity distribution."""
    try:
        statistics = similarity_results['statistics']
        
        # Create similarity distribution plot
        similarity_matrix = np.array(similarity_results['similarity_matrix'])
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        plt.figure(figsize=(12, 8))
        
        # Histogram of similarities
        plt.subplot(2, 2, 1)
        plt.hist(upper_triangle, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(statistics['mean_similarity'], color='red', linestyle='--', 
                   label=f'Mean: {statistics["mean_similarity"]:.3f}')
        plt.axvline(statistics['median_similarity'], color='green', linestyle='--', 
                   label=f'Median: {statistics["median_similarity"]:.3f}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Speaker Similarity Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(upper_triangle, vert=True)
        plt.ylabel('Cosine Similarity')
        plt.title('Similarity Score Box Plot')
        plt.grid(True, alpha=0.3)
        
        # CDF plot
        plt.subplot(2, 2, 3)
        sorted_similarities = np.sort(upper_triangle)
        y = np.arange(1, len(sorted_similarities) + 1) / len(sorted_similarities)
        plt.plot(sorted_similarities, y, linewidth=2)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        plt.grid(True, alpha=0.3)
        
        # High similarity pairs analysis
        plt.subplot(2, 2, 4)
        high_sim_pairs = similarity_results['high_similarity_pairs']
        if high_sim_pairs:
            similarities = [pair['similarity'] for pair in high_sim_pairs]
            plt.hist(similarities, bins=20, alpha=0.7, color='orange', edgecolor='black')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Number of Pairs')
            plt.title(f'High Similarity Pairs (≥{statistics["threshold_used"]})')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'No pairs above\nthreshold {statistics["threshold_used"]}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('High Similarity Pairs')
        
        plt.tight_layout()
        
        # Save the plot
        analysis_file = os.path.join(output_dir, 'similarity_analysis.png')
        plt.savefig(analysis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity analysis plot saved to: {analysis_file}")
        
    except Exception as e:
        logger.warning(f"Error creating similarity analysis: {e}")

def main():
    args = parse_args()
    logger = get_logger()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Merge embeddings from all ranks
    logger.info("Merging embeddings from all ranks...")
    utterance_embeddings, speaker_embeddings = merge_embeddings(args.input_dir, logger)
    
    if utterance_embeddings is None or speaker_embeddings is None:
        logger.error("Failed to merge embeddings")
        return
    
    # Merge individual embedding files if requested
    if args.merge_individual:
        individual_counts = merge_individual_embeddings(args.input_dir, args.output_dir, logger)
        
        # Create index for individual files
        individual_dir = os.path.join(args.output_dir, 'embeddings_individual')
        if os.path.exists(individual_dir):
            file_index = create_individual_file_index(individual_dir, logger)
    
    # Save merged results
    logger.info("Saving merged embeddings...")
    
    with open(os.path.join(args.output_dir, 'utterance_embeddings.pkl'), 'wb') as f:
        pickle.dump(utterance_embeddings, f)
    
    with open(os.path.join(args.output_dir, 'speaker_embeddings.pkl'), 'wb') as f:
        pickle.dump(speaker_embeddings, f)
    
    # Create summary information
    utterance_list = []
    speaker_list = []
    dataset_stats = defaultdict(lambda: {'speakers': 0, 'utterances': 0})
    
    for utt_key, utt_info in utterance_embeddings.items():
        utterance_list.append({
            'key': utt_key,
            'dataset': utt_info['dataset'],
            'speaker_id': utt_info['speaker_id'],
            'utterance_id': utt_info['utterance_id'],
            'path': utt_info['path']
        })
        dataset_stats[utt_info['dataset']]['utterances'] += 1
    
    for spk_key, spk_info in speaker_embeddings.items():
        speaker_list.append({
            'key': spk_key,
            'dataset': spk_info['dataset'],
            'speaker_id': spk_info['speaker_id'],
            'num_utterances': spk_info['num_utterances']
        })
        dataset_stats[spk_info['dataset']]['speakers'] += 1
    
    # Save lists
    with open(os.path.join(args.output_dir, 'utterance_list.json'), 'w') as f:
        json.dump(utterance_list, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'speaker_list.json'), 'w') as f:
        json.dump(speaker_list, f, indent=2)
    
    # Save summary
    summary = {
        'total_utterances': len(utterance_embeddings),
        'total_speakers': len(speaker_embeddings),
        'datasets': dict(dataset_stats),
        'num_datasets': len(dataset_stats),
        'individual_files_available': args.merge_individual
    }
    
    if args.merge_individual and 'individual_counts' in locals():
        summary['individual_file_counts'] = individual_counts
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Merged embeddings saved to {args.output_dir}")
    logger.info(f"Summary: {summary['total_utterances']} utterances, {summary['total_speakers']} speakers, {summary['num_datasets']} datasets")
    
    if args.merge_individual:
        logger.info("Individual embedding files are available in 'embeddings_individual' subdirectory")
    
    # Compute similarity if requested
    if args.compute_similarity:
        logger.info("Computing speaker embedding similarity...")
        similarity_results = compute_speaker_similarity(
            speaker_embeddings, 
            args.output_dir, 
            threshold=args.similarity_threshold,
            logger=logger
        )
        
        if similarity_results:
            # Create detailed analysis
            analyze_similarity_distribution(similarity_results, args.output_dir, logger)
            
            # Save high similarity pairs to separate file for easy access
            high_sim_pairs = similarity_results['high_similarity_pairs']
            if high_sim_pairs:
                with open(os.path.join(args.output_dir, 'high_similarity_pairs.json'), 'w') as f:
                    json.dump(high_sim_pairs, f, indent=2)
                logger.info(f"High similarity pairs saved to high_similarity_pairs.json")

if __name__ == "__main__":
    main() 