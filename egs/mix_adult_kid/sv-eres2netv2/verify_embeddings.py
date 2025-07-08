#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import json
import pickle
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Verify embedding quality by computing intra/inter-speaker similarities.')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing individual embedding files')
    parser.add_argument('--num_speakers', type=int, default=50,
                        help='Number of speakers to sample for analysis')
    parser.add_argument('--num_utterances_per_speaker', type=int, default=10,
                        help='Maximum number of utterances per speaker to use')
    parser.add_argument('--output_dir', type=str, default='verification_results',
                        help='Output directory for verification results')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--create_plots', action='store_true', default=True,
                        help='Create visualization plots')
    
    return parser.parse_args()

def load_file_index(embeddings_dir):
    """Load the file index."""
    index_file = os.path.join(embeddings_dir, 'file_index.json')
    if not os.path.exists(index_file):
        print(f"File index not found: {index_file}")
        return None
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    return index

def load_individual_embedding(file_path):
    """Load a single individual embedding file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data['embedding'], data['info']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def group_utterances_by_speaker(index):
    """Group utterances by speaker."""
    speaker_utterances = defaultdict(list)
    
    for utterance_key, file_path in index['utterances'].items():
        # Parse key: dataset_speaker_utterance
        parts = utterance_key.split('_')
        if len(parts) >= 3:
            dataset = parts[0]
            speaker_id = parts[1]
            utterance_id = '_'.join(parts[2:])
            
            speaker_key = f"{dataset}_{speaker_id}"
            speaker_utterances[speaker_key].append({
                'key': utterance_key,
                'file_path': file_path,
                'utterance_id': utterance_id
            })
    
    return speaker_utterances

def sample_speakers_and_utterances(speaker_utterances, num_speakers, num_utterances_per_speaker, random_seed):
    """Sample speakers and their utterances for analysis."""
    random.seed(random_seed)
    
    # Filter speakers with enough utterances
    valid_speakers = {
        speaker: utterances for speaker, utterances in speaker_utterances.items()
        if len(utterances) >= 2  # Need at least 2 utterances for intra-speaker comparison
    }
    
    if len(valid_speakers) < num_speakers:
        print(f"Warning: Only {len(valid_speakers)} speakers have enough utterances")
        num_speakers = len(valid_speakers)
    
    # Sample speakers
    selected_speakers = random.sample(list(valid_speakers.keys()), num_speakers)
    
    # Sample utterances for each speaker
    sampled_data = {}
    for speaker in selected_speakers:
        utterances = valid_speakers[speaker]
        num_to_sample = min(num_utterances_per_speaker, len(utterances))
        sampled_utterances = random.sample(utterances, num_to_sample)
        sampled_data[speaker] = sampled_utterances
    
    return sampled_data

def load_embeddings_for_analysis(sampled_data):
    """Load embeddings for the sampled data."""
    speaker_embeddings = {}
    
    for speaker, utterances in sampled_data.items():
        embeddings = []
        valid_utterances = []
        
        for utt_info in utterances:
            embedding, info = load_individual_embedding(utt_info['file_path'])
            if embedding is not None:
                embeddings.append(embedding)
                valid_utterances.append(utt_info)
        
        if embeddings:
            speaker_embeddings[speaker] = {
                'embeddings': np.array(embeddings),
                'utterances': valid_utterances
            }
    
    return speaker_embeddings

def compute_intra_speaker_similarities(speaker_embeddings):
    """Compute intra-speaker similarities."""
    intra_similarities = []
    speaker_stats = {}
    
    for speaker, data in speaker_embeddings.items():
        embeddings = data['embeddings']
        if len(embeddings) < 2:
            continue
        
        # Compute pairwise similarities within speaker
        sim_matrix = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        
        intra_similarities.extend(upper_triangle)
        
        speaker_stats[speaker] = {
            'num_utterances': len(embeddings),
            'mean_similarity': float(np.mean(upper_triangle)),
            'std_similarity': float(np.std(upper_triangle)),
            'min_similarity': float(np.min(upper_triangle)),
            'max_similarity': float(np.max(upper_triangle))
        }
    
    return np.array(intra_similarities), speaker_stats

def compute_inter_speaker_similarities(speaker_embeddings, max_pairs=1000):
    """Compute inter-speaker similarities."""
    speakers = list(speaker_embeddings.keys())
    inter_similarities = []
    
    # Sample pairs to avoid computing too many similarities
    if len(speakers) > 20:  # If too many speakers, sample pairs
        num_pairs = min(max_pairs, len(speakers) * (len(speakers) - 1) // 2)
        speaker_pairs = []
        
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                speaker_pairs.append((speakers[i], speakers[j]))
        
        if len(speaker_pairs) > num_pairs:
            speaker_pairs = random.sample(speaker_pairs, num_pairs)
    else:
        speaker_pairs = [(speakers[i], speakers[j]) 
                        for i in range(len(speakers)) 
                        for j in range(i + 1, len(speakers))]
    
    for speaker1, speaker2 in speaker_pairs:
        embeddings1 = speaker_embeddings[speaker1]['embeddings']
        embeddings2 = speaker_embeddings[speaker2]['embeddings']
        
        # Compute cross similarities
        cross_similarities = cosine_similarity(embeddings1, embeddings2)
        inter_similarities.extend(cross_similarities.flatten())
    
    return np.array(inter_similarities)

def compute_verification_metrics(intra_similarities, inter_similarities, thresholds=None):
    """Compute verification metrics."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.01)
    
    metrics = {}
    
    for threshold in thresholds:
        # True positives: intra-speaker similarities >= threshold
        tp = np.sum(intra_similarities >= threshold)
        # False negatives: intra-speaker similarities < threshold
        fn = np.sum(intra_similarities < threshold)
        # False positives: inter-speaker similarities >= threshold
        fp = np.sum(inter_similarities >= threshold)
        # True negatives: inter-speaker similarities < threshold
        tn = np.sum(inter_similarities < threshold)
        
        # Compute metrics
        if tp + fn > 0:
            tar = tp / (tp + fn)  # True Acceptance Rate (Recall)
        else:
            tar = 0.0
        
        if fp + tn > 0:
            far = fp / (fp + tn)  # False Acceptance Rate
        else:
            far = 0.0
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        
        if tar + precision > 0:
            f1 = 2 * (precision * tar) / (precision + tar)
        else:
            f1 = 0.0
        
        metrics[threshold] = {
            'tar': tar,
            'far': far,
            'precision': precision,
            'f1': f1,
            'eer': abs(tar - (1 - far))  # Equal Error Rate approximation
        }
    
    return metrics

def find_optimal_threshold(metrics):
    """Find optimal threshold based on minimum EER."""
    min_eer = float('inf')
    optimal_threshold = 0.5
    
    for threshold, metric in metrics.items():
        if metric['eer'] < min_eer:
            min_eer = metric['eer']
            optimal_threshold = threshold
    
    return optimal_threshold, min_eer

def create_visualization_plots(intra_similarities, inter_similarities, metrics, output_dir):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Similarity distributions
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(intra_similarities, bins=50, alpha=0.7, label='Intra-speaker', color='blue', density=True)
    plt.hist(inter_similarities, bins=50, alpha=0.7, label='Inter-speaker', color='red', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plots
    plt.subplot(2, 2, 2)
    data_to_plot = [intra_similarities, inter_similarities]
    labels = ['Intra-speaker', 'Inter-speaker']
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Box Plots')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: ROC-like curve
    plt.subplot(2, 2, 3)
    thresholds = sorted(metrics.keys())
    tars = [metrics[t]['tar'] for t in thresholds]
    fars = [metrics[t]['far'] for t in thresholds]
    
    plt.plot(fars, tars, 'b-', linewidth=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('True Acceptance Rate')
    plt.title('ROC-like Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: EER vs Threshold
    plt.subplot(2, 2, 4)
    eers = [metrics[t]['eer'] for t in thresholds]
    plt.plot(thresholds, eers, 'g-', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Equal Error Rate')
    plt.title('EER vs Threshold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Detailed distribution comparison
    plt.figure(figsize=(10, 6))
    
    # Create overlapping histograms with transparency
    bins = np.linspace(min(np.min(intra_similarities), np.min(inter_similarities)),
                      max(np.max(intra_similarities), np.max(inter_similarities)), 50)
    
    plt.hist(intra_similarities, bins=bins, alpha=0.6, label=f'Intra-speaker (n={len(intra_similarities)})', 
             color='blue', density=True)
    plt.hist(inter_similarities, bins=bins, alpha=0.6, label=f'Inter-speaker (n={len(inter_similarities)})', 
             color='red', density=True)
    
    # Add vertical lines for means
    plt.axvline(np.mean(intra_similarities), color='blue', linestyle='--', alpha=0.8, 
                label=f'Intra mean: {np.mean(intra_similarities):.3f}')
    plt.axvline(np.mean(inter_similarities), color='red', linestyle='--', alpha=0.8, 
                label=f'Inter mean: {np.mean(inter_similarities):.3f}')
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Intra-speaker vs Inter-speaker Similarity Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'detailed_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    print("Starting embedding verification...")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load file index
    print("Loading file index...")
    index = load_file_index(args.embeddings_dir)
    if index is None:
        print("Could not load file index. Exiting.")
        return
    
    print(f"Found {len(index['utterances'])} utterance files")
    
    # Group utterances by speaker
    print("Grouping utterances by speaker...")
    speaker_utterances = group_utterances_by_speaker(index)
    print(f"Found {len(speaker_utterances)} speakers")
    
    # Sample speakers and utterances
    print(f"Sampling {args.num_speakers} speakers with up to {args.num_utterances_per_speaker} utterances each...")
    sampled_data = sample_speakers_and_utterances(
        speaker_utterances, args.num_speakers, args.num_utterances_per_speaker, args.random_seed
    )
    
    print(f"Selected {len(sampled_data)} speakers for analysis")
    
    # Load embeddings
    print("Loading embeddings...")
    speaker_embeddings = load_embeddings_for_analysis(sampled_data)
    
    total_utterances = sum(len(data['embeddings']) for data in speaker_embeddings.values())
    print(f"Loaded embeddings for {len(speaker_embeddings)} speakers, {total_utterances} utterances")
    
    # Compute intra-speaker similarities
    print("Computing intra-speaker similarities...")
    intra_similarities, speaker_stats = compute_intra_speaker_similarities(speaker_embeddings)
    print(f"Computed {len(intra_similarities)} intra-speaker similarity pairs")
    
    # Compute inter-speaker similarities
    print("Computing inter-speaker similarities...")
    inter_similarities = compute_inter_speaker_similarities(speaker_embeddings)
    print(f"Computed {len(inter_similarities)} inter-speaker similarity pairs")
    
    # Compute verification metrics
    print("Computing verification metrics...")
    metrics = compute_verification_metrics(intra_similarities, inter_similarities)
    
    # Find optimal threshold
    optimal_threshold, min_eer = find_optimal_threshold(metrics)
    
    # Print results
    print(f"\n=== EMBEDDING VERIFICATION RESULTS ===")
    print(f"Intra-speaker similarities:")
    print(f"  Count: {len(intra_similarities)}")
    print(f"  Mean: {np.mean(intra_similarities):.4f}")
    print(f"  Std: {np.std(intra_similarities):.4f}")
    print(f"  Min: {np.min(intra_similarities):.4f}")
    print(f"  Max: {np.max(intra_similarities):.4f}")
    
    print(f"\nInter-speaker similarities:")
    print(f"  Count: {len(inter_similarities)}")
    print(f"  Mean: {np.mean(inter_similarities):.4f}")
    print(f"  Std: {np.std(inter_similarities):.4f}")
    print(f"  Min: {np.min(inter_similarities):.4f}")
    print(f"  Max: {np.max(inter_similarities):.4f}")
    
    print(f"\nSeparation quality:")
    mean_diff = np.mean(intra_similarities) - np.mean(inter_similarities)
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Minimum EER: {min_eer:.4f}")
    
    optimal_metrics = metrics[optimal_threshold]
    print(f"\nPerformance at optimal threshold ({optimal_threshold:.3f}):")
    print(f"  True Acceptance Rate: {optimal_metrics['tar']:.4f}")
    print(f"  False Acceptance Rate: {optimal_metrics['far']:.4f}")
    print(f"  Precision: {optimal_metrics['precision']:.4f}")
    print(f"  F1 Score: {optimal_metrics['f1']:.4f}")
    
    # Quality assessment
    if mean_diff > 0.2:
        quality = "Excellent"
    elif mean_diff > 0.1:
        quality = "Good"
    elif mean_diff > 0.05:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"\nEmbedding Quality Assessment: {quality}")
    
    # Show some examples
    print(f"\nSample high intra-speaker similarities:")
    high_intra_indices = np.argsort(intra_similarities)[-5:]
    for i, idx in enumerate(high_intra_indices):
        print(f"  {i+1}. {intra_similarities[idx]:.4f}")
    
    print(f"\nSample low inter-speaker similarities:")
    low_inter_indices = np.argsort(inter_similarities)[:5]
    for i, idx in enumerate(low_inter_indices):
        print(f"  {i+1}. {inter_similarities[idx]:.4f}")
    
    # Save detailed results
    results = {
        'summary': {
            'num_speakers': len(speaker_embeddings),
            'total_utterances': total_utterances,
            'intra_similarities_count': len(intra_similarities),
            'inter_similarities_count': len(inter_similarities),
            'mean_intra_similarity': float(np.mean(intra_similarities)),
            'mean_inter_similarity': float(np.mean(inter_similarities)),
            'mean_difference': float(mean_diff),
            'optimal_threshold': float(optimal_threshold),
            'min_eer': float(min_eer),
            'quality_assessment': quality
        },
        'intra_similarity_stats': {
            'mean': float(np.mean(intra_similarities)),
            'std': float(np.std(intra_similarities)),
            'min': float(np.min(intra_similarities)),
            'max': float(np.max(intra_similarities)),
            'median': float(np.median(intra_similarities))
        },
        'inter_similarity_stats': {
            'mean': float(np.mean(inter_similarities)),
            'std': float(np.std(inter_similarities)),
            'min': float(np.min(inter_similarities)),
            'max': float(np.max(inter_similarities)),
            'median': float(np.median(inter_similarities))
        },
        'speaker_individual_stats': speaker_stats,
        'optimal_metrics': {
            'threshold': float(optimal_threshold),
            'tar': float(optimal_metrics['tar']),
            'far': float(optimal_metrics['far']),
            'precision': float(optimal_metrics['precision']),
            'f1': float(optimal_metrics['f1']),
            'eer': float(optimal_metrics['eer'])
        }
    }
    
    results_file = os.path.join(args.output_dir, 'verification_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create plots
    if args.create_plots:
        print("Creating visualization plots...")
        try:
            create_visualization_plots(intra_similarities, inter_similarities, metrics, args.output_dir)
            print(f"Plots saved to: {args.output_dir}")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    print("Verification completed!")

if __name__ == "__main__":
    main() 