#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import json
import numpy as np
import pickle
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze speaker embedding similarity results.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing similarity analysis results')
    parser.add_argument('--output_csv', type=str, default='similarity_analysis.csv',
                        help='Output CSV file for detailed analysis')
    parser.add_argument('--min_similarity', type=float, default=0.5,
                        help='Minimum similarity threshold for reporting')
    parser.add_argument('--max_results', type=int, default=100,
                        help='Maximum number of similar pairs to show')
    
    return parser.parse_args()

def load_similarity_results(results_dir):
    """Load similarity analysis results."""
    similarity_file = os.path.join(results_dir, 'speaker_similarity.json')
    matrix_file = os.path.join(results_dir, 'similarity_matrix.npy')
    mapping_file = os.path.join(results_dir, 'speaker_keys_mapping.json')
    
    if not os.path.exists(similarity_file):
        print(f"Error: Similarity results not found in {results_dir}")
        return None, None, None
    
    # Load JSON results
    with open(similarity_file, 'r') as f:
        similarity_results = json.load(f)
    
    # Load similarity matrix
    similarity_matrix = None
    if os.path.exists(matrix_file):
        similarity_matrix = np.load(matrix_file)
    
    # Load speaker mapping
    speaker_mapping = None
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            speaker_mapping = json.load(f)
    
    return similarity_results, similarity_matrix, speaker_mapping

def analyze_similarity_patterns(similarity_results, min_similarity=0.5):
    """Analyze patterns in speaker similarity."""
    high_sim_pairs = similarity_results['high_similarity_pairs']
    
    # Dataset-level analysis
    dataset_pairs = {}
    cross_dataset_pairs = []
    intra_dataset_pairs = []
    
    for pair in high_sim_pairs:
        if pair['similarity'] >= min_similarity:
            dataset1 = pair['speaker1_info']['dataset']
            dataset2 = pair['speaker2_info']['dataset']
            
            if dataset1 == dataset2:
                intra_dataset_pairs.append(pair)
                if dataset1 not in dataset_pairs:
                    dataset_pairs[dataset1] = []
                dataset_pairs[dataset1].append(pair)
            else:
                cross_dataset_pairs.append(pair)
    
    return {
        'intra_dataset_pairs': intra_dataset_pairs,
        'cross_dataset_pairs': cross_dataset_pairs,
        'dataset_pairs': dataset_pairs
    }

def create_similarity_report(similarity_results, analysis_patterns, output_file):
    """Create detailed similarity report as CSV."""
    high_sim_pairs = similarity_results['high_similarity_pairs']
    
    # Prepare data for CSV
    csv_data = []
    
    for pair in high_sim_pairs:
        csv_data.append({
            'speaker1_key': pair['speaker1'],
            'speaker2_key': pair['speaker2'],
            'similarity': pair['similarity'],
            'speaker1_dataset': pair['speaker1_info']['dataset'],
            'speaker1_id': pair['speaker1_info']['speaker_id'],
            'speaker1_utterances': pair['speaker1_info']['num_utterances'],
            'speaker2_dataset': pair['speaker2_info']['dataset'],
            'speaker2_id': pair['speaker2_info']['speaker_id'],
            'speaker2_utterances': pair['speaker2_info']['num_utterances'],
            'same_dataset': pair['speaker1_info']['dataset'] == pair['speaker2_info']['dataset'],
            'pair_type': 'intra_dataset' if pair['speaker1_info']['dataset'] == pair['speaker2_info']['dataset'] else 'cross_dataset'
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    if not df.empty:
        df = df.sort_values('similarity', ascending=False)
        df.to_csv(output_file, index=False)
        print(f"Detailed similarity report saved to: {output_file}")
    else:
        print("No high similarity pairs found for CSV export.")
    
    return df

def print_summary_report(similarity_results, analysis_patterns, args):
    """Print summary report to console."""
    stats = similarity_results['statistics']
    
    print("\n" + "="*60)
    print("SPEAKER EMBEDDING SIMILARITY ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nBasic Statistics:")
    print(f"  Total speakers analyzed: {len(similarity_results['speaker_keys'])}")
    print(f"  Total speaker pairs: {stats['total_pairs']}")
    print(f"  Mean similarity: {stats['mean_similarity']:.4f}")
    print(f"  Median similarity: {stats['median_similarity']:.4f}")
    print(f"  Standard deviation: {stats['std_similarity']:.4f}")
    print(f"  Min similarity: {stats['min_similarity']:.4f}")
    print(f"  Max similarity: {stats['max_similarity']:.4f}")
    
    print(f"\nHigh Similarity Analysis (threshold >= {stats['threshold_used']}):")
    print(f"  Total high similarity pairs: {stats['num_high_similarity_pairs']}")
    
    if analysis_patterns['intra_dataset_pairs']:
        print(f"  Intra-dataset pairs: {len(analysis_patterns['intra_dataset_pairs'])}")
    
    if analysis_patterns['cross_dataset_pairs']:
        print(f"  Cross-dataset pairs: {len(analysis_patterns['cross_dataset_pairs'])}")
    
    # Dataset breakdown
    if analysis_patterns['dataset_pairs']:
        print(f"\nDataset-specific Analysis:")
        for dataset, pairs in analysis_patterns['dataset_pairs'].items():
            avg_sim = np.mean([p['similarity'] for p in pairs])
            print(f"  {dataset}: {len(pairs)} pairs, avg similarity: {avg_sim:.4f}")
    
    # Top similar pairs
    high_sim_pairs = similarity_results['high_similarity_pairs']
    if high_sim_pairs:
        print(f"\nTop {min(args.max_results, len(high_sim_pairs))} Most Similar Speaker Pairs:")
        sorted_pairs = sorted(high_sim_pairs, key=lambda x: x['similarity'], reverse=True)
        
        for i, pair in enumerate(sorted_pairs[:args.max_results], 1):
            if pair['similarity'] >= args.min_similarity:
                same_dataset = pair['speaker1_info']['dataset'] == pair['speaker2_info']['dataset']
                pair_type = "Same Dataset" if same_dataset else "Cross Dataset"
                
                print(f"\n  {i}. Similarity: {pair['similarity']:.4f} ({pair_type})")
                print(f"     Speaker 1: {pair['speaker1_info']['dataset']}/{pair['speaker1_info']['speaker_id']} ({pair['speaker1_info']['num_utterances']} utts)")
                print(f"     Speaker 2: {pair['speaker2_info']['dataset']}/{pair['speaker2_info']['speaker_id']} ({pair['speaker2_info']['num_utterances']} utts)")
    
    # Potential issues analysis
    print(f"\nPotential Data Quality Issues:")
    very_high_sim = [p for p in high_sim_pairs if p['similarity'] > 0.95]
    if very_high_sim:
        print(f"  Very high similarity pairs (>0.95): {len(very_high_sim)}")
        print("  These pairs might indicate:")
        print("    - Duplicate speakers across datasets")
        print("    - Same person recorded in different conditions")
        print("    - Data quality issues")
    else:
        print("  No extremely high similarity pairs detected.")
    
    cross_dataset_high = [p for p in analysis_patterns['cross_dataset_pairs'] if p['similarity'] > 0.9]
    if cross_dataset_high:
        print(f"  High cross-dataset similarity pairs (>0.9): {len(cross_dataset_high)}")
        print("  These might indicate speakers appearing in multiple datasets.")
    
    print("\n" + "="*60)

def main():
    args = parse_args()
    
    # Load results
    print(f"Loading similarity results from: {args.results_dir}")
    similarity_results, similarity_matrix, speaker_mapping = load_similarity_results(args.results_dir)
    
    if similarity_results is None:
        return
    
    print(f"Loaded results for {len(similarity_results['speaker_keys'])} speakers")
    
    # Analyze patterns
    analysis_patterns = analyze_similarity_patterns(similarity_results, args.min_similarity)
    
    # Create CSV report
    df = create_similarity_report(similarity_results, analysis_patterns, args.output_csv)
    
    # Print summary report
    print_summary_report(similarity_results, analysis_patterns, args)
    
    # Additional analysis suggestions
    print(f"\nFor further analysis, you can:")
    print(f"  - Load the similarity matrix: np.load('{os.path.join(args.results_dir, 'similarity_matrix.npy')}')")
    print(f"  - View visualization: {os.path.join(args.results_dir, 'similarity_heatmap.png')}")
    print(f"  - Check analysis plots: {os.path.join(args.results_dir, 'similarity_analysis.png')}")
    print(f"  - Review detailed CSV: {args.output_csv}")

if __name__ == "__main__":
    main() 