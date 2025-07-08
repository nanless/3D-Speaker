#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import json
import pickle
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Load and demonstrate individual embedding files.')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing individual embedding files')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to analyze (optional)')
    parser.add_argument('--speaker_id', type=str, default=None,
                        help='Specific speaker to analyze (optional)')
    parser.add_argument('--utterance_id', type=str, default=None,
                        help='Specific utterance to analyze (optional)')
    parser.add_argument('--show_stats', action='store_true',
                        help='Show embedding statistics')
    
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

def show_embedding_stats(embedding, label="Embedding"):
    """Show statistics for an embedding."""
    print(f"\n{label} Statistics:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Mean: {np.mean(embedding):.6f}")
    print(f"  Std: {np.std(embedding):.6f}")
    print(f"  Min: {np.min(embedding):.6f}")
    print(f"  Max: {np.max(embedding):.6f}")
    print(f"  L2 Norm: {np.linalg.norm(embedding):.6f}")

def list_available_datasets(embeddings_dir):
    """List all available datasets."""
    datasets = set()
    
    # Check utterances directory
    utt_dir = os.path.join(embeddings_dir, 'utterances')
    if os.path.exists(utt_dir):
        for dataset_dir in Path(utt_dir).iterdir():
            if dataset_dir.is_dir():
                datasets.add(dataset_dir.name)
    
    # Check speakers directory
    spk_dir = os.path.join(embeddings_dir, 'speakers')
    if os.path.exists(spk_dir):
        for dataset_dir in Path(spk_dir).iterdir():
            if dataset_dir.is_dir():
                datasets.add(dataset_dir.name)
    
    return sorted(datasets)

def list_speakers_in_dataset(embeddings_dir, dataset):
    """List all speakers in a dataset."""
    speakers = set()
    
    # Check utterances directory
    utt_dataset_dir = os.path.join(embeddings_dir, 'utterances', dataset)
    if os.path.exists(utt_dataset_dir):
        for speaker_dir in Path(utt_dataset_dir).iterdir():
            if speaker_dir.is_dir():
                speakers.add(speaker_dir.name)
    
    # Check speakers directory
    spk_dataset_dir = os.path.join(embeddings_dir, 'speakers', dataset)
    if os.path.exists(spk_dataset_dir):
        for spk_file in Path(spk_dataset_dir).iterdir():
            if spk_file.suffix == '.pkl':
                speakers.add(spk_file.stem)
    
    return sorted(speakers)

def list_utterances_for_speaker(embeddings_dir, dataset, speaker_id):
    """List all utterances for a specific speaker."""
    utterances = []
    
    utt_speaker_dir = os.path.join(embeddings_dir, 'utterances', dataset, speaker_id)
    if os.path.exists(utt_speaker_dir):
        for utt_file in Path(utt_speaker_dir).iterdir():
            if utt_file.suffix == '.pkl':
                utterances.append(utt_file.stem)
    
    return sorted(utterances)

def compare_embeddings(embedding1, embedding2, label1="Embedding 1", label2="Embedding 2"):
    """Compare two embeddings."""
    print(f"\nComparing {label1} and {label2}:")
    
    # Cosine similarity
    cosine_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(embedding1 - embedding2)
    print(f"  Euclidean Distance: {euclidean_dist:.6f}")
    
    # Manhattan distance
    manhattan_dist = np.sum(np.abs(embedding1 - embedding2))
    print(f"  Manhattan Distance: {manhattan_dist:.6f}")

def main():
    args = parse_args()
    
    if not os.path.exists(args.embeddings_dir):
        print(f"Embeddings directory not found: {args.embeddings_dir}")
        return
    
    print(f"Loading embeddings from: {args.embeddings_dir}")
    
    # Load file index
    index = load_file_index(args.embeddings_dir)
    if index is None:
        print("Could not load file index. Proceeding without index...")
        index = {'utterances': {}, 'speakers': {}}
    
    print(f"File index loaded:")
    print(f"  Utterance files: {len(index['utterances'])}")
    print(f"  Speaker files: {len(index['speakers'])}")
    
    # List available datasets
    datasets = list_available_datasets(args.embeddings_dir)
    print(f"\nAvailable datasets ({len(datasets)}):")
    for dataset in datasets[:10]:  # Show first 10
        print(f"  - {dataset}")
    if len(datasets) > 10:
        print(f"  ... and {len(datasets) - 10} more")
    
    # If specific dataset is provided
    if args.dataset:
        if args.dataset not in datasets:
            print(f"Dataset '{args.dataset}' not found in available datasets")
            return
        
        print(f"\nAnalyzing dataset: {args.dataset}")
        
        # List speakers in dataset
        speakers = list_speakers_in_dataset(args.embeddings_dir, args.dataset)
        print(f"Speakers in {args.dataset} ({len(speakers)}):")
        for speaker in speakers[:5]:  # Show first 5
            print(f"  - {speaker}")
        if len(speakers) > 5:
            print(f"  ... and {len(speakers) - 5} more")
        
        # Load speaker embedding if available
        speaker_key = f"{args.dataset}_{args.speaker_id}" if args.speaker_id else f"{args.dataset}_{speakers[0]}"
        if speaker_key in index['speakers']:
            speaker_file = index['speakers'][speaker_key]
            embedding, info = load_individual_embedding(speaker_file)
            if embedding is not None:
                print(f"\nLoaded speaker embedding for: {speaker_key}")
                print(f"Speaker info: {info}")
                if args.show_stats:
                    show_embedding_stats(embedding, f"Speaker {speaker_key}")
        
        # If specific speaker is provided
        if args.speaker_id:
            if args.speaker_id not in speakers:
                print(f"Speaker '{args.speaker_id}' not found in dataset '{args.dataset}'")
                return
            
            print(f"\nAnalyzing speaker: {args.speaker_id}")
            
            # List utterances for speaker
            utterances = list_utterances_for_speaker(args.embeddings_dir, args.dataset, args.speaker_id)
            print(f"Utterances for {args.speaker_id} ({len(utterances)}):")
            for utterance in utterances[:5]:  # Show first 5
                print(f"  - {utterance}")
            if len(utterances) > 5:
                print(f"  ... and {len(utterances) - 5} more")
            
            # If specific utterance is provided
            if args.utterance_id:
                if args.utterance_id not in utterances:
                    print(f"Utterance '{args.utterance_id}' not found for speaker '{args.speaker_id}'")
                    return
                
                # Load utterance embedding
                utt_key = f"{args.dataset}_{args.speaker_id}_{args.utterance_id}"
                if utt_key in index['utterances']:
                    utt_file = index['utterances'][utt_key]
                    embedding, info = load_individual_embedding(utt_file)
                    if embedding is not None:
                        print(f"\nLoaded utterance embedding: {utt_key}")
                        print(f"Utterance info: {info}")
                        if args.show_stats:
                            show_embedding_stats(embedding, f"Utterance {utt_key}")
                        
                        # Compare with speaker embedding if available
                        speaker_key = f"{args.dataset}_{args.speaker_id}"
                        if speaker_key in index['speakers']:
                            spk_file = index['speakers'][speaker_key]
                            spk_embedding, spk_info = load_individual_embedding(spk_file)
                            if spk_embedding is not None:
                                compare_embeddings(embedding, spk_embedding, 
                                                 f"Utterance {args.utterance_id}", 
                                                 f"Speaker {args.speaker_id}")
            else:
                # Load and compare multiple utterances from the same speaker
                if len(utterances) >= 2:
                    print(f"\nComparing first two utterances from {args.speaker_id}:")
                    
                    utt1_key = f"{args.dataset}_{args.speaker_id}_{utterances[0]}"
                    utt2_key = f"{args.dataset}_{args.speaker_id}_{utterances[1]}"
                    
                    if utt1_key in index['utterances'] and utt2_key in index['utterances']:
                        emb1, info1 = load_individual_embedding(index['utterances'][utt1_key])
                        emb2, info2 = load_individual_embedding(index['utterances'][utt2_key])
                        
                        if emb1 is not None and emb2 is not None:
                            compare_embeddings(emb1, emb2, utterances[0], utterances[1])
                            
                            if args.show_stats:
                                show_embedding_stats(emb1, f"Utterance {utterances[0]}")
                                show_embedding_stats(emb2, f"Utterance {utterances[1]}")
    
    # Demonstrate loading random samples
    if not args.dataset and len(index['utterances']) > 0:
        print(f"\nLoading random samples...")
        
        # Load a few random utterances
        sample_keys = list(index['utterances'].keys())[:3]
        
        for key in sample_keys:
            file_path = index['utterances'][key]
            embedding, info = load_individual_embedding(file_path)
            if embedding is not None:
                print(f"\nSample utterance: {key}")
                print(f"Info: {info}")
                if args.show_stats:
                    show_embedding_stats(embedding, f"Sample {key}")

if __name__ == "__main__":
    main() 