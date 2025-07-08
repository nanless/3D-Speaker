#!/usr/bin/env python3
"""
Example script showing how to load and use the extracted embeddings.
"""

import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embeddings_dir='final_embeddings'):
    """Load all embedding files and metadata."""
    
    print(f"Loading embeddings from {embeddings_dir}...")
    
    # Load utterance embeddings
    with open(f'{embeddings_dir}/utterance_embeddings.pkl', 'rb') as f:
        utterance_embeddings = pickle.load(f)
    
    # Load speaker embeddings
    with open(f'{embeddings_dir}/speaker_embeddings.pkl', 'rb') as f:
        speaker_embeddings = pickle.load(f)
    
    # Load metadata
    with open(f'{embeddings_dir}/utterance_list.json', 'r') as f:
        utterance_list = json.load(f)
    
    with open(f'{embeddings_dir}/speaker_list.json', 'r') as f:
        speaker_list = json.load(f)
    
    with open(f'{embeddings_dir}/summary.json', 'r') as f:
        summary = json.load(f)
    
    return utterance_embeddings, speaker_embeddings, utterance_list, speaker_list, summary

def demonstrate_usage():
    """Demonstrate various ways to use the embeddings."""
    
    # Load embeddings
    utt_emb, spk_emb, utt_list, spk_list, summary = load_embeddings()
    
    print("=== Summary Statistics ===")
    print(f"Total utterances: {summary['total_utterances']}")
    print(f"Total speakers: {summary['total_speakers']}")
    print(f"Total datasets: {summary['total_datasets']}")
    print(f"Embedding dimension: {summary['embedding_dimension']}")
    print()
    
    print("Dataset breakdown:")
    for dataset, stats in summary['dataset_statistics'].items():
        print(f"  - {dataset}: {stats['speakers']} speakers, {stats['utterances']} utterances")
    print()
    
    # Example 1: Access individual utterance embedding
    print("=== Example 1: Access individual utterance ===")
    first_utt_key = list(utt_emb.keys())[0]
    first_utt = utt_emb[first_utt_key]
    print(f"Utterance key: {first_utt_key}")
    print(f"Dataset: {first_utt['dataset']}")
    print(f"Speaker ID: {first_utt['speaker_id']}")
    print(f"Utterance ID: {first_utt['utterance_id']}")
    print(f"Embedding shape: {first_utt['embedding'].shape}")
    print(f"Audio path: {first_utt['path']}")
    print()
    
    # Example 2: Access speaker-level embedding
    print("=== Example 2: Access speaker-level embedding ===")
    first_spk_key = list(spk_emb.keys())[0]
    first_spk = spk_emb[first_spk_key]
    print(f"Speaker key: {first_spk_key}")
    print(f"Dataset: {first_spk['dataset']}")
    print(f"Speaker ID: {first_spk['speaker_id']}")
    print(f"Number of utterances: {first_spk['num_utterances']}")
    print(f"Embedding shape: {first_spk['embedding'].shape}")
    print()
    
    # Example 3: Compute similarity between utterances
    print("=== Example 3: Compute utterance similarity ===")
    utt_keys = list(utt_emb.keys())[:5]  # Take first 5 utterances
    embeddings_matrix = np.array([utt_emb[key]['embedding'] for key in utt_keys])
    
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings_matrix)
    
    print("Cosine similarity matrix (first 5 utterances):")
    for i, key1 in enumerate(utt_keys):
        print(f"{key1[:30]:<30}", end=" ")
        for j, key2 in enumerate(utt_keys):
            print(f"{sim_matrix[i,j]:.3f}", end=" ")
        print()
    print()
    
    # Example 4: Find utterances from same speaker
    print("=== Example 4: Find utterances from same speaker ===")
    # Group utterances by speaker
    speaker_utterances = {}
    for utt_key, utt_data in utt_emb.items():
        spk_key = f"{utt_data['dataset']}_{utt_data['speaker_id']}"
        if spk_key not in speaker_utterances:
            speaker_utterances[spk_key] = []
        speaker_utterances[spk_key].append(utt_key)
    
    # Find a speaker with multiple utterances
    multi_utt_speakers = {k: v for k, v in speaker_utterances.items() if len(v) > 1}
    if multi_utt_speakers:
        example_speaker = list(multi_utt_speakers.keys())[0]
        example_utterances = multi_utt_speakers[example_speaker][:3]  # Take first 3
        
        print(f"Speaker: {example_speaker}")
        print(f"Utterances: {len(speaker_utterances[example_speaker])}")
        print("Sample utterances:")
        for utt_key in example_utterances:
            utt_data = utt_emb[utt_key]
            print(f"  - {utt_key}: {utt_data['path']}")
        
        # Compute similarity between utterances from same speaker
        same_spk_embeddings = np.array([utt_emb[key]['embedding'] for key in example_utterances])
        same_spk_sim = cosine_similarity(same_spk_embeddings)
        
        print(f"Average intra-speaker similarity: {np.mean(same_spk_sim[np.triu_indices(len(same_spk_sim), k=1)]):.3f}")
    print()
    
    # Example 5: Compare speaker embeddings
    print("=== Example 5: Compare speaker embeddings ===")
    spk_keys = list(spk_emb.keys())[:3]  # Take first 3 speakers
    spk_embeddings_matrix = np.array([spk_emb[key]['embedding'] for key in spk_keys])
    spk_sim_matrix = cosine_similarity(spk_embeddings_matrix)
    
    print("Speaker similarity matrix (first 3 speakers):")
    for i, key1 in enumerate(spk_keys):
        print(f"{key1[:30]:<30}", end=" ")
        for j, key2 in enumerate(spk_keys):
            print(f"{spk_sim_matrix[i,j]:.3f}", end=" ")
        print()
    print()
    
    print("=== Usage Examples Complete ===")

def search_by_criteria(dataset=None, speaker_id=None, embeddings_dir='final_embeddings'):
    """Search embeddings by specific criteria."""
    
    utt_emb, spk_emb, _, _, _ = load_embeddings(embeddings_dir)
    
    # Search utterances
    matching_utterances = []
    for utt_key, utt_data in utt_emb.items():
        match = True
        if dataset and utt_data['dataset'] != dataset:
            match = False
        if speaker_id and utt_data['speaker_id'] != speaker_id:
            match = False
        
        if match:
            matching_utterances.append((utt_key, utt_data))
    
    # Search speakers
    matching_speakers = []
    for spk_key, spk_data in spk_emb.items():
        match = True
        if dataset and spk_data['dataset'] != dataset:
            match = False
        if speaker_id and spk_data['speaker_id'] != speaker_id:
            match = False
        
        if match:
            matching_speakers.append((spk_key, spk_data))
    
    return matching_utterances, matching_speakers

if __name__ == "__main__":
    demonstrate_usage() 