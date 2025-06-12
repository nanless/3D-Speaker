#!/usr/bin/env python3

import os
import random
import argparse
from collections import defaultdict

def load_spk2utt(spk2utt_path):
    """Load spk2utt file and categorize speakers by dataset prefix"""
    adult_spks = []  # VOX1_
    child_spks = []  # Online_20250530_
    spk2utts = defaultdict(list)
    
    with open(spk2utt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            spk = parts[0]
            utts = parts[1:]
            spk2utts[spk] = utts
            
            if spk.startswith('VOX1_'):
                adult_spks.append(spk)
            elif spk.startswith('Online_20250530_'):
                child_spks.append(spk)
    
    return spk2utts, adult_spks, child_spks

def create_positive_pairs(spk2utts, adult_spks, child_spks, num_positive_needed, target_ratio=0.9):
    """Create positive pairs (same speaker) with specified child vs adult ratio"""
    child_positive_pairs = []
    adult_positive_pairs = []
    
    # Create positive pairs for child speakers
    for spk in child_spks:
        if spk in spk2utts:
            utts = spk2utts[spk]
            if len(utts) >= 2:
                pairs = [(utts[i], utts[j]) 
                        for i in range(len(utts)) 
                        for j in range(i + 1, len(utts))]
                child_positive_pairs.extend(pairs)
    
    # Create positive pairs for adult speakers
    for spk in adult_spks:
        if spk in spk2utts:
            utts = spk2utts[spk]
            if len(utts) >= 2:
                pairs = [(utts[i], utts[j]) 
                        for i in range(len(utts)) 
                        for j in range(i + 1, len(utts))]
                adult_positive_pairs.extend(pairs)
    
    # Shuffle the pairs
    random.shuffle(child_positive_pairs)
    random.shuffle(adult_positive_pairs)
    
    # Calculate the number of pairs needed
    total_child_pairs = len(child_positive_pairs)
    total_adult_pairs = len(adult_positive_pairs)
    
    if total_child_pairs == 0:
        print("Warning: No child positive pairs available")
        # Repeat adult pairs if needed
        if num_positive_needed <= total_adult_pairs:
            return adult_positive_pairs[:num_positive_needed]
        else:
            repeated_adult_pairs = []
            while len(repeated_adult_pairs) < num_positive_needed:
                repeated_adult_pairs.extend(adult_positive_pairs)
            return repeated_adult_pairs[:num_positive_needed]
    
    # Calculate target numbers based on specified ratio
    target_child_pairs = int(num_positive_needed * target_ratio)
    target_adult_pairs = num_positive_needed - target_child_pairs
    
    # Select child pairs (with repetition if needed)
    if target_child_pairs <= total_child_pairs:
        selected_child_pairs = child_positive_pairs[:target_child_pairs]
    else:
        print(f"Warning: Requested {target_child_pairs} child pairs but only {total_child_pairs} available. Will repeat pairs.")
        selected_child_pairs = []
        while len(selected_child_pairs) < target_child_pairs:
            selected_child_pairs.extend(child_positive_pairs)
        selected_child_pairs = selected_child_pairs[:target_child_pairs]
    
    # Select adult pairs (with repetition if needed)
    if target_adult_pairs <= total_adult_pairs:
        selected_adult_pairs = adult_positive_pairs[:target_adult_pairs]
    else:
        print(f"Warning: Requested {target_adult_pairs} adult pairs but only {total_adult_pairs} available. Will repeat pairs.")
        selected_adult_pairs = []
        while len(selected_adult_pairs) < target_adult_pairs:
            selected_adult_pairs.extend(adult_positive_pairs)
        selected_adult_pairs = selected_adult_pairs[:target_adult_pairs]
    
    # Combine and shuffle
    all_positive_pairs = selected_child_pairs + selected_adult_pairs
    random.shuffle(all_positive_pairs)
    
    actual_total = len(selected_child_pairs) + len(selected_adult_pairs)
    child_percentage = len(selected_child_pairs) / actual_total * 100 if actual_total > 0 else 0
    print(f"Positive pairs: {len(selected_child_pairs)} child pairs ({child_percentage:.1f}%), {len(selected_adult_pairs)} adult pairs")
    
    return all_positive_pairs

def create_negative_pairs(spk2utts, adult_spks, child_spks, num_pairs, ratio=(0.8, 0.15, 0.05)):
    """Create negative pairs with specified ratios"""
    adult_child_ratio, child_child_ratio, adult_adult_ratio = ratio
    negative_pairs = []
    
    # Calculate number of pairs for each category
    n_adult_child = int(num_pairs * adult_child_ratio)
    n_child_child = int(num_pairs * child_child_ratio)
    n_adult_adult = int(num_pairs * adult_adult_ratio)
    
    # 1. Adult vs Child pairs (80%)
    for _ in range(n_adult_child):
        adult_spk = random.choice(adult_spks)
        child_spk = random.choice(child_spks)
        adult_utt = random.choice(spk2utts[adult_spk])
        child_utt = random.choice(spk2utts[child_spk])
        negative_pairs.append((adult_utt, child_utt))
    
    # 2. Child vs Child pairs (15%)
    for _ in range(n_child_child):
        child_spk1, child_spk2 = random.sample(child_spks, 2)
        child_utt1 = random.choice(spk2utts[child_spk1])
        child_utt2 = random.choice(spk2utts[child_spk2])
        negative_pairs.append((child_utt1, child_utt2))
    
    # 3. Adult vs Adult pairs (5%)
    for _ in range(n_adult_adult):
        adult_spk1, adult_spk2 = random.sample(adult_spks, 2)
        adult_utt1 = random.choice(spk2utts[adult_spk1])
        adult_utt2 = random.choice(spk2utts[adult_spk2])
        negative_pairs.append((adult_utt1, adult_utt2))
    
    return negative_pairs

def main():
    parser = argparse.ArgumentParser(description='Create trial file with specified ratios')
    parser.add_argument('--data-dir', required=True, help='Directory containing spk2utt file')
    parser.add_argument('--output-trials', required=True, help='Output trials file path')
    parser.add_argument('--num-trials', type=int, default=5000, help='Total number of trials to generate (default: 20000)')
    parser.add_argument('--pos-neg-ratio', default='1:1', help='Ratio of positive to negative pairs (e.g., 1:1, 1:2, 2:1, default: 1:1)')
    parser.add_argument('--pos-child-ratio', type=float, default=0.9, help='Ratio of child pairs in positive pairs (default: 0.9)')
    parser.add_argument('--neg-ratios', default='0.8:0.15:0.05', help='Ratios in negative pairs - adult_vs_child:child_vs_child:adult_vs_adult (default: 0.8:0.15:0.05)')
    args = parser.parse_args()
    
    # Parse positive:negative ratio
    try:
        pos_ratio, neg_ratio = map(int, args.pos_neg_ratio.split(':'))
        total_ratio = pos_ratio + neg_ratio
        num_positive = int(args.num_trials * pos_ratio / total_ratio)
        num_negative = args.num_trials - num_positive
    except ValueError:
        print(f"Error: Invalid pos-neg-ratio format '{args.pos_neg_ratio}'. Use format like '1:1' or '2:1'")
        return
    
    # Parse negative ratios
    try:
        neg_ratio_parts = list(map(float, args.neg_ratios.split(':')))
        if len(neg_ratio_parts) != 3:
            raise ValueError("Need exactly 3 ratios")
        
        # Normalize ratios to sum to 1
        total_neg_ratio = sum(neg_ratio_parts)
        if total_neg_ratio <= 0:
            raise ValueError("Ratios must be positive")
        
        adult_child_ratio = neg_ratio_parts[0] / total_neg_ratio
        child_child_ratio = neg_ratio_parts[1] / total_neg_ratio
        adult_adult_ratio = neg_ratio_parts[2] / total_neg_ratio
        negative_ratios = (adult_child_ratio, child_child_ratio, adult_adult_ratio)
        
    except ValueError as e:
        print(f"Error: Invalid neg-ratios format '{args.neg_ratios}'. Use format like '0.8:0.15:0.05' or '8:1.5:0.5'. {e}")
        return
    
    spk2utt_path = os.path.join(args.data_dir, 'spk2utt')
    spk2utts, adult_spks, child_spks = load_spk2utt(spk2utt_path)
    
    print(f"Creating {args.num_trials} trials with {num_positive} positive and {num_negative} negative pairs")
    print(f"Child:Adult ratio in positive pairs: {args.pos_child_ratio:.1f}:{1-args.pos_child_ratio:.1f}")
    print(f"Negative pairs ratios - Adult vs Child: {adult_child_ratio:.1%}, Child vs Child: {child_child_ratio:.1%}, Adult vs Adult: {adult_adult_ratio:.1%}")
    
    # Create positive pairs
    positive_pairs = create_positive_pairs(spk2utts, adult_spks, child_spks, num_positive, args.pos_child_ratio)
    
    # Create negative pairs
    negative_pairs = create_negative_pairs(spk2utts, adult_spks, child_spks, 
                                         num_negative, 
                                         ratio=negative_ratios)
    
    # Write trials file
    with open(args.output_trials, 'w') as f:
        # Write positive pairs
        for enroll, test in positive_pairs:
            f.write(f'{enroll} {test} target\n')
        
        # Write negative pairs
        for enroll, test in negative_pairs:
            f.write(f'{enroll} {test} nontarget\n')
    
    print(f'Created trial file with {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs')
    print(f'Trial file saved to: {args.output_trials}')

if __name__ == '__main__':
    main()
