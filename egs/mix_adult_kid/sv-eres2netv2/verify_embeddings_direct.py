#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# =====================
# 本脚本用于验证说话人嵌入（embedding）的区分能力，通过计算同说话人（intra-speaker）和不同说话人（inter-speaker）之间的相似度，评估嵌入的质量。
# 支持自动采样说话人和语音，输出详细的统计、可视化图表和评估指标。
# =====================

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

# 解析命令行参数
# 包括嵌入文件目录、采样说话人数、每人采样语音数、输出目录、随机种子、是否生成可视化图表等
# 返回参数对象

def parse_args():
    parser = argparse.ArgumentParser(description='Verify embedding quality by computing intra/inter-speaker similarities.')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing individual embedding files')
    parser.add_argument('--num_speakers', type=int, default=100,
                        help='Number of speakers to sample for analysis')
    parser.add_argument('--num_utterances_per_speaker', type=int, default=5,
                        help='Maximum number of utterances per speaker to use')
    parser.add_argument('--output_dir', type=str, default='verification_results',
                        help='Output directory for verification results')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--create_plots', action='store_true', default=True,
                        help='Create visualization plots')
    
    return parser.parse_args()

# 扫描嵌入文件目录，收集每个说话人的所有语音嵌入文件路径
# 目录结构假定为 embeddings_dir/utterances/数据集/说话人/语音.pkl
# 返回：{说话人key: [每个语音的字典信息]}
def scan_embedding_files(embeddings_dir):
    """Scan embedding files directly from directory structure."""
    speaker_utterances = defaultdict(list)
    
    utterances_dir = os.path.join(embeddings_dir, 'utterances')
    if not os.path.exists(utterances_dir):
        print(f"Utterances directory not found: {utterances_dir}")
        return speaker_utterances
    
    for dataset_dir in Path(utterances_dir).iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            # 说话人key由数据集名+说话人id组成，保证唯一性
            speaker_key = f"{dataset_name}_{speaker_id}"
            
            for pkl_file in speaker_dir.iterdir():
                if pkl_file.suffix == '.pkl':
                    utterance_id = pkl_file.stem
                    speaker_utterances[speaker_key].append({
                        'file_path': str(pkl_file),
                        'utterance_id': utterance_id,
                        'dataset': dataset_name,
                        'speaker_id': speaker_id
                    })
    
    return speaker_utterances

# 加载单个嵌入文件，返回嵌入向量和附加信息
def load_individual_embedding(file_path):
    """Load a single individual embedding file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data['embedding'], data['info']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# 从所有说话人中采样指定数量的说话人和每人指定数量的语音
# 只采样语音数>=2的说话人（便于做intra-speaker对比）
# 返回采样后的数据结构 {说话人: [语音信息列表]}
def sample_speakers_and_utterances(speaker_utterances, num_speakers, num_utterances_per_speaker, random_seed):
    """Sample speakers and their utterances for analysis."""
    random.seed(random_seed)
    
    # 只保留语音数>=2的说话人
    valid_speakers = {
        speaker: utterances for speaker, utterances in speaker_utterances.items()
        if len(utterances) >= 2  # 需要至少2条语音做同说话人对比
    }
    
    if len(valid_speakers) < num_speakers:
        print(f"Warning: Only {len(valid_speakers)} speakers have enough utterances")
        num_speakers = len(valid_speakers)
    
    # 随机采样说话人
    selected_speakers = random.sample(list(valid_speakers.keys()), num_speakers)
    
    # 每个说话人采样指定数量的语音
    sampled_data = {}
    for speaker in selected_speakers:
        utterances = valid_speakers[speaker]
        num_to_sample = min(num_utterances_per_speaker, len(utterances))
        sampled_utterances = random.sample(utterances, num_to_sample)
        sampled_data[speaker] = sampled_utterances
    
    return sampled_data

# 加载采样到的所有说话人的嵌入向量，返回结构：{说话人: {'embeddings': ndarray, 'utterances': [语音信息]}}
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

# 计算同说话人（intra-speaker）之间的相似度
# 返回：所有intra相似度数组、每个说话人的统计、详细配对信息
def compute_intra_speaker_similarities(speaker_embeddings):
    """Compute intra-speaker similarities."""
    intra_similarities = []
    speaker_stats = {}
    detailed_pairs = []
    
    for speaker, data in speaker_embeddings.items():
        embeddings = data['embeddings']
        utterances = data['utterances']
        
        if len(embeddings) < 2:
            continue
        
        # 计算同说话人所有语音两两之间的余弦相似度
        sim_matrix = cosine_similarity(embeddings)
        
        # 只取上三角（不含对角线），并记录配对信息
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = sim_matrix[i, j]
                intra_similarities.append(similarity)
                detailed_pairs.append({
                    'speaker': speaker,
                    'utterance1': utterances[i]['utterance_id'],
                    'utterance2': utterances[j]['utterance_id'],
                    'similarity': float(similarity),
                    'type': 'intra'
                })
        
        # 统计该说话人所有intra相似度的均值、方差、最小、最大
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        speaker_stats[speaker] = {
            'num_utterances': len(embeddings),
            'mean_similarity': float(np.mean(upper_triangle)),
            'std_similarity': float(np.std(upper_triangle)),
            'min_similarity': float(np.min(upper_triangle)),
            'max_similarity': float(np.max(upper_triangle))
        }
    
    return np.array(intra_similarities), speaker_stats, detailed_pairs

# 计算不同说话人（inter-speaker）之间的相似度
# 为避免组合过多，若说话人较多则随机采样部分说话人对
# 返回：所有inter相似度数组、详细配对信息
def compute_inter_speaker_similarities(speaker_embeddings, max_pairs=1000):
    """Compute inter-speaker similarities."""
    speakers = list(speaker_embeddings.keys())
    inter_similarities = []
    detailed_pairs = []
    
    # 采样说话人对，避免组合过多
    if len(speakers) > 100:  # 说话人多时采样
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
        utterances1 = speaker_embeddings[speaker1]['utterances']
        utterances2 = speaker_embeddings[speaker2]['utterances']
        
        # 计算两个说话人所有语音两两之间的余弦相似度
        cross_similarities = cosine_similarity(embeddings1, embeddings2)
        
        for i in range(len(embeddings1)):
            for j in range(len(embeddings2)):
                similarity = cross_similarities[i, j]
                inter_similarities.append(similarity)
                detailed_pairs.append({
                    'speaker1': speaker1,
                    'speaker2': speaker2,
                    'utterance1': utterances1[i]['utterance_id'],
                    'utterance2': utterances2[j]['utterance_id'],
                    'similarity': float(similarity),
                    'type': 'inter'
                })
    
    return np.array(inter_similarities), detailed_pairs

# 计算验证指标，包括：
# - 真接受率（TAR）、假接受率（FAR）、精度、F1分数、EER近似
# - 遍历不同阈值，统计每个阈值下的指标
# 返回：{阈值: 指标字典}
def compute_verification_metrics(intra_similarities, inter_similarities, thresholds=None):
    """Compute verification metrics."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.01)
    
    metrics = {}
    
    for threshold in thresholds:
        # intra >= 阈值视为同说话人识别正确（TP），否则为FN
        tp = np.sum(intra_similarities >= threshold)
        fn = np.sum(intra_similarities < threshold)
        # inter >= 阈值视为误识别（FP），否则为TN
        fp = np.sum(inter_similarities >= threshold)
        tn = np.sum(inter_similarities < threshold)
        
        # 计算各项指标
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

# 找到EER（等错误率）最小的最佳阈值
def find_optimal_threshold(metrics):
    """Find optimal threshold based on minimum EER."""
    min_eer = float('inf')
    optimal_threshold = 0.5
    
    for threshold, metric in metrics.items():
        if metric['eer'] < min_eer:
            min_eer = metric['eer']
            optimal_threshold = threshold
    
    return optimal_threshold, min_eer

# 生成可视化图表，包括：
# - 相似度分布直方图、箱线图、ROC曲线、EER随阈值变化曲线、详细分布对比
# 图表保存到输出目录
def create_visualization_plots(intra_similarities, inter_similarities, metrics, output_dir):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1：相似度分布
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(intra_similarities, bins=50, alpha=0.7, label='Intra-speaker', color='blue', density=True)
    plt.hist(inter_similarities, bins=50, alpha=0.7, label='Inter-speaker', color='red', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 图2：箱线图
    plt.subplot(2, 2, 2)
    data_to_plot = [intra_similarities, inter_similarities]
    labels = ['Intra-speaker', 'Inter-speaker']
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Box Plots')
    plt.grid(True, alpha=0.3)
    
    # 图3：ROC曲线
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
    
    # 图4：EER随阈值变化
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
    
    # 图5：详细分布对比
    plt.figure(figsize=(10, 6))
    
    # 叠加直方图
    bins = np.linspace(min(np.min(intra_similarities), np.min(inter_similarities)),
                      max(np.max(intra_similarities), np.max(inter_similarities)), 50)
    
    plt.hist(intra_similarities, bins=bins, alpha=0.6, label=f'Intra-speaker (n={len(intra_similarities)})', 
             color='blue', density=True)
    plt.hist(inter_similarities, bins=bins, alpha=0.6, label=f'Inter-speaker (n={len(inter_similarities)})', 
             color='red', density=True)
    
    # 均值虚线
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

# 打印部分intra/inter相似度的具体例子，便于人工检查
def show_similarity_examples(intra_pairs, inter_pairs, num_examples=5):
    """Show specific similarity examples."""
    print(f"\n=== SIMILARITY EXAMPLES ===")
    
    # 按相似度排序
    intra_pairs_sorted = sorted(intra_pairs, key=lambda x: x['similarity'], reverse=True)
    inter_pairs_sorted = sorted(inter_pairs, key=lambda x: x['similarity'], reverse=True)
    
    print(f"\nTop {num_examples} highest intra-speaker similarities (same speaker):")
    for i, pair in enumerate(intra_pairs_sorted[:num_examples]):
        print(f"  {i+1}. Speaker: {pair['speaker']}")
        print(f"     Utterances: {pair['utterance1']} <-> {pair['utterance2']}")
        print(f"     Similarity: {pair['similarity']:.4f}")
        print()
    
    print(f"Top {num_examples} highest inter-speaker similarities (different speakers):")
    for i, pair in enumerate(inter_pairs_sorted[:num_examples]):
        print(f"  {i+1}. Speakers: {pair['speaker1']} <-> {pair['speaker2']}")
        print(f"     Utterances: {pair['utterance1']} <-> {pair['utterance2']}")
        print(f"     Similarity: {pair['similarity']:.4f}")
        print()
    
    print(f"Bottom {num_examples} lowest intra-speaker similarities (same speaker):")
    for i, pair in enumerate(intra_pairs_sorted[-num_examples:]):
        print(f"  {i+1}. Speaker: {pair['speaker']}")
        print(f"     Utterances: {pair['utterance1']} <-> {pair['utterance2']}")
        print(f"     Similarity: {pair['similarity']:.4f}")
        print()

# 主流程：
# 1. 解析参数 2. 扫描嵌入文件 3. 采样说话人和语音 4. 加载嵌入 5. 计算intra/inter相似度
# 6. 计算评估指标 7. 输出统计和评估 8. 保存详细结果 9. 生成可视化图表
def main():
    args = parse_args()
    
    print("Starting embedding verification...")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 扫描嵌入文件
    print("Scanning embedding files...")
    speaker_utterances = scan_embedding_files(args.embeddings_dir)
    
    if not speaker_utterances:
        print("No embedding files found. Please check the directory structure.")
        return
    
    print(f"Found {len(speaker_utterances)} speakers")
    total_utterances = sum(len(utts) for utts in speaker_utterances.values())
    print(f"Total utterances: {total_utterances}")
    
    # 采样说话人和语音
    print(f"Sampling {args.num_speakers} speakers with up to {args.num_utterances_per_speaker} utterances each...")
    sampled_data = sample_speakers_and_utterances(
        speaker_utterances, args.num_speakers, args.num_utterances_per_speaker, args.random_seed
    )
    
    print(f"Selected {len(sampled_data)} speakers for analysis")
    
    # 加载嵌入
    print("Loading embeddings...")
    speaker_embeddings = load_embeddings_for_analysis(sampled_data)
    
    total_utterances_loaded = sum(len(data['embeddings']) for data in speaker_embeddings.values())
    print(f"Loaded embeddings for {len(speaker_embeddings)} speakers, {total_utterances_loaded} utterances")
    
    # 计算intra-speaker相似度
    print("Computing intra-speaker similarities...")
    intra_similarities, speaker_stats, intra_pairs = compute_intra_speaker_similarities(speaker_embeddings)
    print(f"Computed {len(intra_similarities)} intra-speaker similarity pairs")
    
    # 计算inter-speaker相似度
    print("Computing inter-speaker similarities...")
    inter_similarities, inter_pairs = compute_inter_speaker_similarities(speaker_embeddings)
    print(f"Computed {len(inter_similarities)} inter-speaker similarity pairs")
    
    # 计算验证指标
    print("Computing verification metrics...")
    metrics = compute_verification_metrics(intra_similarities, inter_similarities)
    
    # 找到最佳阈值
    optimal_threshold, min_eer = find_optimal_threshold(metrics)
    
    # 打印主要统计和评估结果
    print(f"\n=== EMBEDDING VERIFICATION RESULTS ===")
    print(f"Analysis Summary:")
    print(f"  Speakers analyzed: {len(speaker_embeddings)}")
    print(f"  Total utterances: {total_utterances_loaded}")
    print(f"  Intra-speaker pairs: {len(intra_similarities)}")
    print(f"  Inter-speaker pairs: {len(inter_similarities)}")
    
    print(f"\nIntra-speaker similarities (same speaker):")
    print(f"  Mean: {np.mean(intra_similarities):.4f}")
    print(f"  Std: {np.std(intra_similarities):.4f}")
    print(f"  Min: {np.min(intra_similarities):.4f}")
    print(f"  Max: {np.max(intra_similarities):.4f}")
    
    print(f"\nInter-speaker similarities (different speakers):")
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
    
    # 嵌入质量评估
    if mean_diff > 0.2:
        quality = "Excellent"
    elif mean_diff > 0.1:
        quality = "Good"
    elif mean_diff > 0.05:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"\nEmbedding Quality Assessment: {quality}")
    print(f"Interpretation:")
    if quality == "Excellent":
        print("  The embeddings show excellent discrimination between speakers.")
        print("  Same-speaker utterances are very similar, different speakers are well separated.")
    elif quality == "Good":
        print("  The embeddings show good speaker discrimination.")
        print("  Most same-speaker utterances are similar, with reasonable separation.")
    elif quality == "Fair":
        print("  The embeddings show moderate speaker discrimination.")
        print("  There is some overlap between same-speaker and different-speaker similarities.")
    else:
        print("  The embeddings show poor speaker discrimination.")
        print("  Significant overlap between same-speaker and different-speaker similarities.")
    
    # 打印详细相似度例子
    show_similarity_examples(intra_pairs, inter_pairs)
    
    # 保存详细结果到json
    results = {
        'summary': {
            'num_speakers': len(speaker_embeddings),
            'total_utterances': total_utterances_loaded,
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
        },
        'detailed_examples': {
            'top_intra_pairs': sorted(intra_pairs, key=lambda x: x['similarity'], reverse=True)[:10],
            'top_inter_pairs': sorted(inter_pairs, key=lambda x: x['similarity'], reverse=True)[:10],
            'bottom_intra_pairs': sorted(intra_pairs, key=lambda x: x['similarity'])[:10]
        }
    }
    
    results_file = os.path.join(args.output_dir, 'verification_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # 生成可视化图表
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