#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
基于所有utterance的说话人相似度计算与分析脚本
功能特性：
1. 基于utterance级别的相似度计算
2. 说话人验证评估（EER, minDCF等）
3. ROC曲线和DET曲线绘制
4. 相似度分布分析和可视化
5. 混淆矩阵分析
6. 多种距离度量支持
7. 详细的统计分析和报告
"""

import os
import sys
import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    """设置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Utterance-level speaker similarity analysis')
    parser.add_argument('--embeddings_dir', type=str, 
                        default='/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings',
                        help='Base embeddings directory')
    parser.add_argument('--utterances_subdir', type=str, default='embeddings_individual/utterances',
                        help='Subdirectory containing utterance embeddings')
    parser.add_argument('--output_dir', type=str, default='utterance_analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--num_workers', type=int, default=min(16, mp.cpu_count()),
                        help='Number of worker processes')
    parser.add_argument('--max_speakers', type=int, default=None,
                        help='Maximum number of speakers to process')
    parser.add_argument('--max_utterances_per_speaker', type=int, default=100,
                        help='Maximum utterances per speaker to use')
    parser.add_argument('--distance_metrics', nargs='+', 
                        default=['cosine', 'euclidean'],
                        choices=['cosine', 'euclidean', 'manhattan'],
                        help='Distance metrics to compute')
    parser.add_argument('--plot_format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format for plots')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved plots')
    parser.add_argument('--memory_efficient', action='store_true',
                        help='Use memory-efficient processing for large datasets')
    parser.add_argument('--chunk_size', type=int, default=5000,
                        help='Chunk size for memory-efficient processing')
    
    return parser.parse_args()

def load_utterance_embeddings_batch(utterance_batch):
    """批量加载utterance embeddings"""
    results = []
    
    for utt_info in utterance_batch:
        try:
            with open(utt_info['file_path'], 'rb') as f:
                data = pickle.load(f)
            
            if 'embedding' in data and data['embedding'] is not None:
                embedding = data['embedding']
                # 检查embedding有效性
                if not (np.isnan(embedding).any() or np.isinf(embedding).any()):
                    results.append({
                        'speaker_key': utt_info['speaker_key'],
                        'utterance_id': utt_info['utterance_id'],
                        'dataset': utt_info['dataset'],
                        'speaker_id': utt_info['speaker_id'],
                        'embedding': embedding,
                        'file_path': utt_info['file_path']
                    })
        except Exception as e:
            continue
    
    return results

def scan_utterance_files(utterances_dir, max_speakers=None, max_utterances_per_speaker=50):
    """扫描所有utterance文件"""
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning utterance files in: {utterances_dir}")
    
    speaker_utterances = defaultdict(list)
    speaker_count = 0
    
    for dataset_dir in Path(utterances_dir).iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        logger.info(f"Processing dataset: {dataset_name}")
        
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            if max_speakers and speaker_count >= max_speakers:
                break
                
            speaker_id = speaker_dir.name
            speaker_key = f"{dataset_name}_{speaker_id}"
            
            # 收集该说话人的utterance文件
            pkl_files = list(speaker_dir.glob('*.pkl'))
            
            # 限制每个说话人的utterance数量
            if len(pkl_files) > max_utterances_per_speaker:
                pkl_files = pkl_files[:max_utterances_per_speaker]
            
            for pkl_file in pkl_files:
                speaker_utterances[speaker_key].append({
                    'file_path': str(pkl_file),
                    'dataset': dataset_name,
                    'speaker_id': speaker_id,
                    'speaker_key': speaker_key,
                    'utterance_id': pkl_file.stem
                })
            
            speaker_count += 1
            
        if max_speakers and speaker_count >= max_speakers:
            break
    
    logger.info(f"Found {len(speaker_utterances)} speakers")
    total_utterances = sum(len(utts) for utts in speaker_utterances.values())
    logger.info(f"Total utterances: {total_utterances}")
    
    return speaker_utterances

def load_all_utterances_parallel(speaker_utterances, num_workers):
    """并行加载所有utterance embeddings"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading all utterance embeddings with {num_workers} workers")
    
    # 准备所有utterance信息
    all_utterances = []
    for speaker_key, utterances in speaker_utterances.items():
        all_utterances.extend(utterances)
    
    logger.info(f"Total utterances to load: {len(all_utterances)}")
    
    # 分批处理
    batch_size = max(1, len(all_utterances) // (num_workers * 4))
    utterance_data = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(all_utterances), batch_size):
            batch = all_utterances[i:i+batch_size]
            future = executor.submit(load_utterance_embeddings_batch, batch)
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading utterances"):
            try:
                batch_results = future.result()
                utterance_data.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch loading error: {e}")
                continue
    
    logger.info(f"Successfully loaded {len(utterance_data)} utterances")
    return utterance_data

def compute_distance_batch(args):
    """批量计算距离矩阵的一部分"""
    embeddings, metric, start_idx, end_idx = args
    n_samples = embeddings.shape[0]
    
    # 计算指定行范围的距离
    batch_distances = np.zeros((end_idx - start_idx, n_samples))
    
    for i, global_i in enumerate(range(start_idx, end_idx)):
        embedding_i = embeddings[global_i:global_i+1]  # 保持2D形状
        
        if metric == 'cosine':
            # 余弦相似度转换为距离
            similarities = cosine_similarity(embedding_i, embeddings)[0]
            distances = 1 - similarities
        elif metric == 'euclidean':
            distances = euclidean_distances(embedding_i, embeddings)[0]
        elif metric == 'manhattan':
            from sklearn.metrics.pairwise import manhattan_distances
            distances = manhattan_distances(embedding_i, embeddings)[0]
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        batch_distances[i] = distances
    
    return start_idx, end_idx, batch_distances

def compute_distance_matrix_parallel(embeddings, metric='cosine', num_workers=8):
    """并行计算距离矩阵"""
    logger = logging.getLogger(__name__)
    logger.info(f"Computing {metric} distance matrix with {num_workers} workers...")
    
    n_samples = embeddings.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    # 分批处理
    batch_size = max(1, n_samples // (num_workers * 4))
    batch_args = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_args.append((embeddings, metric, i, end_idx))
    
    # 并行计算
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_distance_batch, args) for args in batch_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Computing {metric} distances"):
            try:
                start_idx, end_idx, batch_distances = future.result()
                distance_matrix[start_idx:end_idx] = batch_distances
            except Exception as e:
                logger.error(f"Distance batch error: {e}")
                continue
    
    logger.info(f"{metric} distance matrix computation completed")
    return distance_matrix

def compute_distance_matrix(embeddings, metric='cosine', num_workers=1):
    """计算距离矩阵（支持并行和非并行）"""
    if num_workers > 1 and embeddings.shape[0] > 1000:
        # 大规模数据使用并行计算
        return compute_distance_matrix_parallel(embeddings, metric, num_workers)
    else:
        # 小规模数据使用单线程计算
        logger = logging.getLogger(__name__)
        logger.info(f"Computing {metric} distance matrix (single-threaded)...")
        
        if metric == 'cosine':
            # 余弦相似度转换为距离
            similarity_matrix = cosine_similarity(embeddings)
            distance_matrix = 1 - similarity_matrix
        elif metric == 'euclidean':
            distance_matrix = euclidean_distances(embeddings)
        elif metric == 'manhattan':
            from sklearn.metrics.pairwise import manhattan_distances
            distance_matrix = manhattan_distances(embeddings)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return distance_matrix

def evaluate_speaker_verification_memory_efficient(pairs, embeddings, metric, num_workers=8, chunk_size=5000):
    """内存高效的说话人验证评估（不存储完整距离矩阵）"""
    logger = logging.getLogger(__name__)
    logger.info(f"Memory-efficient evaluation for {metric} with chunk size {chunk_size}...")
    
    y_true = []
    y_scores = []
    
    # 分块处理pairs
    for i in tqdm(range(0, len(pairs), chunk_size), desc=f"Processing {metric} chunks"):
        chunk_pairs = pairs[i:i+chunk_size]
        
        # 收集这个chunk需要的embedding索引
        indices_needed = set()
        for idx1, idx2, _ in chunk_pairs:
            indices_needed.add(idx1)
            indices_needed.add(idx2)
        
        indices_list = sorted(list(indices_needed))
        chunk_embeddings = embeddings[indices_list]
        
        # 创建索引映射
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices_list)}
        
        # 计算这个chunk的距离矩阵
        chunk_distance_matrix = compute_distance_matrix(chunk_embeddings, metric, 1)  # 单线程避免嵌套并行
        
        # 计算分数
        for idx1, idx2, label in chunk_pairs:
            new_idx1 = index_map[idx1]
            new_idx2 = index_map[idx2]
            distance = chunk_distance_matrix[new_idx1, new_idx2]
            similarity_score = -distance
            
            y_true.append(label)
            y_scores.append(similarity_score)
        
        # 清理内存
        del chunk_distance_matrix
        gc.collect()
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 计算各种评估指标
    eer, eer_threshold = compute_eer(y_true, y_scores)
    min_dcf = compute_mindcf(y_true, y_scores)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    results = {
        'metric': metric,
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'min_dcf': float(min_dcf),
        'auc': float(roc_auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'y_true': y_true.tolist(),
        'y_scores': y_scores.tolist()
    }
    
    logger.info(f"{metric} - EER: {eer:.4f}, minDCF: {min_dcf:.4f}, AUC: {roc_auc:.4f}")
    return results

def generate_speaker_pairs(utterance_data):
    """生成说话人对（正样本和负样本）"""
    logger = logging.getLogger(__name__)
    logger.info("Generating speaker pairs for evaluation...")
    
    # 按说话人分组
    speaker_groups = defaultdict(list)
    for i, utt in enumerate(utterance_data):
        speaker_groups[utt['speaker_key']].append(i)
    
    positive_pairs = []
    negative_pairs = []
    
    # 生成正样本对（同一说话人的不同utterance）
    for speaker_key, indices in speaker_groups.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                positive_pairs.append((indices[i], indices[j], 1))
    
    # 生成负样本对（不同说话人的utterance）
    speaker_keys = list(speaker_groups.keys())
    neg_count = 0
    target_neg_pairs = len(positive_pairs) * 2  # 负样本数量为正样本的2倍
    
    import random
    random.seed(42)
    
    while neg_count < target_neg_pairs:
        # 随机选择两个不同的说话人
        speaker1, speaker2 = random.sample(speaker_keys, 2)
        idx1 = random.choice(speaker_groups[speaker1])
        idx2 = random.choice(speaker_groups[speaker2])
        
        negative_pairs.append((idx1, idx2, 0))
        neg_count += 1
    
    # 合并正负样本
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    logger.info(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
    return all_pairs

def compute_eer(y_true, y_scores):
    """计算等错误率(EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # 找到FPR和FNR相等的点
    eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer = 1. - interp1d(fpr, tpr)(eer_threshold)
    
    return eer, eer_threshold

def compute_mindcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
    """计算最小检测代价函数(minDCF)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf = np.min(dcf)
    
    return min_dcf

def compute_verification_scores_batch(args):
    """批量计算验证分数"""
    pairs_batch, distance_matrix = args
    
    y_true_batch = []
    y_scores_batch = []
    
    for idx1, idx2, label in pairs_batch:
        distance = distance_matrix[idx1, idx2]
        # 对于距离度量，距离越小表示越相似，所以用负距离作为相似度分数
        similarity_score = -distance
        
        y_true_batch.append(label)
        y_scores_batch.append(similarity_score)
    
    return y_true_batch, y_scores_batch

def evaluate_speaker_verification(pairs, distance_matrix, metric_name, num_workers=8):
    """评估说话人验证性能（支持并行）"""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating speaker verification for {metric_name}...")
    
    if len(pairs) > 10000 and num_workers > 1:
        # 大规模数据使用并行计算
        logger.info(f"Using parallel evaluation with {num_workers} workers...")
        
        # 分批处理
        batch_size = max(1, len(pairs) // (num_workers * 4))
        batch_args = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_args.append((batch_pairs, distance_matrix))
        
        y_true = []
        y_scores = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(compute_verification_scores_batch, args) for args in batch_args]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Computing {metric_name} scores"):
                try:
                    y_true_batch, y_scores_batch = future.result()
                    y_true.extend(y_true_batch)
                    y_scores.extend(y_scores_batch)
                except Exception as e:
                    logger.error(f"Verification batch error: {e}")
                    continue
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
    else:
        # 小规模数据使用单线程计算
        y_true = []
        y_scores = []
        
        for idx1, idx2, label in tqdm(pairs, desc=f"Computing {metric_name} scores"):
            distance = distance_matrix[idx1, idx2]
            # 对于距离度量，距离越小表示越相似，所以用负距离作为相似度分数
            similarity_score = -distance
            
            y_true.append(label)
            y_scores.append(similarity_score)
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
    
    # 计算各种评估指标
    eer, eer_threshold = compute_eer(y_true, y_scores)
    min_dcf = compute_mindcf(y_true, y_scores)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    results = {
        'metric': metric_name,
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'min_dcf': float(min_dcf),
        'auc': float(roc_auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'y_true': y_true.tolist(),
        'y_scores': y_scores.tolist()
    }
    
    logger.info(f"{metric_name} - EER: {eer:.4f}, minDCF: {min_dcf:.4f}, AUC: {roc_auc:.4f}")
    return results

def plot_roc_curves(evaluation_results, output_dir, plot_format='png', dpi=300):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    for result in evaluation_results:
        plt.plot(result['fpr'], result['tpr'], 
                label=f"{result['metric']} (AUC = {result['auc']:.3f})",
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Speaker Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'roc_curves.{plot_format}')
    plt.savefig(output_file, format=plot_format, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_det_curves(evaluation_results, output_dir, plot_format='png', dpi=300):
    """绘制DET曲线"""
    plt.figure(figsize=(10, 8))
    
    for result in evaluation_results:
        fpr = np.array(result['fpr'])
        fnr = 1 - np.array(result['tpr'])
        
        # DET曲线使用正态分位数变换
        fpr_norm = stats.norm.ppf(np.clip(fpr, 1e-8, 1-1e-8))
        fnr_norm = stats.norm.ppf(np.clip(fnr, 1e-8, 1-1e-8))
        
        plt.plot(fpr_norm, fnr_norm, 
                label=f"{result['metric']} (EER = {result['eer']:.3f})",
                linewidth=2)
    
    plt.xlabel('False Positive Rate (%)', fontsize=12)
    plt.ylabel('False Negative Rate (%)', fontsize=12)
    plt.title('DET Curves for Speaker Verification', fontsize=14, fontweight='bold')
    
    # 设置刻度标签
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    tick_locations = stats.norm.ppf(np.array(ticks) / 100)
    plt.xticks(tick_locations, ticks)
    plt.yticks(tick_locations, ticks)
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'det_curves.{plot_format}')
    plt.savefig(output_file, format=plot_format, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_score_distributions(evaluation_results, output_dir, plot_format='png', dpi=300):
    """绘制分数分布图"""
    for result in evaluation_results:
        plt.figure(figsize=(12, 6))
        
        y_true = np.array(result['y_true'])
        y_scores = np.array(result['y_scores'])
        
        # 分离正负样本的分数
        positive_scores = y_scores[y_true == 1]
        negative_scores = y_scores[y_true == 0]
        
        # 绘制直方图
        plt.hist(negative_scores, bins=50, alpha=0.7, label='Different Speakers', 
                color='red', density=True)
        plt.hist(positive_scores, bins=50, alpha=0.7, label='Same Speaker', 
                color='blue', density=True)
        
        # 添加EER阈值线
        plt.axvline(x=result['eer_threshold'], color='green', linestyle='--', 
                   label=f'EER Threshold ({result["eer_threshold"]:.3f})')
        
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Score Distribution - {result["metric"]}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'score_distribution_{result["metric"]}.{plot_format}')
        plt.savefig(output_file, format=plot_format, dpi=dpi, bbox_inches='tight')
        plt.close()

def analyze_speaker_statistics(utterance_data, output_dir):
    """分析说话人统计信息"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing speaker statistics...")
    
    # 统计每个说话人的utterance数量
    speaker_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    speaker_datasets = defaultdict(set)
    
    for utt in utterance_data:
        speaker_counts[utt['speaker_key']] += 1
        dataset_counts[utt['dataset']] += 1
        speaker_datasets[utt['dataset']].add(utt['speaker_key'])
    
    # 生成统计信息
    stats = {
        'total_speakers': len(speaker_counts),
        'total_utterances': len(utterance_data),
        'avg_utterances_per_speaker': np.mean(list(speaker_counts.values())),
        'std_utterances_per_speaker': np.std(list(speaker_counts.values())),
        'min_utterances_per_speaker': min(speaker_counts.values()),
        'max_utterances_per_speaker': max(speaker_counts.values()),
        'datasets': {
            dataset: {
                'utterances': count,
                'speakers': len(speakers)
            } for dataset, count in dataset_counts.items()
            for speakers in [speaker_datasets[dataset]]
        }
    }
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'speaker_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return stats

def plot_speaker_statistics(stats, output_dir, plot_format='png', dpi=300):
    """绘制说话人统计图表"""
    # 1. 数据集分布饼图
    plt.figure(figsize=(10, 8))
    datasets = list(stats['datasets'].keys())
    utterance_counts = [stats['datasets'][d]['utterances'] for d in datasets]
    
    plt.pie(utterance_counts, labels=datasets, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Utterances by Dataset', fontsize=14, fontweight='bold')
    plt.axis('equal')
    
    output_file = os.path.join(output_dir, f'dataset_distribution.{plot_format}')
    plt.savefig(output_file, format=plot_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. 说话人utterance数量分布
    plt.figure(figsize=(12, 6))
    speaker_counts = []
    for dataset_info in stats['datasets'].values():
        # 这里需要重新计算，因为stats中没有详细的每个说话人的utterance数
        pass
    
    # 简化版本：显示数据集中的说话人数量
    datasets = list(stats['datasets'].keys())
    speaker_counts = [stats['datasets'][d]['speakers'] for d in datasets]
    
    plt.bar(datasets, speaker_counts, color='skyblue', alpha=0.7)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Number of Speakers', fontsize=12)
    plt.title('Number of Speakers by Dataset', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'speakers_by_dataset.{plot_format}')
    plt.savefig(output_file, format=plot_format, dpi=dpi, bbox_inches='tight')
    plt.close()

def generate_analysis_report(evaluation_results, stats, output_dir):
    """生成分析报告"""
    logger = logging.getLogger(__name__)
    logger.info("Generating analysis report...")
    
    report = {
        'analysis_summary': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_speakers': stats['total_speakers'],
            'total_utterances': stats['total_utterances'],
            'datasets_analyzed': len(stats['datasets'])
        },
        'speaker_statistics': stats,
        'verification_results': {}
    }
    
    # 添加验证结果
    for result in evaluation_results:
        metric = result['metric']
        report['verification_results'][metric] = {
            'eer': result['eer'],
            'eer_threshold': result['eer_threshold'],
            'min_dcf': result['min_dcf'],
            'auc': result['auc']
        }
    
    # 保存报告
    report_file = os.path.join(output_dir, 'analysis_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    md_report = generate_markdown_report(report)
    md_file = os.path.join(output_dir, 'analysis_report.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    return report

def generate_markdown_report(report):
    """生成Markdown格式的报告"""
    md_content = f"""# Speaker Verification Analysis Report

## Analysis Summary
- **Analysis Time**: {report['analysis_summary']['timestamp']}
- **Total Speakers**: {report['analysis_summary']['total_speakers']:,}
- **Total Utterances**: {report['analysis_summary']['total_utterances']:,}
- **Datasets Analyzed**: {report['analysis_summary']['datasets_analyzed']}

## Speaker Statistics
- **Average Utterances per Speaker**: {report['speaker_statistics']['avg_utterances_per_speaker']:.2f}
- **Standard Deviation**: {report['speaker_statistics']['std_utterances_per_speaker']:.2f}
- **Min Utterances per Speaker**: {report['speaker_statistics']['min_utterances_per_speaker']}
- **Max Utterances per Speaker**: {report['speaker_statistics']['max_utterances_per_speaker']}

## Dataset Distribution
"""
    
    for dataset, info in report['speaker_statistics']['datasets'].items():
        md_content += f"- **{dataset}**: {info['utterances']:,} utterances, {info['speakers']:,} speakers\n"
    
    md_content += "\n## Verification Results\n\n"
    md_content += "| Metric | EER | EER Threshold | minDCF | AUC |\n"
    md_content += "|--------|-----|---------------|--------|----- |\n"
    
    for metric, results in report['verification_results'].items():
        md_content += f"| {metric} | {results['eer']:.4f} | {results['eer_threshold']:.4f} | {results['min_dcf']:.4f} | {results['auc']:.4f} |\n"
    
    md_content += "\n## Files Generated\n"
    md_content += "- `roc_curves.png` - ROC curves comparison\n"
    md_content += "- `det_curves.png` - DET curves comparison\n"
    md_content += "- `score_distribution_*.png` - Score distributions for each metric\n"
    md_content += "- `dataset_distribution.png` - Dataset distribution pie chart\n"
    md_content += "- `speakers_by_dataset.png` - Speakers count by dataset\n"
    md_content += "- `analysis_report.json` - Detailed analysis results\n"
    
    return md_content

def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging()
    
    start_time = time.time()
    logger.info("Starting utterance-level speaker similarity analysis...")
    logger.info(f"Configuration:")
    logger.info(f"  - Workers: {args.num_workers}")
    logger.info(f"  - Distance metrics: {args.distance_metrics}")
    logger.info(f"  - Max speakers: {args.max_speakers}")
    logger.info(f"  - Max utterances per speaker: {args.max_utterances_per_speaker}")
    logger.info(f"  - Memory efficient mode: {args.memory_efficient}")
    if args.memory_efficient:
        logger.info(f"  - Chunk size: {args.chunk_size}")
    
    # 设置路径
    utterances_dir = os.path.join(args.embeddings_dir, args.utterances_subdir)
    output_dir = os.path.join(args.embeddings_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Utterances directory: {utterances_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # 检查输入目录
    if not os.path.exists(utterances_dir):
        logger.error(f"Utterances directory not found: {utterances_dir}")
        return
    
    # 1. 扫描utterance文件
    logger.info("=== Stage 1: Scanning utterance files ===")
    speaker_utterances = scan_utterance_files(
        utterances_dir, args.max_speakers, args.max_utterances_per_speaker
    )
    if not speaker_utterances:
        logger.error("No utterances found!")
        return
    
    # 2. 加载所有utterance embeddings
    logger.info("=== Stage 2: Loading utterance embeddings ===")
    utterance_data = load_all_utterances_parallel(speaker_utterances, args.num_workers)
    if not utterance_data:
        logger.error("No valid utterances loaded!")
        return
    
    # 3. 分析说话人统计信息
    logger.info("=== Stage 3: Analyzing speaker statistics ===")
    stats = analyze_speaker_statistics(utterance_data, output_dir)
    plot_speaker_statistics(stats, output_dir, args.plot_format, args.dpi)
    
    # 4. 生成说话人对
    logger.info("=== Stage 4: Generating speaker pairs ===")
    pairs = generate_speaker_pairs(utterance_data)
    
    # 5. 计算距离矩阵和评估
    logger.info("=== Stage 5: Computing distances and evaluation ===")
    embeddings = np.array([utt['embedding'] for utt in utterance_data])
    evaluation_results = []
    
    for metric in args.distance_metrics:
        logger.info(f"Processing metric: {metric}")
        
        if args.memory_efficient:
            # 内存高效模式：不存储完整距离矩阵
            result = evaluate_speaker_verification_memory_efficient(
                pairs, embeddings, metric, args.num_workers, args.chunk_size
            )
        else:
            # 标准模式：计算完整距离矩阵
            distance_matrix = compute_distance_matrix(embeddings, metric, args.num_workers)
            
            # 评估说话人验证性能（并行）
            result = evaluate_speaker_verification(pairs, distance_matrix, metric, args.num_workers)
            
            # 清理内存
            del distance_matrix
            gc.collect()
        
        evaluation_results.append(result)
        
        # 保存详细结果
        result_file = os.path.join(output_dir, f'evaluation_results_{metric}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 6. 生成可视化图表
    logger.info("=== Stage 6: Generating visualizations ===")
    plot_roc_curves(evaluation_results, output_dir, args.plot_format, args.dpi)
    plot_det_curves(evaluation_results, output_dir, args.plot_format, args.dpi)
    plot_score_distributions(evaluation_results, output_dir, args.plot_format, args.dpi)
    
    # 7. 生成分析报告
    logger.info("=== Stage 7: Generating analysis report ===")
    report = generate_analysis_report(evaluation_results, stats, output_dir)
    
    # 打印总结
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("UTTERANCE-LEVEL ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {output_dir}")
    print(f"\nVerification Results:")
    for result in evaluation_results:
        print(f"  {result['metric']:>10}: EER={result['eer']:.4f}, minDCF={result['min_dcf']:.4f}, AUC={result['auc']:.4f}")
    print(f"{'='*80}")
    
    logger.info("Utterance-level speaker similarity analysis completed successfully!")

if __name__ == "__main__":
    main() 