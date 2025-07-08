#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
高性能说话人相似度计算脚本
优化特性：
1. 多进程并行处理
2. 批量文件加载
3. 进度保存和恢复
4. 内存优化
5. 分批处理支持
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
from sklearn.metrics.pairwise import cosine_similarity
import logging
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# 设置日志记录器，方便后续调试和信息输出
# 该函数会配置日志的输出格式和级别
# 返回一个logger对象用于全局调用

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# 解析命令行参数，支持自定义输入输出路径、并行进程数、批量大小等
# 方便在不同环境和需求下灵活调整脚本行为

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Fast speaker similarity computation')
    parser.add_argument('--embeddings_dir', type=str, 
                        default='/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings',
                        help='Base embeddings directory')
    parser.add_argument('--utterances_subdir', type=str, default='embeddings_individual/utterances',
                        help='Subdirectory containing utterance embeddings')
    parser.add_argument('--speakers_output_subdir', type=str, default='embeddings_individual/speakers',
                        help='Subdirectory for speaker output')
    parser.add_argument('--similarities_output_subdir', type=str, default='speaker_similarity_analysis',
                        help='Subdirectory for similarity results')
    parser.add_argument('--num_workers', type=int, default=min(32, mp.cpu_count()),
                        help='Number of worker processes')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for processing speakers')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous progress')
    parser.add_argument('--max_speakers', type=int, default=None,
                        help='Maximum number of speakers to process (for testing)')
    parser.add_argument('--skip_similarity', action='store_true',
                        help='Skip similarity computation (only process speaker embeddings)')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top similar speakers to compute for each speaker')
    
    return parser.parse_args()

# 并行处理一批说话人的embedding，计算每个说话人的平均embedding
# 返回每个说话人的平均embedding和相关信息
# 该函数会被多进程池调用

def load_speaker_embeddings_batch(speaker_batch):
    """并行处理一批说话人的embedding"""
    result = {}
    info_result = {}
    
    for speaker_key, utterances in speaker_batch:
        try:
            embeddings = []  # 存储该说话人的所有utterance的embedding
            valid_utterances = []  # 存储有效的utterance信息
            dataset_name = utterances[0]['dataset']  # 数据集名称
            speaker_id = utterances[0]['speaker_id']  # 说话人ID
            
            # 批量加载该说话人的所有embedding
            for utt_info in utterances:
                try:
                    with open(utt_info['file_path'], 'rb') as f:
                        data = pickle.load(f)
                    if 'embedding' in data and data['embedding'] is not None:
                        embedding = data['embedding']
                        # 检查embedding是否包含NaN或无穷大值
                        if not (np.isnan(embedding).any() or np.isinf(embedding).any()):
                            embeddings.append(embedding)
                            valid_utterances.append(utt_info)
                        else:
                            logging.warning(f"Utterance {utt_info['utterance_id']} has invalid embedding (NaN/Inf), skipping")
                except Exception as e:
                    continue  # 跳过损坏的文件
            
            if embeddings:
                # 计算平均embedding
                avg_embedding = np.mean(embeddings, axis=0)
                # 再次检查平均embedding是否有效
                if not (np.isnan(avg_embedding).any() or np.isinf(avg_embedding).any()):
                    result[speaker_key] = avg_embedding
                    
                    # 收集说话人信息，便于后续追溯和分析
                    info_result[speaker_key] = {
                        'dataset': dataset_name,
                        'speaker_id': speaker_id,
                        'speaker_key': speaker_key,
                        'num_utterances': len(valid_utterances),
                        'embedding_dim': len(avg_embedding),
                        'utterances': [
                            {
                                'utterance_id': utt['utterance_id'],
                                'file_path': utt['file_path']
                            } for utt in valid_utterances[:10]  # 只保留前10个，减少内存
                        ]
                    }
                else:
                    logging.warning(f"Speaker {speaker_key} has invalid average embedding (NaN/Inf), skipping")
        except Exception as e:
            logging.error(f"Error processing speaker {speaker_key}: {e}")
            continue
    
    return result, info_result

# 快速扫描所有说话人的语音文件，构建说话人到utterance的映射
# 返回一个字典，key为说话人，value为该说话人的utterance信息列表

def scan_speaker_utterances_fast(utterances_dir, max_speakers=None):
    """快速扫描所有说话人的语音文件"""
    logger = logging.getLogger(__name__)
    logger.info(f"Fast scanning utterances in: {utterances_dir}")
    
    speaker_utterances = defaultdict(list)  # 说话人到utterance的映射
    speaker_count = 0  # 计数器，支持最大说话人数限制
    
    for dataset_dir in Path(utterances_dir).iterdir():  # 遍历每个数据集目录
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        logger.info(f"Processing dataset: {dataset_name}")
        
        for speaker_dir in dataset_dir.iterdir():  # 遍历每个说话人目录
            if not speaker_dir.is_dir():
                continue
                
            if max_speakers and speaker_count >= max_speakers:
                break
                
            speaker_id = speaker_dir.name
            speaker_key = speaker_id
            
            # 快速收集该说话人的所有pkl文件
            pkl_files = list(speaker_dir.glob('*.pkl'))
            for pkl_file in pkl_files:
                speaker_utterances[speaker_key].append({
                    'file_path': str(pkl_file),
                    'dataset': dataset_name,
                    'speaker_id': speaker_id,
                    'utterance_id': pkl_file.stem
                })
            
            speaker_count += 1
            
        if max_speakers and speaker_count >= max_speakers:
            break
    
    logger.info(f"Found {len(speaker_utterances)} speakers")
    total_utterances = sum(len(utts) for utts in speaker_utterances.values())
    logger.info(f"Total utterances: {total_utterances}")
    
    return speaker_utterances

# 保存处理进度到json文件，便于断点续跑
# 包含已处理的说话人embedding、信息和已处理的说话人列表

def save_progress(speaker_embeddings, speaker_info, progress_file):
    """保存进度"""
    progress_data = {
        'speaker_embeddings': {k: v.tolist() for k, v in speaker_embeddings.items()},
        'speaker_info': speaker_info,
        'processed_speakers': list(speaker_embeddings.keys())
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

# 加载进度文件，恢复已处理的说话人embedding和信息
# 若文件不存在或损坏，则返回空结果

def load_progress(progress_file):
    """加载进度"""
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        speaker_embeddings = {k: np.array(v) for k, v in progress_data['speaker_embeddings'].items()}
        speaker_info = progress_data['speaker_info']
        processed_speakers = set(progress_data['processed_speakers'])
        
        return speaker_embeddings, speaker_info, processed_speakers
    except:
        return {}, {}, set()

# 并行计算所有说话人的平均embedding，支持断点续跑
# 每处理一定批次会自动保存进度，防止中途意外丢失
# 返回所有说话人的embedding和信息

def compute_speaker_embeddings_parallel(speaker_utterances, num_workers, batch_size, progress_file=None):
    """并行计算说话人embeddings"""
    logger = logging.getLogger(__name__)
    logger.info(f"Computing speaker embeddings with {num_workers} workers, batch size {batch_size}")
    
    # 加载之前的进度
    speaker_embeddings, speaker_info, processed_speakers = {}, {}, set()
    if progress_file and os.path.exists(progress_file):
        speaker_embeddings, speaker_info, processed_speakers = load_progress(progress_file)
        logger.info(f"Resumed from progress: {len(processed_speakers)} speakers already processed")
    
    # 过滤已处理的说话人
    remaining_speakers = [(k, v) for k, v in speaker_utterances.items() if k not in processed_speakers]
    
    if not remaining_speakers:
        logger.info("All speakers already processed!")
        return speaker_embeddings, speaker_info
    
    logger.info(f"Processing {len(remaining_speakers)} remaining speakers...")
    
    # 分批处理，避免单批过大导致内存溢出
    total_batches = (len(remaining_speakers) + batch_size - 1) // batch_size
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有批次任务
        futures = []
        for i in range(0, len(remaining_speakers), batch_size):
            batch = remaining_speakers[i:i+batch_size]
            future = executor.submit(load_speaker_embeddings_batch, batch)
            futures.append(future)
        
        # 收集结果，as_completed保证结果顺序与提交无关
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing batches")):
            try:
                batch_embeddings, batch_info = future.result()
                speaker_embeddings.update(batch_embeddings)
                speaker_info.update(batch_info)
                
                # 每10个批次保存一次进度，防止长时间未保存导致数据丢失
                if progress_file and (i + 1) % 10 == 0:
                    save_progress(speaker_embeddings, speaker_info, progress_file)
                    logger.info(f"Progress saved: {len(speaker_embeddings)} speakers processed")
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                continue
    
    # 最终保存进度
    if progress_file:
        save_progress(speaker_embeddings, speaker_info, progress_file)
    
    logger.info(f"Computed embeddings for {len(speaker_embeddings)} speakers")
    return speaker_embeddings, speaker_info

# 批量保存每个说话人的embedding和信息到指定目录
# embedding以pkl格式保存，信息以json格式保存，便于后续分析和复用

def save_speaker_files_fast(speaker_embeddings, speaker_info, speakers_output_dir):
    """快速保存说话人文件"""
    logger = logging.getLogger(__name__)
    logger.info(f"Saving {len(speaker_embeddings)} speaker files to: {speakers_output_dir}")
    
    os.makedirs(speakers_output_dir, exist_ok=True)
    
    # 预创建所有数据集目录，避免多进程下重复创建
    datasets = set(info['dataset'] for info in speaker_info.values())
    for dataset in datasets:
        os.makedirs(os.path.join(speakers_output_dir, dataset), exist_ok=True)
    
    # 批量保存文件
    for speaker_key, embedding in tqdm(speaker_embeddings.items(), desc="Saving speaker files"):
        info = speaker_info[speaker_key]
        dataset_name = info['dataset']
        speaker_id = info['speaker_id']
        
        dataset_dir = os.path.join(speakers_output_dir, dataset_name)
        
        # 保存embedding文件（pkl格式，包含embedding和info）
        embedding_file = os.path.join(dataset_dir, f"{speaker_id}.pkl")
        embedding_data = {
            'embedding': embedding,
            'info': info
        }
        with open(embedding_file, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        # 保存信息文件（json格式，便于查看）
        info_file = os.path.join(dataset_dir, f"{speaker_id}_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

# 批量计算一组说话人的相似度，返回该批次的相似度字典
# 用于多进程并行加速

def compute_similarities_batch(args):
    """批量计算相似度"""
    speaker_keys, embeddings_matrix, start_idx, end_idx = args
    
    batch_similarities = {}
    
    for i in range(start_idx, end_idx):
        speaker1 = speaker_keys[i]
        embedding1 = embeddings_matrix[i:i+1]  # 保持2D形状
        
        # 计算与所有其他说话人的相似度（余弦相似度）
        similarities = cosine_similarity(embedding1, embeddings_matrix)[0]
        
        batch_similarities[speaker1] = {
            speaker_keys[j]: float(similarities[j]) for j in range(len(speaker_keys))
        }
    
    return batch_similarities

# 并行计算所有说话人之间的相似度，返回相似度字典、矩阵和说话人key列表
# 支持大规模说话人并行加速

def compute_speaker_similarities_fast(speaker_embeddings, num_workers):
    """快速计算说话人相似度"""
    logger = logging.getLogger(__name__)
    logger.info(f"Computing similarities for {len(speaker_embeddings)} speakers with {num_workers} workers")
    
    # 过滤掉包含NaN或无穷大值的embedding
    valid_speakers = {}
    invalid_count = 0
    
    for speaker_key, embedding in speaker_embeddings.items():
        if not (np.isnan(embedding).any() or np.isinf(embedding).any()):
            valid_speakers[speaker_key] = embedding
        else:
            invalid_count += 1
            logger.warning(f"Speaker {speaker_key} has invalid embedding (NaN/Inf), excluding from similarity computation")
    
    if invalid_count > 0:
        logger.warning(f"Excluded {invalid_count} speakers with invalid embeddings")
    
    if len(valid_speakers) == 0:
        logger.error("No valid speakers found for similarity computation!")
        return {}, np.array([]), []
    
    logger.info(f"Computing similarities for {len(valid_speakers)} valid speakers")
    
    speaker_keys = list(valid_speakers.keys())  # 所有有效说话人key
    embeddings_matrix = np.array([valid_speakers[key] for key in speaker_keys])  # 所有有效说话人embedding矩阵
    
    # 最后一次检查矩阵是否包含NaN
    if np.isnan(embeddings_matrix).any() or np.isinf(embeddings_matrix).any():
        logger.error("Embeddings matrix still contains NaN/Inf values after filtering!")
        return {}, np.array([]), []
    
    # 分批计算相似度，避免单批过大导致内存溢出
    batch_size = max(1, len(speaker_keys) // (num_workers * 4))
    similarities = {}
    
    # 准备批次参数，每个批次处理一部分说话人
    batch_args = []
    for i in range(0, len(speaker_keys), batch_size):
        end_idx = min(i + batch_size, len(speaker_keys))
        batch_args.append((speaker_keys, embeddings_matrix, i, end_idx))
    
    # 并行计算
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_similarities_batch, args) for args in batch_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing similarities"):
            try:
                batch_result = future.result()
                similarities.update(batch_result)
            except Exception as e:
                logger.error(f"Similarity batch error: {e}")
                continue
    
    # 计算完整的相似度矩阵（二维数组，便于后续分析）
    try:
        similarity_matrix = cosine_similarity(embeddings_matrix)
        logger.info(f"Computed similarities for {len(speaker_keys)} speakers")
        return similarities, similarity_matrix, speaker_keys
    except Exception as e:
        logger.error(f"Error computing similarity matrix: {e}")
        return similarities, np.array([]), speaker_keys

# 保存相似度结果，包括相似度字典、矩阵、key映射和统计信息
# 支持大规模数据分块保存，避免内存溢出

def save_similarity_results_fast(similarities, similarity_matrix, speaker_keys, similarities_output_dir):
    """快速保存相似度结果"""
    logger = logging.getLogger(__name__)
    logger.info(f"Saving similarity results to: {similarities_output_dir}")
    
    os.makedirs(similarities_output_dir, exist_ok=True)
    
    # 保存相似度字典
    similarities_file = os.path.join(similarities_output_dir, 'speaker_similarities.json')
    with open(similarities_file, 'w', encoding='utf-8') as f:
        json.dump(similarities, f, indent=2, ensure_ascii=False)
    
    # 保存相似度矩阵（npy格式，便于后续快速加载）
    matrix_file = os.path.join(similarities_output_dir, 'similarity_matrix.npy')
    np.save(matrix_file, similarity_matrix)
    
    # 保存说话人key映射，便于后续查找
    keys_mapping_file = os.path.join(similarities_output_dir, 'speaker_keys_mapping.json')
    keys_mapping = {i: key for i, key in enumerate(speaker_keys)}
    with open(keys_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(keys_mapping, f, indent=2, ensure_ascii=False)
    
    # 生成统计信息，便于整体评估
    stats = {
        'total_speakers': len(speaker_keys),
        'similarity_matrix_shape': list(similarity_matrix.shape),
        'mean_similarity': float(np.mean(similarity_matrix)),
        'std_similarity': float(np.std(similarity_matrix)),
        'min_similarity': float(np.min(similarity_matrix)),
        'max_similarity': float(np.max(similarity_matrix))
    }
    
    stats_file = os.path.join(similarities_output_dir, 'similarity_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Similarity results saved successfully")
    return stats

# 详细分析相似度矩阵，计算各种统计信息和排名
# 包括上三角矩阵分析、每个说话人的top相似说话人等

def analyze_similarity_matrix(similarity_matrix, speaker_keys, similarities_output_dir, top_k=100):
    """详细分析相似度矩阵"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting detailed similarity matrix analysis...")
    
    n_speakers = len(speaker_keys)
    
    # 1. 提取上三角矩阵（排除对角线）
    logger.info("Extracting upper triangular matrix...")
    upper_tri_indices = np.triu_indices(n_speakers, k=1)  # k=1排除对角线
    upper_tri_similarities = similarity_matrix[upper_tri_indices]
    
    # 2. 计算上三角矩阵的统计信息
    logger.info("Computing upper triangular matrix statistics...")
    upper_tri_stats = {
        'total_pairs': len(upper_tri_similarities),
        'mean_similarity': float(np.mean(upper_tri_similarities)),
        'std_similarity': float(np.std(upper_tri_similarities)),
        'min_similarity': float(np.min(upper_tri_similarities)),
        'max_similarity': float(np.max(upper_tri_similarities)),
        'median_similarity': float(np.median(upper_tri_similarities)),
        'q25_similarity': float(np.percentile(upper_tri_similarities, 25)),
        'q75_similarity': float(np.percentile(upper_tri_similarities, 75))
    }
    
    # 3. 找到最相似的说话人对 (Top 1000)
    logger.info("Finding top 1000 most similar speaker pairs...")
    # 按相似度降序排序，取前1000个
    sorted_indices = np.argsort(upper_tri_similarities)[::-1]  # 降序
    top_n_similar = min(1000, len(upper_tri_similarities))  # 防止超出范围
    
    most_similar_pairs = []
    for i in range(top_n_similar):
        idx = sorted_indices[i]
        pair_i, pair_j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        most_similar_pairs.append({
            'rank': i + 1,
            'speaker1': speaker_keys[pair_i],
            'speaker2': speaker_keys[pair_j],
            'similarity': float(upper_tri_similarities[idx])
        })
    
    # 4. 找到最不相似的说话人对 (Bottom 1000)
    logger.info("Finding top 1000 least similar speaker pairs...")
    # 按相似度升序排序，取前1000个
    bottom_n_similar = min(1000, len(upper_tri_similarities))  # 防止超出范围
    
    least_similar_pairs = []
    for i in range(bottom_n_similar):
        idx = sorted_indices[-(i+1)]  # 从末尾开始取
        pair_i, pair_j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        least_similar_pairs.append({
            'rank': i + 1,
            'speaker1': speaker_keys[pair_i],
            'speaker2': speaker_keys[pair_j],
            'similarity': float(upper_tri_similarities[idx])
        })
    
    # 5. 计算每个说话人的top-k最相似说话人
    logger.info(f"Computing top-{top_k} similar speakers for each speaker...")
    speaker_top_similarities = {}
    
    for i, speaker_key in enumerate(tqdm(speaker_keys, desc="Computing top similarities")):
        # 获取该说话人与所有其他说话人的相似度
        speaker_similarities = similarity_matrix[i]
        
        # 排除自己（对角线元素）
        other_similarities = np.concatenate([speaker_similarities[:i], speaker_similarities[i+1:]])
        other_speaker_keys = speaker_keys[:i] + speaker_keys[i+1:]
        
        # 找到top-k最相似的说话人
        top_indices = np.argsort(other_similarities)[-top_k:][::-1]  # 降序排列
        
        top_similar_speakers = []
        for idx in top_indices:
            top_similar_speakers.append({
                'speaker': other_speaker_keys[idx],
                'similarity': float(other_similarities[idx])
            })
        
        speaker_top_similarities[speaker_key] = {
            'top_similarities': top_similar_speakers,
            'mean_similarity_to_others': float(np.mean(other_similarities)),
            'max_similarity_to_others': float(np.max(other_similarities)),
            'min_similarity_to_others': float(np.min(other_similarities))
        }
    
    # 6. 计算相似度分布直方图数据
    logger.info("Computing similarity distribution...")
    hist_counts, hist_bins = np.histogram(upper_tri_similarities, bins=50)
    similarity_distribution = {
        'bins': hist_bins.tolist(),
        'counts': hist_counts.tolist()
    }
    
    # 7. 找到每个相似度阈值下的说话人对数量
    thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    threshold_stats = {}
    for threshold in thresholds:
        count = np.sum(upper_tri_similarities >= threshold)
        percentage = (count / len(upper_tri_similarities)) * 100
        threshold_stats[f"threshold_{threshold}"] = {
            'count': int(count),
            'percentage': float(percentage)
        }
    
    # 8. 保存分析结果
    logger.info("Saving analysis results...")
    
    # 保存上三角矩阵统计
    upper_tri_stats_file = os.path.join(similarities_output_dir, 'upper_triangular_statistics.json')
    with open(upper_tri_stats_file, 'w', encoding='utf-8') as f:
        json.dump(upper_tri_stats, f, indent=2, ensure_ascii=False)
    
    # 保存最相似和最不相似的说话人对
    extreme_pairs_file = os.path.join(similarities_output_dir, 'extreme_similarity_pairs.json')
    extreme_pairs = {
        'most_similar_pairs': most_similar_pairs,
        'least_similar_pairs': least_similar_pairs,
        'metadata': {
            'num_most_similar': len(most_similar_pairs),
            'num_least_similar': len(least_similar_pairs),
            'total_pairs_analyzed': len(upper_tri_similarities)
        }
    }
    with open(extreme_pairs_file, 'w', encoding='utf-8') as f:
        json.dump(extreme_pairs, f, indent=2, ensure_ascii=False)
    
    # 保存每个说话人的top相似说话人
    top_similarities_file = os.path.join(similarities_output_dir, 'speaker_top_similarities.json')
    with open(top_similarities_file, 'w', encoding='utf-8') as f:
        json.dump(speaker_top_similarities, f, indent=2, ensure_ascii=False)
    
    # 保存相似度分布
    distribution_file = os.path.join(similarities_output_dir, 'similarity_distribution.json')
    with open(distribution_file, 'w', encoding='utf-8') as f:
        json.dump(similarity_distribution, f, indent=2, ensure_ascii=False)
    
    # 保存阈值统计
    threshold_stats_file = os.path.join(similarities_output_dir, 'threshold_statistics.json')
    with open(threshold_stats_file, 'w', encoding='utf-8') as f:
        json.dump(threshold_stats, f, indent=2, ensure_ascii=False)
    
    # 保存上三角矩阵数据
    upper_tri_file = os.path.join(similarities_output_dir, 'upper_triangular_similarities.npy')
    np.save(upper_tri_file, upper_tri_similarities)
    
    # 生成综合分析报告
    analysis_summary = {
        'analysis_info': {
            'total_speakers': n_speakers,
            'total_speaker_pairs': upper_tri_stats['total_pairs'],
            'top_k_analyzed': top_k,
            'extreme_pairs_analyzed': {
                'most_similar_count': len(most_similar_pairs),
                'least_similar_count': len(least_similar_pairs)
            }
        },
        'upper_triangular_stats': upper_tri_stats,
        'extreme_pairs_summary': {
            'most_similar_top1': most_similar_pairs[0] if most_similar_pairs else None,
            'least_similar_top1': least_similar_pairs[0] if least_similar_pairs else None,
            'total_extreme_pairs': len(most_similar_pairs) + len(least_similar_pairs)
        },
        'threshold_statistics': threshold_stats,
        'distribution_info': {
            'histogram_bins': len(hist_bins) - 1,
            'most_frequent_range': f"{hist_bins[np.argmax(hist_counts)]:.3f} - {hist_bins[np.argmax(hist_counts)+1]:.3f}"
        }
    }
    
    summary_file = os.path.join(similarities_output_dir, 'analysis_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    logger.info("Similarity matrix analysis completed successfully!")
    
    # 打印关键统计信息
    print(f"\n{'='*80}")
    print("SIMILARITY MATRIX ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total speakers: {n_speakers}")
    print(f"Total speaker pairs (upper triangular): {upper_tri_stats['total_pairs']:,}")
    print(f"Mean inter-speaker similarity: {upper_tri_stats['mean_similarity']:.4f}")
    print(f"Std inter-speaker similarity: {upper_tri_stats['std_similarity']:.4f}")
    print(f"Min inter-speaker similarity: {upper_tri_stats['min_similarity']:.4f}")
    print(f"Max inter-speaker similarity: {upper_tri_stats['max_similarity']:.4f}")
    print(f"Median inter-speaker similarity: {upper_tri_stats['median_similarity']:.4f}")
    print(f"\nTop 1000 most similar pairs analyzed: {len(most_similar_pairs)}")
    if most_similar_pairs:
        print(f"Most similar pair (Rank #1):")
        print(f"  {most_similar_pairs[0]['speaker1']} <-> {most_similar_pairs[0]['speaker2']}")
        print(f"  Similarity: {most_similar_pairs[0]['similarity']:.4f}")
    
    print(f"\nTop 1000 least similar pairs analyzed: {len(least_similar_pairs)}")
    if least_similar_pairs:
        print(f"Least similar pair (Rank #1):")
        print(f"  {least_similar_pairs[0]['speaker1']} <-> {least_similar_pairs[0]['speaker2']}")
        print(f"  Similarity: {least_similar_pairs[0]['similarity']:.4f}")
    print(f"\nHigh similarity pairs (>0.9): {threshold_stats['threshold_0.9']['count']:,} ({threshold_stats['threshold_0.9']['percentage']:.2f}%)")
    print(f"Very high similarity pairs (>0.95): {threshold_stats['threshold_0.95']['count']:,} ({threshold_stats['threshold_0.95']['percentage']:.2f}%)")
    print(f"Extremely high similarity pairs (>0.99): {threshold_stats['threshold_0.99']['count']:,} ({threshold_stats['threshold_0.99']['percentage']:.2f}%)")
    print(f"{'='*80}")
    
    return analysis_summary

# 主函数，串联各个处理阶段
# 包括扫描、计算embedding、保存、计算相似度、保存结果等
# 支持断点续跑和跳过相似度计算

def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging()
    
    start_time = time.time()
    logger.info("Starting FAST speaker similarity computation...")
    logger.info(f"Using {args.num_workers} workers, batch size {args.batch_size}")
    
    # 设置路径
    utterances_dir = os.path.join(args.embeddings_dir, args.utterances_subdir)
    speakers_output_dir = os.path.join(args.embeddings_dir, args.speakers_output_subdir)
    similarities_output_dir = os.path.join(args.embeddings_dir, args.similarities_output_subdir)
    progress_file = os.path.join(args.embeddings_dir, 'processing_progress.json')
    
    logger.info(f"Utterances directory: {utterances_dir}")
    logger.info(f"Speakers output directory: {speakers_output_dir}")
    logger.info(f"Similarities output directory: {similarities_output_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(utterances_dir):
        logger.error(f"Utterances directory not found: {utterances_dir}")
        return
    
    # 1. 快速扫描所有说话人的语音文件，构建说话人-utterance映射
    logger.info("=== Stage 1: Scanning speaker files ===")
    scan_start = time.time()
    speaker_utterances = scan_speaker_utterances_fast(utterances_dir, args.max_speakers)
    if not speaker_utterances:
        logger.error("No speaker utterances found!")
        return
    scan_time = time.time() - scan_start
    logger.info(f"Scanning completed in {scan_time:.2f} seconds")
    
    # 2. 并行计算每个说话人的平均embedding，支持断点续跑
    logger.info("=== Stage 2: Computing speaker embeddings ===")
    embed_start = time.time()
    speaker_embeddings, speaker_info = compute_speaker_embeddings_parallel(
        speaker_utterances, args.num_workers, args.batch_size, 
        progress_file if args.resume else None
    )
    if not speaker_embeddings:
        logger.error("No speaker embeddings computed!")
        return
    embed_time = time.time() - embed_start
    logger.info(f"Embedding computation completed in {embed_time:.2f} seconds")
    
    # 3. 保存每个说话人的embedding和信息到文件
    logger.info("=== Stage 3: Saving speaker files ===")
    save_start = time.time()
    save_speaker_files_fast(speaker_embeddings, speaker_info, speakers_output_dir)
    save_time = time.time() - save_start
    logger.info(f"Speaker files saved in {save_time:.2f} seconds")
    
    # 4. 计算说话人相似度（可选，支持跳过）
    if not args.skip_similarity:
        logger.info("=== Stage 4: Computing similarities ===")
        sim_start = time.time()
        similarities, similarity_matrix, speaker_keys = compute_speaker_similarities_fast(
            speaker_embeddings, args.num_workers
        )
        sim_time = time.time() - sim_start
        logger.info(f"Similarity computation completed in {sim_time:.2f} seconds")
        
        # 5. 保存相似度结果，包括分块保存和统计信息
        logger.info("=== Stage 5: Saving similarity results ===")
        sim_save_start = time.time()
        stats = save_similarity_results_fast(similarities, similarity_matrix, speaker_keys, similarities_output_dir)
        sim_save_time = time.time() - sim_save_start
        logger.info(f"Similarity results saved in {sim_save_time:.2f} seconds")
        
        # 6. 详细分析相似度矩阵
        if len(similarity_matrix) > 0:  # 确保有有效的相似度矩阵
            logger.info("=== Stage 6: Analyzing similarity matrix ===")
            analysis_start = time.time()
            analysis_summary = analyze_similarity_matrix(similarity_matrix, speaker_keys, similarities_output_dir, args.top_k)
            analysis_time = time.time() - analysis_start
            logger.info(f"Similarity matrix analysis completed in {analysis_time:.2f} seconds")
        else:
            logger.warning("No valid similarity matrix to analyze")
        
        # 打印简要统计信息，便于快速了解整体情况
        print(f"\n{'='*60}")
        print("SIMILARITY COMPUTATION COMPLETED")
        print(f"{'='*60}")
        print(f"Total speakers: {stats['total_speakers']}")
        print(f"Mean similarity: {stats['mean_similarity']:.4f}")
        print(f"Min similarity: {stats['min_similarity']:.4f}")
        print(f"Max similarity: {stats['max_similarity']:.4f}")
        print(f"{'='*60}")
    else:
        logger.info("Similarity computation skipped as requested")
    
    # 清理进度文件，避免下次误用
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    total_time = time.time() - start_time
    logger.info(f"TOTAL PROCESSING TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("Fast speaker similarity computation completed successfully!")

if __name__ == "__main__":
    main() 