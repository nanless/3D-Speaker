#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
基于预提取embedding的说话人边界检测脚本
从embedding文件中读取数据进行边界检测和说话人分割
"""

import os
import sys
import json
import numpy as np
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import seaborn as sns
from collections import defaultdict
import warnings
import time
import shutil
warnings.filterwarnings('ignore')

def load_embeddings_from_merged_file(embeddings_file):
    """从合并的embedding文件加载数据"""
    print(f"📂 加载合并的embedding文件: {embeddings_file}")
    
    try:
        with open(embeddings_file, 'rb') as f:
            all_embeddings_data = pickle.load(f)
        
        print(f"✅ 成功加载 {len(all_embeddings_data)} 个embedding")
        
        # 按音频文件路径排序确保顺序
        all_embeddings_data.sort(key=lambda x: x['audio_file'])
        
        # 提取embedding矩阵和音频信息
        embeddings = []
        audio_info = []
        failed_count = 0
        
        for data in all_embeddings_data:
            try:
                embedding = data['embedding']
                if embedding is not None and len(embedding) > 0:
                    embeddings.append(embedding)
                    
                    # 确保relative_path信息正确
                    original_path = data.get('audio_file', data.get('original_path', ''))
                    relative_path = data.get('relative_path', '')
                    
                    # 如果没有relative_path，尝试从original_path推导
                    if not relative_path and original_path:
                        relative_path = os.path.basename(original_path)
                    
                    audio_info.append({
                        'original_path': original_path,
                        'relative_path': relative_path,
                        'filename': data.get('filename', os.path.basename(original_path)),
                        'duration': data.get('duration', 0.0),
                        'embedding_dim': data.get('embedding_dim', len(embedding))
                    })
                else:
                    failed_count += 1
            except Exception as e:
                print(f"⚠️ 处理embedding数据失败: {e}")
                failed_count += 1
        
        embeddings = np.array(embeddings) if embeddings else np.array([]).reshape(0, -1)
        
        print(f"📊 有效embedding: {len(embeddings)}, 失败: {failed_count}")
        if len(embeddings) > 0:
            print(f"🎯 Embedding维度: {embeddings.shape[1]}")
        
        return embeddings, audio_info, []
        
    except Exception as e:
        print(f"❌ 加载embedding文件失败: {e}")
        return np.array([]).reshape(0, -1), [], []

def load_embeddings_from_individual_files(embeddings_dir):
    """从单独的embedding文件目录加载数据"""
    print(f"📂 扫描单独的embedding文件: {embeddings_dir}")
    
    embedding_files = []
    
    for root, dirs, files in os.walk(embeddings_dir):
        for file in files:
            if file.endswith('.pkl'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, embeddings_dir)
                embedding_files.append({
                    'full_path': full_path,
                    'relative_path': rel_path,
                    'filename': file
                })
    
    # 按相对路径排序
    embedding_files.sort(key=lambda x: x['relative_path'])
    
    print(f"📊 找到 {len(embedding_files)} 个embedding文件")
    
    # 加载所有embedding
    print("🎯 加载所有embedding文件...")
    
    embeddings = []
    audio_info = []
    failed_files = []
    
    for file_info in tqdm(embedding_files, desc="加载embedding"):
        try:
            with open(file_info['full_path'], 'rb') as f:
                data = pickle.load(f)
            
            embedding = data['embedding']
            if embedding is not None and len(embedding) > 0:
                embeddings.append(embedding)
                
                # 获取音频文件的路径信息
                original_path = data.get('audio_file', data.get('original_path', ''))
                relative_path = data.get('relative_path', '')
                
                # 如果没有relative_path，尝试从embedding文件路径推导
                if not relative_path:
                    # 从embedding文件路径推导相对路径
                    emb_relative_path = file_info['relative_path']
                    # 移除.pkl扩展名，恢复音频文件相对路径
                    if emb_relative_path.endswith('.pkl'):
                        # 假设embedding文件的路径结构对应音频文件
                        relative_path = os.path.splitext(emb_relative_path)[0]
                        # 可能需要添加音频文件扩展名，但这里先保持无扩展名
                        # 后续可以根据需要调整
                    else:
                        relative_path = emb_relative_path
                
                audio_info.append({
                    'original_path': original_path,
                    'relative_path': relative_path,
                    'filename': data.get('filename', file_info['filename']),
                    'duration': data.get('duration', 0.0),
                    'embedding_file': file_info['full_path']
                })
            else:
                failed_files.append(file_info['full_path'])
                
        except Exception as e:
            print(f"⚠️ 加载embedding文件失败: {file_info['full_path']}, 错误: {e}")
            failed_files.append(file_info['full_path'])
    
    embeddings = np.array(embeddings) if embeddings else np.array([]).reshape(0, -1)
    
    print(f"✅ 成功加载 {len(embeddings)} 个embedding，失败 {len(failed_files)} 个")
    
    return embeddings, audio_info, failed_files

def calculate_segment_centers(embeddings: np.ndarray, segment_size: int = 1000) -> List[np.ndarray]:
    """计算每个段的聚类中心"""
    segment_centers = []
    n_segments = (len(embeddings) + segment_size - 1) // segment_size
    
    print(f"📊 计算 {n_segments} 个段的聚类中心...")
    
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(embeddings))
        
        segment_embeddings = embeddings[start_idx:end_idx]
        center = np.mean(segment_embeddings, axis=0)
        segment_centers.append(center)
        
        print(f"  段 {i+1}: [{start_idx}:{end_idx}] -> 中心计算完成")
    
    return segment_centers

def train_speaker_gmm(embeddings: np.ndarray, segment_name: str = "", n_components: int = 2, 
                     min_samples: int = 10) -> Optional[GaussianMixture]:
    """为说话人段训练高斯混合模型"""
    if len(embeddings) < min_samples:
        print(f"    ⚠️ {segment_name} 样本数量不足({len(embeddings)} < {min_samples})，跳过GMM训练")
        return None
    
    # 如果样本数少于组件数的5倍，减少组件数
    if len(embeddings) < n_components * 5:
        n_components = max(1, len(embeddings) // 5)
        print(f"    📉 {segment_name} 样本数量较少，调整GMM组件数为 {n_components}")
    
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            max_iter=100,
            n_init=3
        )
        
        gmm.fit(embeddings)
        
        if not gmm.converged_:
            print(f"    ⚠️ {segment_name} GMM未收敛，使用简单均值代替")
            return None
            
        bic = gmm.bic(embeddings)
        aic = gmm.aic(embeddings)
        
        print(f"    ✅ {segment_name} GMM训练成功: {n_components}组件, BIC={bic:.2f}, AIC={aic:.2f}")
        
        return gmm
        
    except Exception as e:
        print(f"    ❌ {segment_name} GMM训练失败: {str(e)}")
        return None

def calculate_gmm_probability(embeddings: np.ndarray, gmm: GaussianMixture) -> float:
    """计算embedding向量在GMM模型下的对数概率"""
    if gmm is None or len(embeddings) == 0:
        return -np.inf
    
    try:
        log_probs = gmm.score_samples(embeddings)
        return np.mean(log_probs)
    except Exception as e:
        print(f"    ⚠️ GMM概率计算失败: {str(e)}")
        return -np.inf

def calculate_boundary_gmm_scores(embeddings: np.ndarray, boundary_idx: int, 
                                left_gmm: GaussianMixture, right_gmm: GaussianMixture,
                                window_size: int = 5) -> Tuple[float, Dict]:
    """使用GMM模型计算边界点的分割质量"""
    # 获取边界附近的embedding
    start_idx = max(0, boundary_idx - window_size)
    end_idx = min(len(embeddings), boundary_idx + window_size)
    
    left_embeddings = embeddings[start_idx:boundary_idx]
    right_embeddings = embeddings[boundary_idx:end_idx]
    
    if len(left_embeddings) == 0 or len(right_embeddings) == 0:
        return -np.inf, {'error': 'Empty segments'}
    
    # 计算左段embedding在左GMM和右GMM中的概率
    left_in_left_prob = calculate_gmm_probability(left_embeddings, left_gmm)
    left_in_right_prob = calculate_gmm_probability(left_embeddings, right_gmm)
    
    # 计算右段embedding在左GMM和右GMM中的概率
    right_in_left_prob = calculate_gmm_probability(right_embeddings, left_gmm)
    right_in_right_prob = calculate_gmm_probability(right_embeddings, right_gmm)
    
    # 计算分割质量：正确分类的概率 - 错误分类的概率
    correct_assignment = left_in_left_prob + right_in_right_prob
    wrong_assignment = left_in_right_prob + right_in_left_prob
    
    separation_score = correct_assignment - wrong_assignment
    
    debug_info = {
        'left_in_left_prob': left_in_left_prob,
        'left_in_right_prob': left_in_right_prob,
        'right_in_left_prob': right_in_left_prob,
        'right_in_right_prob': right_in_right_prob,
        'correct_assignment': correct_assignment,
        'wrong_assignment': wrong_assignment,
        'separation_score': separation_score,
        'left_segment_size': len(left_embeddings),
        'right_segment_size': len(right_embeddings)
    }
    
    return separation_score, debug_info

def find_precise_boundary(embeddings: np.ndarray, theoretical_boundary: int, 
                         left_center: np.ndarray, right_center: np.ndarray,
                         boundary_window: int = 10, debug: bool = False) -> Tuple[int, Dict]:
    """在理论分界点附近找到精确的分界点"""
    start_idx = max(0, theoretical_boundary - boundary_window)
    end_idx = min(len(embeddings), theoretical_boundary + boundary_window + 1)
    
    if start_idx >= end_idx:
        return theoretical_boundary, {'validation': {'overall_accuracy': 0.0}}
    
    best_boundary = theoretical_boundary
    best_score = float('-inf')
    boundary_scores = []
    
    # 在窗口内搜索最佳边界
    for candidate_boundary in range(start_idx, end_idx):
        if candidate_boundary == 0 or candidate_boundary >= len(embeddings):
            continue
        
        # 计算左右两侧与各自中心的相似度
        left_embeddings = embeddings[:candidate_boundary]
        right_embeddings = embeddings[candidate_boundary:]
        
        if len(left_embeddings) == 0 or len(right_embeddings) == 0:
            continue
        
        # 计算相似度分数
        left_similarities = cosine_similarity(left_embeddings, [left_center]).flatten()
        right_similarities = cosine_similarity(right_embeddings, [right_center]).flatten()
        
        # 总分数 = 左侧相似度均值 + 右侧相似度均值
        score = np.mean(left_similarities) + np.mean(right_similarities)
        boundary_scores.append((candidate_boundary, score))
        
        if score > best_score:
            best_score = score
            best_boundary = candidate_boundary
    
    # 边界验证
    validation_info = {}
    if best_boundary > 0 and best_boundary < len(embeddings):
        left_embeddings = embeddings[:best_boundary]
        right_embeddings = embeddings[best_boundary:]
        
        if len(left_embeddings) > 0 and len(right_embeddings) > 0:
            left_sims = cosine_similarity(left_embeddings, [left_center]).flatten()
            right_sims = cosine_similarity(right_embeddings, [right_center]).flatten()
            
            # 计算分类准确率
            left_correct = np.sum(left_sims > 0.5)
            right_correct = np.sum(right_sims > 0.5)
            total_correct = left_correct + right_correct
            total_samples = len(left_embeddings) + len(right_embeddings)
            
            validation_info = {
                'overall_accuracy': total_correct / total_samples if total_samples > 0 else 0,
                'left_accuracy': left_correct / len(left_embeddings),
                'right_accuracy': right_correct / len(right_embeddings),
                'left_avg_similarity': float(np.mean(left_sims)),
                'right_avg_similarity': float(np.mean(right_sims)),
                'boundary_score': float(best_score)
            }
    
    debug_info = {
        'theoretical_boundary': theoretical_boundary,
        'search_window': [start_idx, end_idx],
        'boundary_scores': boundary_scores if debug else [],
        'validation': validation_info
    }
    
    return best_boundary, debug_info

def find_precise_boundary_gmm(embeddings: np.ndarray, theoretical_boundary: int, 
                             left_gmm: GaussianMixture, right_gmm: GaussianMixture,
                             boundary_window: int = 10, debug: bool = False) -> Tuple[int, Dict]:
    """使用GMM模型在理论分界点附近找到精确的分界点"""
    start_idx = max(0, theoretical_boundary - boundary_window)
    end_idx = min(len(embeddings), theoretical_boundary + boundary_window + 1)
    
    if start_idx >= end_idx:
        return theoretical_boundary, {'validation': {'overall_accuracy': 0.0}}
    
    best_boundary = theoretical_boundary
    best_score = float('-inf')
    best_debug_info = {}
    
    print(f"      🔍 GMM边界搜索范围: [{start_idx}:{end_idx}]")
    
    scores = []
    for boundary_idx in range(start_idx, end_idx):
        score, debug_info = calculate_boundary_gmm_scores(
            embeddings, boundary_idx, left_gmm, right_gmm, window_size=5
        )
        
        scores.append(score)
        
        if debug:
            print(f"        位置 {boundary_idx}: GMM分离度={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_boundary = boundary_idx
            best_debug_info = debug_info
    
    # 计算验证信息
    validation_info = {
        'overall_accuracy': min(1.0, max(0.0, (best_score + 10) / 20)),  # 归一化到0-1
        'separation_score': best_score,
        'search_range': [start_idx, end_idx],
        'scores': scores
    }
    
    best_debug_info.update({
        'theoretical_boundary': theoretical_boundary,
        'best_boundary': best_boundary,
        'best_score': best_score,
        'search_window': boundary_window,
        'method': 'GMM',
        'validation': validation_info
    })
    
    print(f"      ✅ GMM最佳边界: {best_boundary} (分离度: {best_score:.4f})")
    
    return best_boundary, best_debug_info

def detect_speaker_boundaries_gmm(embeddings: np.ndarray, audio_info: List[Dict], 
                                segment_size: int = 1000, boundary_window: int = 10, 
                                n_components: int = 2, debug: bool = False) -> Tuple[List[int], Dict]:
    """基于混合高斯模型(GMM)的说话人边界检测"""
    
    if len(embeddings) <= segment_size:
        print("⚠️ 音频文件数量少于segment_size，不进行边界检测")
        boundaries = [0, len(embeddings)]
        debug_info = {
            'total_segments': 1,
            'theoretical_boundaries': [],
            'boundary_debug_info': [],
            'gmm_algorithm': True
        }
        return boundaries, debug_info
    
    print(f"🎭 开始基于GMM的自适应边界检测，预计段数基于 segment_size={segment_size}...")
    print(f"🧠 每个说话人将训练 {n_components} 个聚类中心的GMM模型")
    print(f"📏 使用自适应算法：根据实际边界重新计算后续起点")
    
    boundaries = [0]  # 起始边界
    boundary_debug_info = []
    theoretical_boundaries = []
    gmm_models = []  # 存储每个段的GMM模型
    
    # 自适应边界检测：每次基于前一个实际边界计算下一个理论边界
    current_start = 0
    segment_index = 0
    
    while current_start + segment_size < len(embeddings):
        # 计算当前理论边界位置（基于前一个实际边界）
        theoretical_boundary = current_start + segment_size
        theoretical_boundaries.append(theoretical_boundary)
        
        # 确保不超出数组范围
        if theoretical_boundary >= len(embeddings):
            break
        
        print(f"  🎯 检测边界 {segment_index+1} (理论位置: {theoretical_boundary})")
        
        # 训练当前段的GMM模型（排除边界附近的音频）
        current_segment_end = min(theoretical_boundary, len(embeddings))
        # 左段排除接近边界的后面部分，但确保有足够的训练数据
        left_train_end = max(current_start + boundary_window, current_segment_end - boundary_window)
        if left_train_end <= current_start:
            left_train_end = current_segment_end  # 如果缓冲区太大，使用完整段
        left_embeddings = embeddings[current_start:left_train_end]
        
        print(f"    🏗️ 训练左段GMM模型 (范围: [{current_start}:{left_train_end}], {len(left_embeddings)} 个样本, 排除边界缓冲区)")
        left_gmm = train_speaker_gmm(
            left_embeddings, 
            segment_name=f"左段{segment_index+1}", 
            n_components=n_components
        )
        
        # 训练下一段的GMM模型（排除边界附近的音频）
        next_segment_start = theoretical_boundary
        next_segment_end = min(next_segment_start + segment_size, len(embeddings))
        # 右段排除接近边界的前面部分，但确保有足够的训练数据
        right_train_start = min(next_segment_start + boundary_window, next_segment_end - boundary_window)
        if right_train_start >= next_segment_end:
            right_train_start = next_segment_start  # 如果缓冲区太大，使用完整段
        right_embeddings = embeddings[right_train_start:next_segment_end]
        
        print(f"    🏗️ 训练右段GMM模型 (范围: [{right_train_start}:{next_segment_end}], {len(right_embeddings)} 个样本, 排除边界缓冲区)")
        right_gmm = train_speaker_gmm(
            right_embeddings, 
            segment_name=f"右段{segment_index+2}", 
            n_components=n_components
        )
        
        # 如果GMM训练失败，回退到余弦相似度方法
        if left_gmm is None or right_gmm is None:
            print(f"    ⚠️ GMM训练失败，回退到余弦相似度方法")
            left_center = np.mean(left_embeddings, axis=0)
            right_center = np.mean(right_embeddings, axis=0)
            
            precise_boundary, debug_info = find_precise_boundary(
                embeddings, theoretical_boundary, left_center, right_center, 
                boundary_window, debug
            )
            debug_info['fallback_to_cosine'] = True
        else:
            # 使用GMM模型进行精确边界检测
            print(f"    🔍 使用GMM模型搜索精确边界 (窗口: ±{boundary_window})")
            precise_boundary, debug_info = find_precise_boundary_gmm(
                embeddings, theoretical_boundary, left_gmm, right_gmm, 
                boundary_window, debug
            )
            debug_info['fallback_to_cosine'] = False
        
        boundaries.append(precise_boundary)
        debug_info['segment_index'] = segment_index + 1
        debug_info['current_start'] = current_start
        debug_info['theoretical_boundary'] = theoretical_boundary
        debug_info['actual_segment_size'] = precise_boundary - current_start
        boundary_debug_info.append(debug_info)
        
        # 存储GMM模型
        gmm_models.append({
            'segment_index': segment_index + 1,
            'left_gmm': left_gmm,
            'right_gmm': right_gmm,
            'boundaries': [current_start, precise_boundary]
        })
        
        # 输出详细信息
        actual_segment_size = precise_boundary - current_start
        offset = precise_boundary - theoretical_boundary
        
        if 'validation' in debug_info and 'overall_accuracy' in debug_info['validation']:
            accuracy = debug_info['validation']['overall_accuracy']
            print(f"    ✅ 精确边界: {precise_boundary}")
            print(f"    📊 实际段大小: {actual_segment_size} (目标: {segment_size}, 偏差: {actual_segment_size - segment_size})")
            print(f"    📏 边界偏移: {offset}, 准确率: {accuracy:.2%}")
        else:
            print(f"    ✅ 精确边界: {precise_boundary} (偏移: {offset})")
            print(f"    📊 实际段大小: {actual_segment_size} (目标: {segment_size}, 偏差: {actual_segment_size - segment_size})")
        
        # 更新下一轮的起点为当前检测到的实际边界
        current_start = precise_boundary
        segment_index += 1
        
        # 已移除段数限制，继续处理所有数据
    
    boundaries.append(len(embeddings))  # 结束边界
    
    # 统计信息
    actual_segments = len(boundaries) - 1
    print(f"\n📊 GMM边界检测统计：")
    print(f"   实际段数: {actual_segments}")
    print(f"   GMM组件数: {n_components}")
    print(f"   段大小统计:")
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        segment_size_actual = end_idx - start_idx
        print(f"     段 {i+1}: {segment_size_actual} 个文件 (范围: [{start_idx}:{end_idx}])")
    
    # 为JSON序列化准备GMM模型摘要信息（移除不可序列化的GMM对象）
    gmm_models_summary = []
    for gmm_info in gmm_models:
        summary = {
            'segment_index': gmm_info['segment_index'],
            'boundaries': gmm_info['boundaries'],
            'left_gmm_trained': gmm_info['left_gmm'] is not None,
            'right_gmm_trained': gmm_info['right_gmm'] is not None
        }
        gmm_models_summary.append(summary)
    
    combined_debug = {
        'total_segments': actual_segments,
        'theoretical_boundaries': theoretical_boundaries,
        'boundary_debug_info': boundary_debug_info,
        'gmm_algorithm': True,
        'gmm_components': n_components,
        'gmm_models_summary': gmm_models_summary,  # 使用摘要而不是完整对象
        'target_segment_size': segment_size,
        'actual_segment_sizes': [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
    }
    
    return boundaries, combined_debug

def detect_speaker_boundaries(embeddings: np.ndarray, audio_info: List[Dict], 
                            segment_size: int = 1000, boundary_window: int = 10, 
                            debug: bool = False) -> Tuple[List[int], Dict]:
    """检测说话人边界（基于累积实际边界的自适应算法）"""
    
    if len(embeddings) <= segment_size:
        print("⚠️ 音频文件数量少于segment_size，不进行边界检测")
        boundaries = [0, len(embeddings)]
        debug_info = {
            'total_segments': 1,
            'theoretical_boundaries': [],
            'boundary_debug_info': []
        }
        return boundaries, debug_info
    
    # 计算段中心（用于初始参考）
    segment_centers = calculate_segment_centers(embeddings, segment_size)
    n_segments = len(segment_centers)
    
    print(f"🔍 开始自适应边界检测，预计 {n_segments} 个段...")
    print(f"📏 使用自适应算法：根据实际边界重新计算后续起点")
    
    boundaries = [0]  # 起始边界
    boundary_debug_info = []
    theoretical_boundaries = []
    
    # 自适应边界检测：每次基于前一个实际边界计算下一个理论边界
    current_start = 0
    segment_index = 0
    
    while current_start + segment_size < len(embeddings):
        # 计算当前理论边界位置（基于前一个实际边界）
        theoretical_boundary = current_start + segment_size
        theoretical_boundaries.append(theoretical_boundary)
        
        # 确保不超出数组范围
        if theoretical_boundary >= len(embeddings):
            break
        
        # 重新计算当前段和下一段的中心（排除边界附近的音频）
        # 当前段：从上一个实际边界到理论边界，排除接近边界的后面部分
        current_segment_end = min(theoretical_boundary, len(embeddings))
        left_train_end = max(current_start + boundary_window, current_segment_end - boundary_window)
        if left_train_end <= current_start:
            left_train_end = current_segment_end  # 如果缓冲区太大，使用完整段
        left_embeddings = embeddings[current_start:left_train_end]
        left_center = np.mean(left_embeddings, axis=0)
        
        # 下一段：从理论边界到理论边界+segment_size，排除接近边界的前面部分
        next_segment_start = theoretical_boundary
        next_segment_end = min(next_segment_start + segment_size, len(embeddings))
        right_train_start = min(next_segment_start + boundary_window, next_segment_end - boundary_window)
        if right_train_start >= next_segment_end:
            right_train_start = next_segment_start  # 如果缓冲区太大，使用完整段
        right_embeddings = embeddings[right_train_start:next_segment_end]
        right_center = np.mean(right_embeddings, axis=0)
        
        print(f"  🎯 检测边界 {segment_index+1} (理论位置: {theoretical_boundary})")
        print(f"    左段训练范围: [{current_start}:{left_train_end}] ({len(left_embeddings)} 个文件, 排除边界缓冲区)")
        print(f"    右段训练范围: [{right_train_start}:{next_segment_end}] ({len(right_embeddings)} 个文件, 排除边界缓冲区)")
        
        # 在理论边界附近搜索精确边界
        precise_boundary, debug_info = find_precise_boundary(
            embeddings, theoretical_boundary, left_center, right_center, 
            boundary_window, debug
        )
        
        boundaries.append(precise_boundary)
        debug_info['segment_index'] = segment_index + 1
        debug_info['current_start'] = current_start
        debug_info['theoretical_boundary'] = theoretical_boundary
        debug_info['actual_segment_size'] = precise_boundary - current_start
        boundary_debug_info.append(debug_info)
        
        accuracy = debug_info['validation']['overall_accuracy']
        actual_segment_size = precise_boundary - current_start
        offset = precise_boundary - theoretical_boundary
        
        print(f"    ✅ 精确边界: {precise_boundary}")
        print(f"    📊 实际段大小: {actual_segment_size} (目标: {segment_size}, 偏差: {actual_segment_size - segment_size})")
        print(f"    📏 边界偏移: {offset}, 准确率: {accuracy:.2%}")
        
        # 更新下一轮的起点为当前检测到的实际边界
        current_start = precise_boundary
        segment_index += 1
        
        # 已移除段数限制，继续处理所有数据
    
    boundaries.append(len(embeddings))  # 结束边界
    
    # 统计信息
    actual_segments = len(boundaries) - 1
    print(f"\n📊 边界检测统计：")
    print(f"   预计段数: {n_segments}")
    print(f"   实际段数: {actual_segments}")
    print(f"   段大小统计:")
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        segment_size_actual = end_idx - start_idx
        print(f"     段 {i+1}: {segment_size_actual} 个文件 (范围: [{start_idx}:{end_idx}])")
    
    combined_debug = {
        'total_segments': actual_segments,
        'theoretical_boundaries': theoretical_boundaries,
        'boundary_debug_info': boundary_debug_info,
        'adaptive_algorithm': True,
        'target_segment_size': segment_size,
        'actual_segment_sizes': [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
    }
    
    return boundaries, combined_debug

def convert_to_json_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    try:
        # 首先检查基本的JSON可序列化类型
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # 处理不可序列化的对象
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__class__') and 'sklearn' in str(obj.__class__):
            # 处理sklearn对象（如GaussianMixture），返回描述信息
            return f"<sklearn.{obj.__class__.__name__} object>"
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            # 对于其他不可序列化的对象，返回字符串描述
            return f"<{obj.__class__.__name__} object>"

def save_results(boundaries: List[int], debug_info: Dict, audio_info: List[Dict], output_dir: str) -> Dict:
    """保存检测结果并分割音频文件"""
    print("💾 保存结果并分割音频文件...")
    
    speakers = []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        file_count = end_idx - start_idx
        
        speaker_info = {
            'speaker_id': f'speaker_{i+1:03d}',
            'start_index': start_idx,
            'end_index': end_idx,
            'file_count': file_count,
            'audio_files': []
        }
        
        # 创建说话人目录
        speaker_dir = os.path.join(output_dir, speaker_info['speaker_id'])
        os.makedirs(speaker_dir, exist_ok=True)
        
        # 复制或链接音频文件，保持原始目录结构
        for j in range(start_idx, min(end_idx, len(audio_info))):
            audio_data = audio_info[j]
            original_path = audio_data['original_path']
            relative_path = audio_data.get('relative_path', '')
            filename = audio_data.get('filename', '')
            
            if original_path and os.path.exists(original_path):
                # 使用相对路径保持目录结构，如果没有则使用文件名
                if relative_path:
                    # 保持完整的相对路径结构
                    dst_file = os.path.join(speaker_dir, relative_path)
                    stored_path = relative_path
                else:
                    # 如果没有相对路径，只使用文件名
                    original_filename = os.path.basename(original_path)
                    dst_file = os.path.join(speaker_dir, original_filename)
                    stored_path = original_filename
                
                # 确保目标目录存在
                dst_dir = os.path.dirname(dst_file)
                if dst_dir and dst_dir != speaker_dir:
                    os.makedirs(dst_dir, exist_ok=True)
                
                try:
                    # 复制音频文件
                    shutil.copy2(original_path, dst_file)
                    speaker_info['audio_files'].append(stored_path)
                except Exception as e:
                    print(f"⚠️ 复制文件失败: {original_path} -> {dst_file}, 错误: {e}")
            else:
                print(f"⚠️ 音频文件不存在或路径为空: {original_path}")
        
        speakers.append(speaker_info)
    
    result = {
        'boundaries': boundaries,
        'speakers': speakers,
        'debug_info': debug_info,
        'total_audio_files': len(audio_info),
        'detected_speakers': len(speakers)
    }
    
    # 保存JSON结果（转换为JSON可序列化格式）
    result_file = os.path.join(output_dir, 'speaker_boundary_detection_result.json')
    json_serializable_result = convert_to_json_serializable(result)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(json_serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"📊 结果已保存: {result_file}")
    print(f"   检测到 {len(speakers)} 个说话人")
    print(f"   边界位置: {boundaries}")
    
    # 显示每个说话人的文件数量
    for speaker in speakers:
        print(f"   {speaker['speaker_id']}: {speaker['file_count']} 个音频文件")
    
    return result

def plot_boundary_visualization(embeddings: np.ndarray, boundaries: List[int], 
                               debug_info: Dict, output_file: str):
    """生成边界检测可视化图"""
    try:
        if len(embeddings) == 0:
            print("⚠️ 没有有效的embedding数据，跳过可视化")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 计算PCA降维用于可视化
        if embeddings.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        # 绘制embeddings散点图
        plt.subplot(2, 1, 1)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=range(len(embeddings_2d)), cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, label='File Index')
        plt.title('Speaker Embeddings Visualization (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # 绘制边界线
        for i, boundary in enumerate(boundaries[1:-1], 1):
            if boundary < len(embeddings):
                point = embeddings_2d[boundary]
                plt.axvline(x=point[0], color='red', linestyle='--', alpha=0.7, 
                          label=f'Boundary {i}' if i == 1 else "")
        
        if len(boundaries) > 2:
            plt.legend()
        
        # 绘制相似度曲线
        plt.subplot(2, 1, 2)
        if len(embeddings) > 1:
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            plt.plot(similarities, alpha=0.7, linewidth=1)
            plt.title('Adjacent File Similarity')
            plt.xlabel('File Index')
            plt.ylabel('Cosine Similarity')
            
            # 标记边界位置
            for boundary in boundaries[1:-1]:
                if boundary < len(similarities):
                    plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
                    plt.text(boundary, max(similarities) * 0.9, f'B{boundary}', 
                           rotation=90, ha='right', va='top')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 可视化图已保存: {output_file}")
        
    except Exception as e:
        print(f"⚠️ 生成可视化图失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="基于embedding的说话人边界检测")
    parser.add_argument('--embeddings_path', type=str, default="/root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_embeddings_eres2netv2w24s4ep4",
                       help='embedding文件路径（可以是合并的pkl文件或目录）')
    parser.add_argument('--output_dir', type=str, default="/root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_embeddings_eres2netv2w24s4ep4_boundaries",
                       help='输出目录路径')
    parser.add_argument('--segment_size', type=int, default=1000,
                       help='每段的预期大小（默认1000）')
    parser.add_argument('--boundary_window', type=int, default=20,
                       help='边界搜索窗口大小（默认10）')
    parser.add_argument('--debug', action='store_true',
                       help='开启调试模式')
    parser.add_argument('--use_gmm', action='store_true', default=True,
                       help='使用混合高斯模型(GMM)进行边界检测')
    parser.add_argument('--gmm_components', type=int, default=1,
                       help='GMM模型的组件数量（默认1个聚类中心）')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.embeddings_path):
        print(f"❌ embedding路径不存在: {args.embeddings_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🎯 开始基于embedding的说话人边界检测")
    print(f"📁 Embedding路径: {args.embeddings_path}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📏 段大小: {args.segment_size}")
    print(f"🔧 边界窗口: ±{args.boundary_window}")
    print(f"🐛 调试模式: {args.debug}")
    if args.use_gmm:
        print(f"🎭 检测算法: 混合高斯模型 (GMM)")
        print(f"🧠 GMM组件数: {args.gmm_components}")
    else:
        print(f"🎯 检测算法: 余弦相似度")
    print("")
    
    start_time = time.time()
    
    # 加载embeddings
    if os.path.isfile(args.embeddings_path):
        # 单个合并文件
        embeddings, audio_info, failed_files = load_embeddings_from_merged_file(args.embeddings_path)
    else:
        # 目录中的单独文件
        embeddings, audio_info, failed_files = load_embeddings_from_individual_files(args.embeddings_path)
    
    if len(embeddings) == 0:
        print("❌ 没有有效的embedding数据")
        return
    
    print(f"✅ 成功加载 {len(embeddings)} 个embedding")
    print(f"   第一个文件: {audio_info[0]['filename'] if audio_info else 'N/A'}")
    print(f"   最后一个文件: {audio_info[-1]['filename'] if audio_info else 'N/A'}")
    print(f"   Embedding维度: {embeddings.shape[1]}")
    
    # 检测说话人边界
    print("\n🔍 开始说话人边界检测...")
    if args.use_gmm:
        boundaries, detection_debug = detect_speaker_boundaries_gmm(
            embeddings, audio_info, args.segment_size, args.boundary_window, 
            args.gmm_components, args.debug
        )
    else:
        boundaries, detection_debug = detect_speaker_boundaries(
            embeddings, audio_info, args.segment_size, args.boundary_window, args.debug
        )
    
    # 合并调试信息
    combined_debug = {
        'detection': detection_debug,
        'processing_time': time.time() - start_time,
        'failed_files': failed_files,
        'total_embeddings': len(embeddings)
    }
    
    # 保存结果
    result = save_results(boundaries, combined_debug, audio_info, args.output_dir)
    
    # 生成可视化
    visualization_file = os.path.join(args.output_dir, 'boundary_detection_visualization.png')
    plot_boundary_visualization(embeddings, boundaries, combined_debug, visualization_file)
    
    total_time = time.time() - start_time
    print(f"\n🎉 边界检测完成！")
    print(f"⏱️ 总耗时: {total_time:.2f}秒")
    print(f"🎯 检测到 {len(boundaries) - 1} 个说话人")
    print(f"📊 处理效率: {len(embeddings)/total_time:.1f} 个文件/秒")
    print(f"💾 结果保存到: {args.output_dir}")

if __name__ == "__main__":
    main()