#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
提取最相似和最不相似说话人pairs的音频样本
用于人工验证和分析相似度计算结果
"""

import os
import sys
import json
import pickle
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Extract audio samples for similarity pairs')
    parser.add_argument('--embeddings_dir', type=str, 
                        default='/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings',
                        help='Base embeddings directory')
    parser.add_argument('--audio_data_dir', type=str,
                        default='/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments',
                        help='Directory containing original audio files')
    parser.add_argument('--similarities_subdir', type=str, default='speaker_similarity_analysis',
                        help='Subdirectory containing similarity results')
    parser.add_argument('--utterances_subdir', type=str, default='embeddings_individual/utterances',
                        help='Subdirectory containing utterance embeddings')
    parser.add_argument('--output_dir', type=str, default='similarity_pairs_audio_samples',
                        help='Output directory for audio samples')
    parser.add_argument('--num_samples_per_speaker', type=int, default=2,
                        help='Number of audio samples per speaker')
    parser.add_argument('--top_pairs', type=int, default=1000,
                        help='Number of top pairs to extract (both most and least similar)')
    parser.add_argument('--audio_extensions', nargs='+', default=['.wav', '.flac', '.mp3'],
                        help='Audio file extensions to look for')
    
    return parser.parse_args()

def load_similarity_pairs(similarities_dir):
    """加载相似度pairs结果"""
    logger = logging.getLogger(__name__)
    
    extreme_pairs_file = os.path.join(similarities_dir, 'extreme_similarity_pairs.json')
    if not os.path.exists(extreme_pairs_file):
        logger.error(f"Extreme similarity pairs file not found: {extreme_pairs_file}")
        return None, None
    
    try:
        with open(extreme_pairs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        most_similar_pairs = data['most_similar_pairs']
        least_similar_pairs = data['least_similar_pairs']
        
        logger.info(f"Loaded {len(most_similar_pairs)} most similar pairs")
        logger.info(f"Loaded {len(least_similar_pairs)} least similar pairs")
        
        return most_similar_pairs, least_similar_pairs
    
    except Exception as e:
        logger.error(f"Error loading similarity pairs: {e}")
        return None, None

def find_speaker_utterances(speaker_key, utterances_dir):
    """查找说话人的所有utterance文件"""
    logger = logging.getLogger(__name__)
    
    utterances = []
    
    # 遍历所有数据集和说话人目录
    for dataset_dir in Path(utterances_dir).iterdir():
        if not dataset_dir.is_dir():
            continue
            
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            # 检查是否匹配说话人
            if speaker_dir.name == speaker_key:
                # 找到所有pkl文件
                for pkl_file in speaker_dir.glob('*.pkl'):
                    try:
                        with open(pkl_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        # 检查数据结构，支持两种格式
                        if 'info' in data and isinstance(data['info'], dict):
                            # 新格式：info嵌套在'info'字段中
                            info = data['info']
                            utterances.append({
                                'utterance_id': info.get('utterance_id', pkl_file.stem),
                                'original_path': info.get('path', ''),  # 注意：这里是'path'而不是'original_path'
                                'dataset': info.get('dataset', dataset_dir.name),
                                'speaker_id': info.get('speaker_id', speaker_dir.name),
                                'pkl_path': str(pkl_file)
                            })
                        else:
                            # 旧格式：info直接在顶层
                            utterances.append({
                                'utterance_id': data.get('utterance_id', pkl_file.stem),
                                'original_path': data.get('original_path', ''),
                                'dataset': data.get('dataset', dataset_dir.name),
                                'speaker_id': data.get('speaker_id', speaker_dir.name),
                                'pkl_path': str(pkl_file)
                            })
                    except Exception as e:
                        logger.warning(f"Error loading {pkl_file}: {e}")
                        continue
                
                # 如果找到了utterances，返回结果
                if utterances:
                    logger.debug(f"Found {len(utterances)} utterances for speaker {speaker_key}")
                    return utterances
    
    logger.warning(f"No utterances found for speaker: {speaker_key}")
    return utterances

def find_audio_file(original_path, audio_data_dir, audio_extensions):
    """查找音频文件的实际路径"""
    logger = logging.getLogger(__name__)
    
    # 如果original_path是绝对路径且存在，直接返回
    if os.path.isabs(original_path) and os.path.exists(original_path):
        return original_path
    
    # 从original_path提取相对路径信息
    if original_path:
        # 尝试不同的路径解析方式
        path_parts = Path(original_path).parts
        
        # 查找包含数据集名称的路径段
        for i, part in enumerate(path_parts):
            if i < len(path_parts) - 2:  # 至少需要dataset/speaker/file三级
                candidate_path = Path(audio_data_dir) / Path(*path_parts[i:])
                if candidate_path.exists():
                    return str(candidate_path)
        
        # 尝试直接拼接最后几段路径
        for num_parts in [3, 4, 5]:  # 尝试最后3到5段路径
            if len(path_parts) >= num_parts:
                candidate_path = Path(audio_data_dir) / Path(*path_parts[-num_parts:])
                if candidate_path.exists():
                    return str(candidate_path)
    
    logger.warning(f"Audio file not found for path: {original_path}")
    return None

def create_pair_directory(pair_info, pair_type, output_dir):
    """为pair创建目录"""
    speaker1 = pair_info['speaker1']
    speaker2 = pair_info['speaker2']
    similarity = pair_info['similarity']
    rank = pair_info['rank']
    
    # 创建目录名，包含相似度和排名信息
    dir_name = f"{pair_type}_rank{rank:03d}_{speaker1}_vs_{speaker2}_sim{similarity:.4f}"
    
    # 替换文件名中的特殊字符
    dir_name = dir_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    pair_dir = os.path.join(output_dir, dir_name)
    os.makedirs(pair_dir, exist_ok=True)
    
    return pair_dir

def copy_audio_samples(utterances, pair_dir, speaker_key, num_samples):
    """复制音频样本到目标目录"""
    logger = logging.getLogger(__name__)
    
    if not utterances:
        logger.warning(f"No utterances found for speaker {speaker_key}")
        return []
    
    # 选择前num_samples个utterance
    selected_utterances = utterances[:num_samples]
    copied_files = []
    
    for i, utt in enumerate(selected_utterances):
        original_path = utt.get('original_path', '')
        utterance_id = utt.get('utterance_id', f'utt_{i}')
        
        if not original_path or not os.path.exists(original_path):
            logger.warning(f"Audio file not found: {original_path}")
            continue
        
        # 创建目标文件名
        file_ext = Path(original_path).suffix
        target_filename = f"{speaker_key}_{utterance_id}_{i+1:02d}{file_ext}"
        target_path = os.path.join(pair_dir, target_filename)
        
        try:
            shutil.copy2(original_path, target_path)
            copied_files.append(target_path)
            logger.debug(f"Copied {original_path} -> {target_path}")
        except Exception as e:
            logger.error(f"Error copying {original_path}: {e}")
            continue
    
    return copied_files

def create_pair_info_file(pair_info, pair_dir, speaker1_files, speaker2_files):
    """创建pair信息文件"""
    info = {
        'pair_info': pair_info,
        'speaker1': {
            'speaker_key': pair_info['speaker1'],
            'audio_files': [os.path.basename(f) for f in speaker1_files]
        },
        'speaker2': {
            'speaker_key': pair_info['speaker2'],
            'audio_files': [os.path.basename(f) for f in speaker2_files]
        },
        'total_files': len(speaker1_files) + len(speaker2_files)
    }
    
    info_file = os.path.join(pair_dir, 'pair_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

def process_similarity_pairs(pairs, pair_type, utterances_dir, audio_data_dir, output_dir, 
                           num_samples_per_speaker, top_pairs, audio_extensions):
    """处理相似度pairs"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing {pair_type} pairs...")
    
    # 限制处理的pairs数量
    pairs_to_process = pairs[:top_pairs]
    
    success_count = 0
    error_count = 0
    
    for pair_info in tqdm(pairs_to_process, desc=f"Processing {pair_type} pairs"):
        try:
            # 创建pair目录
            pair_dir = create_pair_directory(pair_info, pair_type, output_dir)
            
            # 查找两个说话人的utterances
            speaker1_utterances = find_speaker_utterances(pair_info['speaker1'], utterances_dir)
            speaker2_utterances = find_speaker_utterances(pair_info['speaker2'], utterances_dir)
            
            if not speaker1_utterances or not speaker2_utterances:
                logger.warning(f"Missing utterances for pair: {pair_info['speaker1']} vs {pair_info['speaker2']}")
                error_count += 1
                continue
            
            # 验证并找到实际的音频文件路径
            for utterances in [speaker1_utterances, speaker2_utterances]:
                for utt in utterances:
                    if utt['original_path']:
                        actual_path = find_audio_file(utt['original_path'], audio_data_dir, audio_extensions)
                        if actual_path:
                            utt['original_path'] = actual_path
            
            # 复制音频样本
            speaker1_files = copy_audio_samples(speaker1_utterances, pair_dir, 
                                              pair_info['speaker1'], num_samples_per_speaker)
            speaker2_files = copy_audio_samples(speaker2_utterances, pair_dir, 
                                              pair_info['speaker2'], num_samples_per_speaker)
            
            if speaker1_files or speaker2_files:
                # 创建pair信息文件
                create_pair_info_file(pair_info, pair_dir, speaker1_files, speaker2_files)
                success_count += 1
            else:
                logger.warning(f"No audio files copied for pair: {pair_info['speaker1']} vs {pair_info['speaker2']}")
                error_count += 1
        
        except Exception as e:
            logger.error(f"Error processing pair {pair_info['speaker1']} vs {pair_info['speaker2']}: {e}")
            error_count += 1
            continue
    
    logger.info(f"{pair_type} pairs processed: {success_count} success, {error_count} errors")
    return success_count, error_count

def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting similarity pairs audio extraction...")
    
    # 设置路径
    similarities_dir = os.path.join(args.embeddings_dir, args.similarities_subdir)
    utterances_dir = os.path.join(args.embeddings_dir, args.utterances_subdir)
    output_dir = os.path.join(args.embeddings_dir, args.output_dir)
    
    logger.info(f"Similarities directory: {similarities_dir}")
    logger.info(f"Utterances directory: {utterances_dir}")
    logger.info(f"Audio data directory: {args.audio_data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # 检查必要目录
    for dir_path, dir_name in [(similarities_dir, "similarities"), 
                               (utterances_dir, "utterances"), 
                               (args.audio_data_dir, "audio data")]:
        if not os.path.exists(dir_path):
            logger.error(f"{dir_name} directory not found: {dir_path}")
            return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载相似度pairs
    logger.info("=== Loading similarity pairs ===")
    most_similar_pairs, least_similar_pairs = load_similarity_pairs(similarities_dir)
    
    if most_similar_pairs is None or least_similar_pairs is None:
        logger.error("Failed to load similarity pairs")
        return
    
    # 创建分类目录
    most_similar_dir = os.path.join(output_dir, "most_similar_pairs")
    least_similar_dir = os.path.join(output_dir, "least_similar_pairs")
    os.makedirs(most_similar_dir, exist_ok=True)
    os.makedirs(least_similar_dir, exist_ok=True)
    
    # 处理最相似的pairs
    logger.info("=== Processing most similar pairs ===")
    most_success, most_errors = process_similarity_pairs(
        most_similar_pairs, "most_similar", utterances_dir, args.audio_data_dir,
        most_similar_dir, args.num_samples_per_speaker, args.top_pairs, args.audio_extensions
    )
    
    # 处理最不相似的pairs
    logger.info("=== Processing least similar pairs ===")
    least_success, least_errors = process_similarity_pairs(
        least_similar_pairs, "least_similar", utterances_dir, args.audio_data_dir,
        least_similar_dir, args.num_samples_per_speaker, args.top_pairs, args.audio_extensions
    )
    
    # 生成总结报告
    summary = {
        'extraction_summary': {
            'total_pairs_processed': args.top_pairs * 2,
            'most_similar_pairs': {
                'success': most_success,
                'errors': most_errors,
                'success_rate': f"{most_success/(most_success+most_errors)*100:.1f}%" if (most_success+most_errors) > 0 else "0%"
            },
            'least_similar_pairs': {
                'success': least_success,
                'errors': least_errors,
                'success_rate': f"{least_success/(least_success+least_errors)*100:.1f}%" if (least_success+least_errors) > 0 else "0%"
            },
            'total_success': most_success + least_success,
            'total_errors': most_errors + least_errors
        },
        'parameters': {
            'num_samples_per_speaker': args.num_samples_per_speaker,
            'top_pairs': args.top_pairs,
            'audio_extensions': args.audio_extensions
        },
        'directories': {
            'similarities_dir': similarities_dir,
            'utterances_dir': utterances_dir,
            'audio_data_dir': args.audio_data_dir,
            'output_dir': output_dir
        }
    }
    
    summary_file = os.path.join(output_dir, 'extraction_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print(f"\n{'='*80}")
    print("SIMILARITY PAIRS AUDIO EXTRACTION COMPLETED")
    print(f"{'='*80}")
    print(f"Total pairs processed: {args.top_pairs * 2}")
    print(f"Most similar pairs: {most_success} success, {most_errors} errors")
    print(f"Least similar pairs: {least_success} success, {least_errors} errors")
    print(f"Total success: {most_success + least_success}")
    print(f"Total errors: {most_errors + least_errors}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")
    
    logger.info("Similarity pairs audio extraction completed successfully!")

if __name__ == "__main__":
    main() 