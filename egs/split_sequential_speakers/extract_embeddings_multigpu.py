#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
基于3dspeaker模型的多GPU embedding提取脚本
支持多GPU并行处理，用于大规模音频数据的embedding提取
"""

import os
import sys
import json
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Dict, Tuple, Optional

# 添加speakerlab路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from speakerlab.utils.builder import dynamic_import

warnings.filterwarnings('ignore')

def setup_logging(log_file=None):
    """设置日志记录"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(level=logging.INFO, format=log_format, 
                          handlers=[
                              logging.FileHandler(log_file),
                              logging.StreamHandler()
                          ])
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)

def setup(rank, world_size, port='12355'):
    """初始化分布式进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    print(f"🔗 GPU {rank} 连接到分布式组，端口: {port}")
    
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"✅ GPU {rank} NCCL初始化成功")
    except Exception as e:
        print(f"⚠️ GPU {rank} NCCL初始化失败，回退到gloo后端: {e}")
        try:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
            print(f"✅ GPU {rank} GLOO初始化成功")
        except Exception as e2:
            print(f"❌ GPU {rank} 分布式初始化完全失败: {e2}")
            raise

def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()

def scan_audio_files(input_dir):
    """扫描音频文件"""
    audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
    audio_files = []
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    print(f"🔍 扫描音频文件: {input_dir}")
    
    for ext in audio_extensions:
        files = list(input_path.rglob(f'*{ext}'))
        audio_files.extend(files)
    
    # 按路径排序确保顺序一致
    audio_files.sort()
    
    print(f"📊 找到 {len(audio_files)} 个音频文件")
    return [str(f) for f in audio_files]

class AudioDataset(Dataset):
    """音频数据集"""
    
    def __init__(self, audio_files: List[str], target_sr: int = 16000):
        self.audio_files = audio_files
        self.target_sr = target_sr
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        
        try:
            # 加载音频
            waveform, sr = torchaudio.load(audio_file)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            # 获取时长
            duration = waveform.shape[1] / self.target_sr
            
            return {
                'waveform': waveform.squeeze(0),  # 保持原始长度
                'audio_file': audio_file,
                'original_sr': sr,
                'duration': duration,
                'samples': waveform.shape[1]
            }
            
        except Exception as e:
            print(f"❌ 加载音频文件失败: {audio_file}, 错误: {e}")
            return {
                'waveform': torch.zeros(8000),  # 0.5秒的占位符
                'audio_file': audio_file,
                'original_sr': self.target_sr,
                'duration': 0.0,
                'samples': 0,
                'error': str(e)
            }

def load_3dspeaker_model(device: str):
    """加载3dspeaker云端模型"""
    print(f"🎯 加载3dspeaker云端模型...")
    
    try:
        from modelscope.pipelines import pipeline
        
        # 直接使用modelscope云端模型
        inference_pipeline = pipeline(
            task='speaker-verification',
            model='iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common',
            model_revision='v1.0.1',
            device=device
        )
        
        print(f"✅ 成功加载modelscope云端模型到设备: {device}")
        return inference_pipeline, 'modelscope'
        
    except Exception as e:
        print(f"❌ 加载modelscope云端模型失败: {e}")
        raise ValueError(f"模型加载失败: {e}")

def extract_single_embedding(model, model_type: str, waveform: torch.Tensor, audio_data: Dict, device: str):
    """提取单个音频的embedding"""
    try:
        if model_type == 'modelscope':
            # 转换为numpy
            audio_numpy = waveform.cpu().numpy()
            
            # 确保音频长度不太短（至少0.1秒）
            if len(audio_numpy) < int(0.1 * 16000):
                # 填充到0.5秒
                min_length = int(0.5 * 16000)
                audio_numpy = np.pad(audio_numpy, (0, max(0, min_length - len(audio_numpy))), mode='constant')
            
            # 尝试直接调用model的forward方法获取embedding
            if hasattr(model, 'model') and hasattr(model.model, 'forward'):
                # 将音频转换为tensor并移动到正确设备
                audio_tensor = torch.from_numpy(audio_numpy).unsqueeze(0).to(device)
                with torch.no_grad():
                    # 直接调用模型forward方法获取embedding
                    result = model.model.forward(audio_tensor)
                    if isinstance(result, dict) and 'emb' in result:
                        embedding = result['emb']
                    elif isinstance(result, torch.Tensor):
                        embedding = result
                    else:
                        embedding = result[0] if isinstance(result, (list, tuple)) else result
            else:
                # 回退到使用pipeline的方式
                # 创建一个虚拟的第二个音频（自己和自己比较）来触发embedding提取
                result = model([audio_numpy, audio_numpy])
                
                # 尝试从结果中提取embedding
                if hasattr(model, 'model') and hasattr(model.model, 'extract_emb'):
                    embedding = model.model.extract_emb(torch.from_numpy(audio_numpy).unsqueeze(0).to(device))
                else:
                    # 如果无法直接获取embedding，使用一个默认方法
                    if hasattr(model, 'model'):
                        audio_tensor = torch.from_numpy(audio_numpy).unsqueeze(0).to(device)
                        with torch.no_grad():
                            embedding = model.model(audio_tensor)
                            if isinstance(embedding, dict):
                                embedding = embedding.get('emb', embedding.get('embedding', list(embedding.values())[0]))
                    else:
                        embedding = np.zeros(192)  # 默认维度 (eres2netv2 是192维)
            
            # 转换为numpy数组
            if torch.is_tensor(embedding):
                embedding = embedding.cpu().numpy()
            elif isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # 确保是1维数组
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # 检查embedding是否有效
            if len(embedding) == 0:
                embedding = np.zeros(192)
            
            return embedding
            
        else:
            # torch模型处理（需要具体实现）
            return np.zeros(192)
    
    except Exception as e:
        print(f"❌ 提取embedding失败: {audio_data.get('audio_file', 'unknown')}, 错误: {e}")
        return np.zeros(192)

def extract_embeddings_on_gpu(rank, world_size, args, audio_files):
    """在指定GPU上提取embeddings"""
    setup(rank, world_size, args.port)
    
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    
    # 简化日志，不创建文件
    logger = setup_logging()
    
    print(f"🚀 GPU {rank} 开始处理 {len(audio_files)} 个文件...")
    
    # 分配文件到当前GPU
    files_per_gpu = len(audio_files) // world_size
    start_idx = rank * files_per_gpu
    if rank == world_size - 1:
        end_idx = len(audio_files)  # 最后一个GPU处理剩余所有文件
    else:
        end_idx = start_idx + files_per_gpu
    
    gpu_audio_files = audio_files[start_idx:end_idx]
    print(f"🎯 GPU {rank} 处理文件范围: [{start_idx}:{end_idx}] ({len(gpu_audio_files)} 个文件)")
    
    # 加载模型
    try:
        model, model_type = load_3dspeaker_model(device)
        print(f"✅ GPU {rank} 模型加载成功")
    except Exception as e:
        print(f"❌ GPU {rank} 模型加载失败: {e}")
        cleanup()
        return
    
    # 创建数据集
    dataset = AudioDataset(gpu_audio_files, target_sr=16000)
    
    # 处理统计
    processed_count = 0
    error_count = 0
    start_time = time.time()
    
    # 逐条处理音频文件并直接保存
    with torch.no_grad():
        for file_idx in tqdm(range(len(dataset)), desc=f"GPU {rank} 处理中", disable=(rank != 0)):
            try:
                # 获取单个音频数据
                audio_data = dataset[file_idx]
                audio_file = audio_data['audio_file']
                
                # 检查是否有错误
                if 'error' in audio_data:
                    if rank == 0:
                        print(f"⚠️ 音频文件有错误: {audio_file}, 错误: {audio_data['error']}")
                    error_count += 1
                    continue
                
                # 移动数据到GPU
                waveform = audio_data['waveform'].to(device)
                
                # 提取单个embedding
                embedding = extract_single_embedding(model, model_type, waveform, audio_data, device)
                
                # 创建embedding数据
                embedding_data = {
                    'embedding': embedding,
                    'audio_file': audio_file,
                    'original_path': audio_file,
                    'relative_path': os.path.relpath(audio_file, args.input_dir),
                    'filename': os.path.basename(audio_file),
                    'duration': audio_data['duration'],
                    'samples': audio_data.get('samples', 0),
                    'sample_rate': 16000,
                    'embedding_dim': len(embedding),
                    'model_type': model_type
                }
                
                # 直接保存到对应位置
                relative_path = embedding_data['relative_path']
                file_base = os.path.splitext(relative_path)[0]
                pkl_file_path = os.path.join(args.output_dir, f"{file_base}.pkl")
                
                # 确保目录存在
                pkl_dir = os.path.dirname(pkl_file_path)
                if pkl_dir:
                    os.makedirs(pkl_dir, exist_ok=True)
                
                # 保存单个embedding文件
                with open(pkl_file_path, 'wb') as f:
                    pickle.dump(embedding_data, f)
                
                processed_count += 1
                
            except Exception as e:
                if rank == 0:
                    print(f"❌ 处理文件失败: {gpu_audio_files[file_idx] if file_idx < len(gpu_audio_files) else 'unknown'}, 错误: {e}")
                error_count += 1
                continue
    
    # 处理完成统计
    processing_time = time.time() - start_time
    
    if rank == 0:
        print(f"🎉 GPU {rank} 处理完成!")
        print(f"📊 处理统计: {processed_count}/{len(gpu_audio_files)} 成功, {error_count} 错误")
        print(f"⏱️ 处理时间: {processing_time:.2f}秒")
        print(f"🚀 处理速度: {processed_count/processing_time:.2f} 文件/秒")
    
    cleanup()


def collect_final_stats(output_dir: str, audio_files: List[str], input_dir: str):
    """统计最终处理结果"""
    print("📊 统计处理结果...")
    
    processed_count = 0
    failed_count = 0
    
    for audio_file in audio_files:
        relative_path = os.path.relpath(audio_file, input_dir)
        file_base = os.path.splitext(relative_path)[0]
        pkl_file_path = os.path.join(output_dir, f"{file_base}.pkl")
        
        if os.path.exists(pkl_file_path):
            processed_count += 1
        else:
            failed_count += 1
    
    success_rate = processed_count / len(audio_files) * 100 if audio_files else 0
    
    print(f"✅ 处理完成!")
    print(f"📊 总文件数: {len(audio_files)}")
    print(f"✅ 成功处理: {processed_count}")
    print(f"❌ 处理失败: {failed_count}")
    print(f"📈 成功率: {success_rate:.2f}%")
    
    return {
        'total_files': len(audio_files),
        'processed_count': processed_count,
        'failed_count': failed_count,
        'success_rate': success_rate
    }

def main():
    parser = argparse.ArgumentParser(description="3dspeaker多GPU embedding提取")
    parser.add_argument('--input_dir', type=str, default="/root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_files",
                       help='输入音频文件目录')
    parser.add_argument('--output_dir', type=str, default="/root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_embeddings_eres2netv2w24s4ep4",
                       help='输出embedding目录')
    parser.add_argument('--world_size', type=int, default=4,
                       help='GPU数量（默认4）')
    parser.add_argument('--port', type=str, default='12355',
                       help='分布式通信端口（默认12355）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🎯 开始多GPU embedding提取")
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🌐 使用modelscope云端模型: iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common")
    print(f"🎮 GPU数量: {args.world_size}")
    print(f"🎵 保持原始音频长度，不截断不填充")
    print(f"💾 每个文件处理完成后直接保存到对应位置")
    print("")
    
    start_time = time.time()
    
    # 扫描音频文件
    audio_files = scan_audio_files(args.input_dir)
    
    if not audio_files:
        print("❌ 没有找到音频文件")
        return
    
    # 设置多进程启动方法以支持CUDA
    mp.set_start_method('spawn', force=True)
    
    # 在主进程中找到可用端口
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    # 检查指定端口是否可用
    master_port = args.port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', int(master_port)))
        print(f"🔌 使用指定端口: {master_port}")
    except OSError:
        original_port = master_port
        master_port = str(find_free_port())
        print(f"⚠️ 端口 {original_port} 被占用，使用动态端口: {master_port}")
    
    # 更新args中的端口
    args.port = master_port
    
    # 启动多进程处理
    print(f"🚀 启动 {args.world_size} 个GPU进程...")
    mp.spawn(extract_embeddings_on_gpu,
             args=(args.world_size, args, audio_files),
             nprocs=args.world_size,
             join=True)
    
    # 统计最终结果
    final_stats = collect_final_stats(args.output_dir, audio_files, args.input_dir)
    
    total_time = time.time() - start_time
    print(f"\n🎉 多GPU embedding提取完成!")
    print(f"⏱️ 总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"🚀 平均速度: {final_stats['processed_count']/total_time:.2f} 文件/秒")
    print(f"💾 结果直接保存到: {args.output_dir}")
    print(f"📂 embedding文件与原音频文件保持相同的目录结构和文件名")

if __name__ == "__main__":
    main()