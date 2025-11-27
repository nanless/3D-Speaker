# 3dspeaker说话人分析核心脚本

本项目提供两个核心脚本，用于基于3dspeaker模型的说话人分析：多GPU embedding提取和基于GMM的说话人边界检测。

## 项目结构

```
egs/split_sequential_speakers/
├── extract_embeddings_multigpu.py    # 多GPU embedding提取脚本
├── detect_boundaries_from_embeddings.py  # 基于GMM的边界检测脚本
└── README.md                         # 本文档
```

## 环境准备

确保已安装以下依赖：
```bash
# Python环境
python >= 3.7

# 必要的Python包
torch >= 1.8.0
torchaudio
numpy
scikit-learn
tqdm
matplotlib
seaborn
modelscope
pickle

# GPU环境
CUDA >= 10.2
```

## 使用说明

### 步骤1：多GPU Embedding提取

使用`extract_embeddings_multigpu.py`进行多卡并行embedding提取：

```bash
python extract_embeddings_multigpu.py \
    --input_dir /root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_files \
    --output_dir /root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_embeddings_eres2netv2w24s4ep4 \
    --model_path /root/workspace/speaker_verification/mix_adult_kid/exp/modelscope_eres2netv2/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common \
    --world_size 4 \
    --batch_size 16 \
    --convert_individual
```

**主要参数说明：**
- `--input_dir`: 输入音频文件目录
- `--output_dir`: 输出embedding目录
- `--model_path`: 3dspeaker模型路径
- `--world_size`: 使用的GPU数量（建议4卡）
- `--batch_size`: 批处理大小（根据GPU内存调整）
- `--convert_individual`: 转换为单独的embedding文件（便于后续处理）

**输出文件：**
```
output_dir/
├── all_embeddings.pkl          # 合并的embedding文件
├── individual/                 # 单个embedding文件目录
├── final_stats.json           # 提取统计信息
└── gpu_*_stats.json          # 各GPU处理统计
```

### 步骤2：基于GMM的说话人边界检测

使用`detect_boundaries_from_embeddings.py`进行说话人边界检测：

```bash
python detect_boundaries_from_embeddings.py \
    --embeddings_path /path/to/all_embeddings.pkl \
    --output_dir /path/to/boundary_results \
    --segment_size 1000 \
    --use_gmm \
    --gmm_components 2 \
    --boundary_window 10
```

**主要参数说明：**
- `--embeddings_path`: embedding文件路径（步骤1的输出文件）
- `--output_dir`: 边界检测结果输出目录
- `--segment_size`: 每段的预期大小（音频文件数量）
- `--use_gmm`: 使用GMM算法（推荐）
- `--gmm_components`: GMM组件数量（通常2个足够）
- `--boundary_window`: 边界搜索窗口大小

**输出文件：**
```
output_dir/
├── speaker_001/               # 说话人1的音频文件
├── speaker_002/               # 说话人2的音频文件
├── speaker_xxx/               # 其他说话人音频文件
├── speaker_boundary_detection_result.json  # 边界检测结果
└── boundary_detection_visualization.png    # 可视化图表
```

## 完整流程示例

以下是用户指定路径的完整示例：

```bash
# 步骤1：多GPU提取embedding
python extract_embeddings_multigpu.py \
    --input_dir /root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_files \
    --output_dir /root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_embeddings_eres2netv2w24s4ep4 \
    --model_path /root/workspace/speaker_verification/mix_adult_kid/exp/modelscope_eres2netv2/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common \
    --world_size 4 \
    --batch_size 16 \
    --convert_individual

# 步骤2：GMM边界检测
python detect_boundaries_from_embeddings.py \
    --embeddings_path /root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_embeddings_eres2netv2w24s4ep4/all_embeddings.pkl \
    --output_dir /root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_embeddings_eres2netv2w24s4ep4/boundary_detection \
    --segment_size 1000 \
    --use_gmm \
    --gmm_components 2 \
    --boundary_window 10
```

## 参数调优建议

### GPU配置优化
- **world_size**: 根据可用GPU数量设置（1-8）
- **batch_size**: 根据GPU内存调整
  - 24GB GPU: batch_size=32
  - 12GB GPU: batch_size=16  
  - 8GB GPU: batch_size=8

### 边界检测优化
- **segment_size**: 根据预期说话人数量调整
  - 较多说话人: 500-800
  - 中等说话人数: 1000-1500
  - 较少说话人: 2000+
  
- **gmm_components**: GMM聚类中心数量
  - 通常2个组件足够
  - 复杂场景可以尝试3-4个

## 性能特性

### 多GPU Embedding提取
- **并行处理**: 支持多GPU并行，显著提升处理速度
- **内存优化**: 分批保存，避免内存溢出
- **容错机制**: 单个文件失败不影响整体处理
- **进度追踪**: 实时显示处理进度和统计信息

### GMM边界检测
- **自适应算法**: 根据前一个边界重新计算后续边界位置
- **GMM建模**: 为每个说话人段训练高斯混合模型
- **边界优化**: 在理论边界附近搜索最优分割点
- **可视化输出**: 生成边界检测的可视化图表

## 故障排除

### 常见问题

**1. GPU内存不足**
```bash
# 解决方案：减少batch_size
--batch_size 8
```

**2. 模型加载失败**  
检查模型路径和文件是否完整：
```bash
ls -la /path/to/model/
```

**3. 边界检测结果不理想**
- 调整segment_size参数
- 尝试不同的gmm_components数量
- 增大boundary_window搜索范围

**4. 音频格式不支持**
确保音频文件为支持的格式：.wav, .flac, .mp3, .m4a

### 性能优化

**提升处理速度：**
- 增加GPU数量
- 使用SSD存储
- 增加num_workers线程数

**降低内存占用：**
- 减少batch_size
- 增加save_interval
- 启用convert_individual模式

## 技术原理

### Embedding提取
- 使用3dspeaker预训练模型提取说话人特征
- 支持多种音频格式和采样率
- 自动处理音频长度标准化

### GMM边界检测  
- 基于高斯混合模型对说话人特征建模
- 自适应边界搜索算法
- 结合余弦相似度进行边界验证

## 许可证

本项目基于Apache License 2.0许可证开源。