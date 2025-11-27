# Diarization 参数调优建议

## 当前性能分析

基于评估结果 (`3dspeaker_is_multispeaker_accuracy_summary.json`):

- **准确率 (Accuracy)**: 75.4%
- **精确率 (Precision)**: 30.2% (较低)
- **召回率 (Recall)**: 50.0% (中等)
- **F1分数**: 37.7% (较低)
- **假阳性率 (False Alarm Rate)**: 20.1%
- **漏检率 (Miss Rate)**: 50.0% (较高)

### 混淆矩阵
- **TP (真阳性)**: 13 - 正确识别为多说话人
- **TN (真阴性)**: 119 - 正确识别为单说话人
- **FP (假阳性)**: 30 - 单说话人被误判为多说话人
- **FN (假阴性)**: 13 - 多说话人被误判为单说话人

### 主要问题

1. **漏检率高 (50%)**: 很多多说话人音频被误判为单说话人
   - 说明聚类阈值可能过高，导致不同说话人的embedding被合并
   
2. **假阳性 (30个)**: 单说话人被误判为多说话人
   - 说明某些情况下聚类过于敏感，将同一说话人的不同片段分开了

3. **精确率低 (30.2%)**: 预测为多说话人的情况下，只有30%是真的多说话人

## 参数调优建议

### 1. 聚类阈值调整 (最重要)

**当前设置**: `mer_cos = 0.3`, `fix_cos_thr = 0.3`

**问题**: 聚类阈值在代码中硬编码，需要修改 `speakerlab/bin/infer_diarization.py` 中的 `get_cluster_backend()` 函数。

**建议调整**:

#### 方案A: 降低阈值以提高召回率（减少漏检）
```python
'mer_cos': 0.25,  # 从0.3降到0.25
'fix_cos_thr': 0.25,  # 从0.3降到0.25
```
- **效果**: 更容易区分不同说话人，减少FN（漏检）
- **风险**: 可能增加FP（假阳性）

#### 方案B: 提高阈值以减少假阳性
```python
'mer_cos': 0.35,  # 从0.3提高到0.35
'fix_cos_thr': 0.35,  # 从0.3提高到0.35
```
- **效果**: 减少单说话人被误判为多说话人
- **风险**: 可能增加FN（漏检）

#### 方案C: 渐进式调优（推荐）
先尝试 `0.25`，如果FP增加太多，再调整到 `0.28` 或 `0.27`

### 2. VAD参数调整

#### 2.1 最小VAD段长度 (`--min_vad_seg_dur`)
**当前**: 0.0秒（默认）

**作用阶段**: 聚类准备阶段（在生成chunks时过滤）

**建议**:
- 如果短片段导致embedding质量差，可以设置 `--min_vad_seg_dur 0.5` (0.5秒)
- 过滤掉太短的片段，提高embedding质量

#### 2.2 VAD合并间隔 (`--vad_merge_gap_ms`)
**当前**: 320ms（默认）

**作用阶段**: VAD后处理阶段（合并相邻段）

**建议**:
- 如果语音片段被过度分割，可以增加到 `--vad_merge_gap_ms 500`
- 如果希望更细粒度，可以减少到 `--vad_merge_gap_ms 200`

#### 2.3 VAD最小活动时长 (`--vad_min_active_ms`)
**⚠️ 注意**: 此参数在代码中**未被使用**，可能是遗留参数。

**实际应该使用**: `--vad_min_speech_ms` (VAD后处理阶段，移除太短的语音段)

**建议**:
- 使用 `--vad_min_speech_ms` 代替 `--vad_min_active_ms`
- 如果短语音片段导致问题，可以设置 `--vad_min_speech_ms 300` (300毫秒)

### 3. Embedding提取模式

#### `--no_chunk_after_vad` 选项

**默认行为** (不使用该选项):
- VAD段会被分割成固定大小的子段，使用滑动窗口提取embedding
- 优点: 更细粒度的embedding，适合长段语音
- 缺点: 可能产生更多噪声embedding

**使用 `--no_chunk_after_vad`**:
- 每个VAD段只提取一个embedding
- 优点: 更稳定的embedding，减少噪声
- 缺点: 对于长段语音可能不够细粒度

**建议**: 
- 如果当前漏检率高，可以尝试使用 `--no_chunk_after_vad`
- 这样可以减少embedding数量，提高聚类稳定性

### 4. 其他VAD后处理参数

这些参数在 `infer_diarization.py` 中可用，但当前脚本未暴露：

- `--vad_min_speech_ms`: 最小语音段时长（默认200ms）
- `--vad_max_silence_ms`: 最大静音间隔（默认300ms）
- `--vad_energy_threshold`: 能量阈值（默认0.05）
- `--vad_boundary_expansion_ms`: 边界扩展（默认10ms）

## 推荐调优流程

### 第一步: 修改聚类阈值（最重要）

1. 编辑 `speakerlab/bin/infer_diarization.py`
2. 找到 `get_cluster_backend()` 函数（约第101行）
3. 修改聚类参数：
```python
def get_cluster_backend():
    conf = {
        'cluster':{
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args':{
                'cluster_type': 'AHC',
                'mer_cos': 0.25,  # 从0.3改为0.25
                'min_cluster_size': 0,
                'fix_cos_thr': 0.25,  # 从0.3改为0.25
            }
        }
    }
    config = Config(conf)
    return build('cluster', config)
```

### 第二步: 使用优化参数运行

```bash
python run_diarization_speech_estimate.py \
    --min_vad_seg_dur 0.5 \
    --vad_merge_gap_ms 400 \
    --no_chunk_after_vad
```

**注意**: `--vad_min_active_ms` 参数未被使用，已从命令中移除。如需过滤短语音段，应使用 `--vad_min_speech_ms`（但该参数在当前脚本中未暴露，需要修改代码或直接修改 `infer_diarization.py`）。

### 第三步: 评估并迭代

1. 运行新的diarization
2. 评估结果
3. 根据新的混淆矩阵调整参数：
   - 如果FN仍然高 → 进一步降低聚类阈值（如0.23）
   - 如果FP增加太多 → 提高聚类阈值（如0.27）
   - 如果精确率仍然低 → 考虑增加 `min_vad_seg_dur`

## 参数优先级

1. **最高优先级**: 聚类阈值 (`mer_cos`, `fix_cos_thr`)
2. **高优先级**: `--min_vad_seg_dur`, `--no_chunk_after_vad`
3. **中优先级**: `--vad_merge_gap_ms`
4. **低优先级**: 其他VAD后处理参数（`--vad_min_speech_ms`, `--vad_max_silence_ms` 等）

**⚠️ 参数说明**:
- `--vad_min_active_ms`: **未使用**，可忽略
- `--min_vad_seg_dur`: 在聚类准备阶段过滤短段（单位：秒）
- `--vad_min_speech_ms`: 在VAD后处理阶段移除短段（单位：毫秒），但当前脚本未暴露此参数

## 预期效果

### 目标指标
- **准确率**: 从75.4%提升到 >80%
- **召回率**: 从50%提升到 >65%（减少漏检）
- **精确率**: 从30.2%提升到 >40%
- **F1分数**: 从37.7%提升到 >50%

### 预期变化
- **FN减少**: 通过降低聚类阈值，更多多说话人被正确识别
- **FP可能增加**: 需要平衡，通过VAD参数优化来缓解

## 注意事项

1. **聚类阈值是双刃剑**: 
   - 太低 → 增加FP（单说话人被误判）
   - 太高 → 增加FN（多说话人被漏检）

2. **需要多次迭代**: 根据实际数据特点调整

3. **考虑数据特性**: 
   - 如果音频质量差，可能需要更保守的阈值
   - 如果说话人声音相似，可能需要更低的阈值

4. **评估方法**: 建议在小样本上先测试，再全量运行

