# 聚类阈值参数化补丁

## 问题

当前 `infer_diarization.py` 中的聚类阈值 (`mer_cos` 和 `fix_cos_thr`) 是硬编码的，无法通过命令行参数调整。

## 解决方案

修改 `speakerlab/bin/infer_diarization.py` 以支持可配置的聚类阈值。

### 步骤1: 添加命令行参数

在 `infer_diarization.py` 的参数解析部分（约第43-62行）添加：

```python
parser.add_argument('--cluster_mer_cos', default=0.3, type=float, 
                    help='Clustering merge cosine threshold (default: 0.3)')
parser.add_argument('--cluster_fix_cos_thr', default=0.3, type=float,
                    help='Clustering fixed cosine threshold (default: 0.3)')
```

### 步骤2: 修改 get_cluster_backend() 函数

将 `get_cluster_backend()` 函数（约第101-114行）修改为：

```python
def get_cluster_backend(mer_cos=0.3, fix_cos_thr=0.3):
    conf = {
        'cluster':{
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args':{
                'cluster_type': 'AHC',
                'mer_cos': mer_cos,
                'min_cluster_size': 0,
                'fix_cos_thr': fix_cos_thr,
            }
        }
    }
    config = Config(conf)
    return build('cluster', config)
```

### 步骤3: 修改 Diarization3Dspeaker 类

在 `Diarization3Dspeaker.__init__()` 方法中（约第208行）添加参数：

```python
def __init__(self, device=None, include_overlap=False, hf_access_token=None, 
             speaker_num=None, model_cache_dir=None,
             no_chunk_after_vad: bool=False, min_vad_seg_dur: float=0.0,
             vad_min_active_ms: float=None, vad_merge_gap_ms: float=None,
             vad_min_speech_ms: float=None, vad_max_silence_ms: float=None,
             vad_energy_threshold: float=None, vad_boundary_expansion_ms: float=None,
             vad_boundary_energy_percentile: float=None,
             cluster_mer_cos: float=0.3, cluster_fix_cos_thr: float=0.3):  # 新增参数
    # ... 现有代码 ...
    self.cluster = get_cluster_backend(cluster_mer_cos, cluster_fix_cos_thr)  # 修改这行
```

### 步骤4: 修改主函数

在 `main()` 函数中（约第850-900行），将参数传递给 `Diarization3Dspeaker`：

```python
diarization_pipeline = Diarization3Dspeaker(
    device,
    args.include_overlap,
    args.hf_access_token,
    args.speaker_num,
    None,
    args.no_chunk_after_vad,
    args.min_vad_seg_dur,
    args.vad_min_active_ms,
    args.vad_merge_gap_ms,
    args.vad_min_speech_ms,
    args.vad_max_silence_ms,
    args.vad_energy_threshold,
    args.vad_boundary_expansion_ms,
    args.vad_boundary_energy_percentile,
    args.cluster_mer_cos,  # 新增
    args.cluster_fix_cos_thr,  # 新增
)
```

## 使用方法

修改后，可以通过命令行参数调整聚类阈值：

```bash
python run_diarization_speech_estimate.py \
    --cluster_mer_cos 0.25 \
    --cluster_fix_cos_thr 0.25 \
    --min_vad_seg_dur 0.5 \
    --no_chunk_after_vad
```

## 快速修改脚本

如果不想手动修改，可以使用以下Python脚本自动应用补丁：

```python
# apply_clustering_patch.py
import re

file_path = 'speakerlab/bin/infer_diarization.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 添加命令行参数
if '--cluster_mer_cos' not in content:
    # 在 parser.add_argument('--vad_boundary_energy_percentile' 之后添加
    pattern = r"(parser\.add_argument\('--vad_boundary_energy_percentile'.*?\n)"
    replacement = r"\1parser.add_argument('--cluster_mer_cos', default=0.3, type=float, help='Clustering merge cosine threshold (default: 0.3)')\nparser.add_argument('--cluster_fix_cos_thr', default=0.3, type=float, help='Clustering fixed cosine threshold (default: 0.3)')\n"
    content = re.sub(pattern, replacement, content)

# 2. 修改 get_cluster_backend 函数
pattern = r"def get_cluster_backend\(\):\s+conf = \{[^}]+'mer_cos': 0\.3,[^}]+'fix_cos_thr': 0\.3,"
replacement = """def get_cluster_backend(mer_cos=0.3, fix_cos_thr=0.3):
    conf = {
        'cluster':{
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args':{
                'cluster_type': 'AHC',
                'mer_cos': mer_cos,
                'min_cluster_size': 0,
                'fix_cos_thr': fix_cos_thr,"""
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 3. 修改 Diarization3Dspeaker.__init__ 签名
# ... (需要更复杂的正则表达式匹配)

# 4. 修改 self.cluster 初始化
pattern = r"self\.cluster = get_cluster_backend\(\)"
replacement = "self.cluster = get_cluster_backend(self.cluster_mer_cos, self.cluster_fix_cos_thr)"
content = re.sub(pattern, replacement, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied successfully!")
```

## 注意事项

1. **备份原文件**: 修改前请备份 `speakerlab/bin/infer_diarization.py`
2. **测试**: 修改后在小样本上测试
3. **版本控制**: 如果使用git，建议创建新分支进行修改

## 临时方案（无需修改代码）

如果不想修改代码，可以直接编辑 `speakerlab/bin/infer_diarization.py` 第107和109行：

```python
'mer_cos': 0.25,  # 从 0.3 改为 0.25
'fix_cos_thr': 0.25,  # 从 0.3 改为 0.25
```

然后重新运行脚本。

