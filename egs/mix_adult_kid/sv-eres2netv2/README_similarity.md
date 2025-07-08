# Speaker Embedding Similarity Analysis

This directory contains tools for extracting speaker embeddings and computing pairwise similarity between speakers.

## Features

- **Multi-GPU Embedding Extraction**: Extract speaker-level embeddings using multiple GPUs with proper progress tracking
- **Individual File Storage**: Save each utterance and speaker embedding as separate files with preserved directory structure
- **Similarity Computation**: Calculate cosine similarity between all speaker pairs
- **Visualization**: Generate heatmaps and distribution plots
- **Analysis Tools**: Identify high similarity pairs and potential data quality issues

## Files

- `extract_speaker_embeddings.py`: Main extraction script with similarity computation
- `merge_embeddings.py`: Merge results from multiple GPU ranks and compute similarity
- `run_embedding_extraction.sh`: Complete pipeline script
- `analyze_speaker_similarity.py`: Detailed analysis tool for similarity results
- `load_individual_embeddings.py`: Demonstration tool for loading individual embedding files

## Quick Start

### 1. Basic Embedding Extraction (without similarity)

```bash
# Extract embeddings only
./run_embedding_extraction.sh --stage 1 --stop_stage 2
```

### 2. Embedding Extraction with Similarity Analysis

```bash
# Enable similarity computation by setting COMPUTE_SIMILARITY=true in the script
# Or run with custom parameters:

# Edit the script to set:
COMPUTE_SIMILARITY=true
SIMILARITY_THRESHOLD=0.8
SAVE_INDIVIDUAL=true
MERGE_INDIVIDUAL=true

# Then run the full pipeline
./run_embedding_extraction.sh --stage 1 --stop_stage 4
```

### 3. Manual Similarity Computation

```bash
# If you already have embeddings and want to compute similarity
python merge_embeddings.py \
    --input_dir embeddings_output \
    --output_dir final_embeddings \
    --compute_similarity \
    --similarity_threshold 0.7 \
    --merge_individual
```

### 4. Detailed Analysis

```bash
# Analyze similarity results
python analyze_speaker_similarity.py \
    --results_dir final_embeddings \
    --output_csv detailed_similarity.csv \
    --min_similarity 0.6 \
    --max_results 50
```

### 5. Load Individual Embeddings

```bash
# Explore individual embedding files
python load_individual_embeddings.py \
    --embeddings_dir final_embeddings/embeddings_individual \
    --dataset dataset_name \
    --speaker_id speaker_001 \
    --utterance_id utterance_001 \
    --show_stats
```

## Configuration Parameters

### In `run_embedding_extraction.sh`:

- `COMPUTE_SIMILARITY`: Set to `true` to enable similarity computation
- `SIMILARITY_THRESHOLD`: Threshold for identifying high similarity pairs (default: 0.8)
- `SAVE_INDIVIDUAL`: Set to `true` to save individual embedding files (default: true)
- `MERGE_INDIVIDUAL`: Set to `true` to merge individual files from all ranks (default: true)
- `BATCH_SIZE`: Batch size for processing (default: 32)
- `GPUS`: GPU IDs to use (default: "0 1 2 3")

### Command Line Arguments:

#### `extract_speaker_embeddings.py`:
```bash
python extract_speaker_embeddings.py \
    --data_root /path/to/audio/data \
    --model_path /path/to/model.ckpt \
    --config_file conf/config.yaml \
    --output_dir embeddings_output \
    --compute_similarity \
    --similarity_threshold 0.8 \
    --save_individual \
    --use_gpu \
    --batch_size 32
```

#### `merge_embeddings.py`:
```bash
python merge_embeddings.py \
    --input_dir embeddings_output \
    --output_dir final_embeddings \
    --compute_similarity \
    --similarity_threshold 0.8 \
    --merge_individual
```

#### `load_individual_embeddings.py`:
```bash
python load_individual_embeddings.py \
    --embeddings_dir final_embeddings/embeddings_individual \
    --dataset dataset_name \
    --speaker_id speaker_001 \
    --utterance_id utterance_001 \
    --show_stats
```

## Multi-GPU Processing

The pipeline now properly supports multi-GPU processing with individual progress tracking:

- **Each GPU rank shows its own progress bar**
- **All ranks process different subsets of the data simultaneously**
- **Individual embedding files are saved by each rank**
- **Results are merged after all ranks complete**

### GPU Status Monitoring:
```bash
# Each rank will show output like:
# Rank 0/4: Starting on device cuda:0
# Rank 1/4: Starting on device cuda:1
# Rank 0: Processing 1000 files (indices 0-999)
# Rank 1: Processing 1000 files (indices 1000-1999)
```

## Output Files

### After running the pipeline, you'll get:

#### Basic Embeddings:
- `utterance_embeddings.pkl`: Utterance-level embeddings (combined)
- `speaker_embeddings.pkl`: Speaker-level averaged embeddings (combined)
- `utterance_list.json`: Utterance metadata
- `speaker_list.json`: Speaker metadata
- `summary.json`: Overall statistics

#### Individual Embedding Files (when enabled):
- `embeddings_individual/utterances/dataset/speaker_id/utterance_id.pkl`: Individual utterance embeddings
- `embeddings_individual/speakers/dataset/speaker_id.pkl`: Individual speaker embeddings
- `embeddings_individual/file_index.json`: Index for quick file lookup

#### Similarity Analysis (when enabled):
- `speaker_similarity.json`: Detailed similarity results and statistics
- `similarity_matrix.npy`: Full similarity matrix as numpy array
- `speaker_keys_mapping.json`: Speaker key to matrix index mapping
- `similarity_heatmap.png`: Similarity heatmap visualization
- `similarity_analysis.png`: Distribution analysis plots
- `high_similarity_pairs.json`: List of high similarity pairs

## Directory Structure

### Individual Embedding Files:
```
final_embeddings/embeddings_individual/
├── utterances/
│   ├── dataset1/
│   │   ├── speaker001/
│   │   │   ├── utterance001.pkl
│   │   │   ├── utterance002.pkl
│   │   │   └── ...
│   │   └── speaker002/
│   │       └── ...
│   └── dataset2/
│       └── ...
├── speakers/
│   ├── dataset1/
│   │   ├── speaker001.pkl
│   │   ├── speaker002.pkl
│   │   └── ...
│   └── dataset2/
│       └── ...
└── file_index.json
```

## Loading Results in Python

### Load Combined Embeddings:
```python
import pickle
import numpy as np

# Load embeddings
with open('final_embeddings/speaker_embeddings.pkl', 'rb') as f:
    speaker_embeddings = pickle.load(f)

with open('final_embeddings/utterance_embeddings.pkl', 'rb') as f:
    utterance_embeddings = pickle.load(f)
```

### Load Individual Embedding Files:
```python
import pickle
import json

# Load file index
with open('final_embeddings/embeddings_individual/file_index.json', 'r') as f:
    index = json.load(f)

# Load a specific utterance
utterance_key = 'dataset1_speaker001_utterance001'
if utterance_key in index['utterances']:
    file_path = index['utterances'][utterance_key]
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        embedding = data['embedding']
        info = data['info']

# Load a specific speaker
speaker_key = 'dataset1_speaker001'
if speaker_key in index['speakers']:
    file_path = index['speakers'][speaker_key]
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        embedding = data['embedding']
        info = data['info']
```

### Load Similarity Results:
```python
import json
import numpy as np

# Load similarity results
with open('final_embeddings/speaker_similarity.json', 'r') as f:
    similarity_results = json.load(f)

# Load similarity matrix
similarity_matrix = np.load('final_embeddings/similarity_matrix.npy')

# Load speaker mapping
with open('final_embeddings/speaker_keys_mapping.json', 'r') as f:
    speaker_mapping = json.load(f)
```

### Example Analysis:
```python
# Get similarity statistics
stats = similarity_results['statistics']
print(f"Mean similarity: {stats['mean_similarity']:.4f}")
print(f"High similarity pairs: {stats['num_high_similarity_pairs']}")

# Find most similar pair
high_sim_pairs = similarity_results['high_similarity_pairs']
if high_sim_pairs:
    most_similar = max(high_sim_pairs, key=lambda x: x['similarity'])
    print(f"Most similar pair: {most_similar['similarity']:.4f}")
    print(f"Speaker 1: {most_similar['speaker1']}")
    print(f"Speaker 2: {most_similar['speaker2']}")
```

## Understanding the Results

### Similarity Scores:
- Range: 0.0 to 1.0 (cosine similarity)
- **> 0.9**: Very high similarity (potential duplicates)
- **0.7-0.9**: High similarity (possibly related speakers)
- **0.5-0.7**: Moderate similarity
- **< 0.5**: Low similarity

### Potential Issues to Check:

1. **Very High Similarity (>0.95)**:
   - May indicate duplicate speakers
   - Same person recorded multiple times
   - Data quality issues

2. **Cross-Dataset High Similarity**:
   - Same speakers appearing in different datasets
   - Potential data leakage

3. **Distribution Analysis**:
   - Check if similarity distribution is reasonable
   - Look for unexpected peaks or patterns

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size or use fewer GPUs
2. **Missing Dependencies**: Install required packages:
   ```bash
   pip install scikit-learn matplotlib seaborn pandas
   ```
3. **No Similarity Results**: Ensure `--compute_similarity` flag is used
4. **Visualization Errors**: Install matplotlib and seaborn
5. **Only Rank 0 Working**: Check distributed training setup and make sure all GPUs are accessible

### Performance Tips:

1. **Large Datasets**: Use higher similarity thresholds to reduce output size
2. **Memory Usage**: Similarity computation requires O(n²) memory for n speakers
3. **GPU Usage**: Multi-GPU extraction is much faster than single GPU
4. **Individual Files**: Saving individual files adds I/O overhead but provides flexibility

### Multi-GPU Debugging:
```bash
# Check GPU availability
nvidia-smi

# Monitor GPU usage during extraction
watch -n 1 nvidia-smi

# Check if all ranks are working
grep "Rank" logs/extraction.log
```

## Pipeline Stages

1. **Stage 1**: Extract embeddings using multi-GPU processing
2. **Stage 2**: Merge results and compute similarity (if enabled)
3. **Stage 3**: Generate analysis report and statistics
4. **Stage 4**: Clean up intermediate files (optional)

You can run specific stages using:
```bash
./run_embedding_extraction.sh --stage 2 --stop_stage 3
```

## Advanced Usage

### Custom Analysis with Individual Files:
```python
# Example: Analyze intra-speaker variability
import os
import pickle
import numpy as np
from pathlib import Path

def analyze_speaker_variability(embeddings_dir, dataset, speaker_id):
    utterance_dir = os.path.join(embeddings_dir, 'utterances', dataset, speaker_id)
    
    embeddings = []
    for pkl_file in Path(utterance_dir).glob('*.pkl'):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            embeddings.append(data['embedding'])
    
    if len(embeddings) > 1:
        embeddings = np.array(embeddings)
        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        
        print(f"Speaker {speaker_id} variability:")
        print(f"  Mean intra-speaker similarity: {np.mean(upper_triangle):.4f}")
        print(f"  Std intra-speaker similarity: {np.std(upper_triangle):.4f}")
        print(f"  Min intra-speaker similarity: {np.min(upper_triangle):.4f}")
        
        return upper_triangle
    
    return None

# Usage
variability = analyze_speaker_variability(
    'final_embeddings/embeddings_individual', 
    'dataset1', 
    'speaker001'
) 