# Speaker Embedding Extraction Pipeline

This pipeline extracts speaker embeddings from audio files organized in a hierarchical directory structure using multi-GPU processing.

## Directory Structure

The pipeline expects audio files to be organized as follows:
```
data_root/
├── dataset1/
│   ├── speaker1/
│   │   ├── utterance1.wav
│   │   ├── utterance2.wav
│   │   └── ...
│   ├── speaker2/
│   │   └── ...
│   └── ...
├── dataset2/
│   └── ...
└── ...
```

## Files Description

- **`extract_speaker_embeddings.py`**: Main extraction script with multi-GPU support
- **`merge_embeddings.py`**: Script to merge results from multiple GPU ranks
- **`run_embedding_extraction.sh`**: Shell script to run the complete pipeline
- **`example_usage.py`**: Example script showing how to use the extracted embeddings

## Usage

### Quick Start

1. **Run the complete pipeline:**
   ```bash
   ./run_embedding_extraction.sh
   ```

2. **Run specific stages:**
   ```bash
   # Stage 1: Extract embeddings
   ./run_embedding_extraction.sh --stage 1 --stop_stage 1
   
   # Stage 2: Merge results
   ./run_embedding_extraction.sh --stage 2 --stop_stage 2
   
   # Stage 3: Clean up
   ./run_embedding_extraction.sh --stage 3 --stop_stage 3
   ```

### Custom Configuration

Edit the configuration variables in `run_embedding_extraction.sh`:

```bash
DATA_ROOT="/path/to/your/audio/data"
MODEL_PATH="/path/to/your/model/checkpoint.ckpt"
CONFIG_FILE="/path/to/your/config.yaml"
OUTPUT_DIR="embeddings_output"
FINAL_OUTPUT_DIR="final_embeddings"
GPUS="0 1 2 3"  # GPU IDs to use
BATCH_SIZE=32
```

### Manual Execution

1. **Extract embeddings using multi-GPU:**
   ```bash
   torchrun --nproc_per_node=4 --master_port=29502 \
       extract_speaker_embeddings.py \
       --data_root /path/to/data \
       --model_path /path/to/model.ckpt \
       --config_file /path/to/config.yaml \
       --output_dir embeddings_output \
       --use_gpu \
       --gpu 0 1 2 3 \
       --batch_size 32
   ```

2. **Merge results:**
   ```bash
   python merge_embeddings.py \
       --input_dir embeddings_output \
       --output_dir final_embeddings
   ```

## Output Files

The pipeline generates the following output files in the `final_embeddings/` directory:

### Primary Output Files

- **`utterance_embeddings.pkl`**: Dictionary containing embeddings for each utterance
  ```python
  {
      'dataset_speaker_utterance': {
          'embedding': numpy.ndarray,      # Embedding vector
          'dataset': str,                  # Dataset name
          'speaker_id': str,               # Speaker identifier
          'utterance_id': str,             # Utterance identifier
          'path': str                      # Original audio file path
      }
  }
  ```

- **`speaker_embeddings.pkl`**: Dictionary containing averaged embeddings for each speaker
  ```python
  {
      'dataset_speaker': {
          'embedding': numpy.ndarray,      # Averaged embedding vector
          'dataset': str,                  # Dataset name
          'speaker_id': str,               # Speaker identifier
          'num_utterances': int            # Number of utterances used for averaging
      }
  }
  ```

### Metadata Files

- **`utterance_list.json`**: List of all utterances with metadata (without embeddings)
- **`speaker_list.json`**: List of all speakers with metadata (without embeddings)
- **`summary.json`**: Overall statistics and dataset breakdown

## Loading and Using Embeddings

### Basic Usage

```python
import pickle
import json

# Load utterance embeddings
with open('final_embeddings/utterance_embeddings.pkl', 'rb') as f:
    utterance_embeddings = pickle.load(f)

# Load speaker embeddings
with open('final_embeddings/speaker_embeddings.pkl', 'rb') as f:
    speaker_embeddings = pickle.load(f)

# Load metadata
with open('final_embeddings/summary.json', 'r') as f:
    summary = json.load(f)

print(f"Total utterances: {summary['total_utterances']}")
print(f"Total speakers: {summary['total_speakers']}")
print(f"Embedding dimension: {summary['embedding_dimension']}")
```

### Advanced Usage

Run the example script to see various usage patterns:
```bash
python example_usage.py
```

This script demonstrates:
- Loading embeddings and metadata
- Computing similarity between utterances
- Finding utterances from the same speaker
- Comparing speaker embeddings
- Searching by dataset or speaker criteria

## Pipeline Configuration

### Model Configuration

The pipeline uses the configuration file specified in `CONFIG_FILE`. Key parameters:

- **`sample_rate`**: Expected audio sample rate (16000 Hz)
- **`fbank_dim`**: Feature dimension (80)
- **`embedding_size`**: Output embedding dimension (192)

### GPU Configuration

- Supports multi-GPU processing using `torchrun`
- Automatically distributes work across available GPUs
- Each GPU processes a subset of audio files independently
- Results are merged after all GPUs complete

### Memory Considerations

- Large datasets may require significant memory for storing embeddings
- Consider the number of utterances and speakers when planning storage
- Intermediate files can be cleaned up after merging (Stage 3)

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce `batch_size` in the configuration
   - Use fewer GPUs or smaller batches per GPU

2. **Sample rate mismatch:**
   - The pipeline automatically resamples audio to match the model's expected sample rate
   - Ensure `torchaudio` is installed for resampling support

3. **Missing dependencies:**
   ```bash
   pip install torch torchaudio numpy scikit-learn tqdm kaldiio
   ```

4. **Path issues:**
   - Ensure all paths in the configuration are absolute paths
   - Verify that the model checkpoint and config files exist

### Performance Tips

- Use SSD storage for faster I/O
- Increase `num_workers` in the data loading if CPU is not the bottleneck
- Monitor GPU utilization and adjust batch size accordingly
- Use appropriate `master_port` to avoid conflicts with other processes

## Technical Details

### Multi-GPU Processing

The pipeline uses PyTorch's `torchrun` for distributed processing:
- Each GPU rank processes a different subset of audio files
- Work is distributed evenly across available GPUs
- Progress bars show processing status for each GPU
- Results are saved separately by rank and then merged

### Embedding Computation

1. **Feature Extraction**: Audio files are converted to mel-scale filterbank features
2. **Model Inference**: Features are passed through the ERes2NetV2 model
3. **Speaker Averaging**: Multiple utterances from the same speaker are averaged

### File Organization

```
output_structure/
├── embeddings_output/           # Intermediate results
│   ├── rank_00/                # GPU 0 results
│   │   ├── utterance_embeddings.pkl
│   │   ├── speaker_embeddings.pkl
│   │   └── metadata.json
│   ├── rank_01/                # GPU 1 results
│   └── ...
└── final_embeddings/           # Final merged results
    ├── utterance_embeddings.pkl
    ├── speaker_embeddings.pkl
    ├── utterance_list.json
    ├── speaker_list.json
    └── summary.json
```

## Citation

If you use this pipeline in your research, please cite the original 3D-Speaker paper:

```bibtex
@article{3dspeaker,
  title={3D-Speaker: A Large-Scale Multi-Device, Multi-Distance, and Multi-Dialect Corpus for Speech Representation Disentanglement},
  author={...},
  journal={...},
  year={...}
}
``` 