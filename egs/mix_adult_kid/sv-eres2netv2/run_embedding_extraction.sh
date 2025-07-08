#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

# Configuration
DATA_ROOT="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments"
MODEL_PATH="/root/workspace/speaker_verification/mix_adult_kid/exp/eres2netv2_lm/models/CKPT-EPOCH-9-00/embedding_model.ckpt"
CONFIG_FILE="conf/eres2netv2_lm.yaml"
OUTPUT_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings"
FINAL_OUTPUT_DIR="final_embeddings"
MASTER_PORT=29502
GPUS="0 1 2 3"  # Available GPUs
BATCH_SIZE=32

# Similarity computation parameters
COMPUTE_SIMILARITY=true  # Set to true to compute speaker similarity
SIMILARITY_THRESHOLD=0.8 # Threshold for identifying similar speakers

# Individual file saving parameters
SAVE_INDIVIDUAL=true     # Set to true to save individual embedding files
MERGE_INDIVIDUAL=true    # Set to true to merge individual files from all ranks

# Parse command line arguments
stage=1
stop_stage=4

. ./utils/parse_options.sh || exit 1

echo "=== Embedding Extraction Pipeline ==="
echo "Data root: $DATA_ROOT"
echo "Model path: $MODEL_PATH"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Final output directory: $FINAL_OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Compute similarity: $COMPUTE_SIMILARITY"
echo "Similarity threshold: $SIMILARITY_THRESHOLD"
echo "Save individual files: $SAVE_INDIVIDUAL"
echo "Merge individual files: $MERGE_INDIVIDUAL"
echo "======================================"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Extracting embeddings using multi-GPU processing..."
    
    # Count number of GPUs
    num_gpu=$(echo $GPUS | awk -F ' ' '{print NF}')
    echo "Using $num_gpu GPUs: $GPUS"
    
    # Prepare similarity computation arguments
    similarity_args=""
    if [ "$COMPUTE_SIMILARITY" = true ]; then
        similarity_args="--compute_similarity --similarity_threshold $SIMILARITY_THRESHOLD"
    fi
    
    # Prepare individual file saving arguments
    individual_args=""
    if [ "$SAVE_INDIVIDUAL" = true ]; then
        individual_args="--save_individual"
    fi
    
    # Run multi-GPU embedding extraction
    torchrun --nproc_per_node=$num_gpu --master_port $MASTER_PORT \
        extract_speaker_embeddings.py \
        --data_root "$DATA_ROOT" \
        --model_path "$MODEL_PATH" \
        --config_file "$CONFIG_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --use_gpu \
        --gpu $GPUS \
        --batch_size $BATCH_SIZE \
        $similarity_args \
        $individual_args
    
    echo "Stage 1 completed."
    
    # Check if individual files were saved
    if [ "$SAVE_INDIVIDUAL" = true ]; then
        individual_dir="$OUTPUT_DIR/embeddings_individual"
        if [ -d "$individual_dir" ]; then
            echo "Individual embedding files saved to: $individual_dir"
            echo "Directory structure:"
            echo "  $individual_dir/utterances/dataset_name/speaker_id/utterance_id.pkl"
            echo "  $individual_dir/speakers/dataset_name/speaker_id.pkl"
        fi
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Merging results from all GPU ranks..."
    
    # Prepare similarity computation arguments for merging
    merge_similarity_args=""
    if [ "$COMPUTE_SIMILARITY" = true ]; then
        merge_similarity_args="--compute_similarity --similarity_threshold $SIMILARITY_THRESHOLD"
    fi
    
    # Prepare individual file merging arguments
    merge_individual_args=""
    if [ "$MERGE_INDIVIDUAL" = true ]; then
        merge_individual_args="--merge_individual"
    fi
    
    python merge_embeddings.py \
        --input_dir "$OUTPUT_DIR" \
        --output_dir "$FINAL_OUTPUT_DIR" \
        $merge_similarity_args \
        $merge_individual_args
    
    echo "Stage 2 completed."
    
    # Check merged individual files
    if [ "$MERGE_INDIVIDUAL" = true ]; then
        final_individual_dir="$FINAL_OUTPUT_DIR/embeddings_individual"
        if [ -d "$final_individual_dir" ]; then
            echo "Merged individual embedding files available in: $final_individual_dir"
            
            # Count files
            utt_count=$(find "$final_individual_dir/utterances" -name "*.pkl" 2>/dev/null | wc -l)
            spk_count=$(find "$final_individual_dir/speakers" -name "*.pkl" 2>/dev/null | wc -l)
            
            echo "  Utterance files: $utt_count"
            echo "  Speaker files: $spk_count"
            
            # Show index file
            index_file="$final_individual_dir/file_index.json"
            if [ -f "$index_file" ]; then
                echo "  File index available: $index_file"
            fi
        fi
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Generating similarity analysis report..."
    
    if [ "$COMPUTE_SIMILARITY" = true ]; then
        echo "Creating detailed similarity analysis report..."
        python -c "
import json
import numpy as np
import os

output_dir = '$FINAL_OUTPUT_DIR'
similarity_file = os.path.join(output_dir, 'speaker_similarity.json')

if os.path.exists(similarity_file):
    with open(similarity_file, 'r') as f:
        results = json.load(f)
    
    stats = results['statistics']
    high_sim_pairs = results['high_similarity_pairs']
    
    print('\\n=== SPEAKER SIMILARITY ANALYSIS REPORT ===')
    print(f'Total number of speakers: {len(results[\"speaker_keys\"])}')
    print(f'Total speaker pairs analyzed: {stats[\"total_pairs\"]}')
    print(f'Similarity threshold used: {stats[\"threshold_used\"]}')
    print('')
    print('Similarity Statistics:')
    print(f'  Mean similarity: {stats[\"mean_similarity\"]:.4f}')
    print(f'  Median similarity: {stats[\"median_similarity\"]:.4f}')
    print(f'  Standard deviation: {stats[\"std_similarity\"]:.4f}')
    print(f'  Min similarity: {stats[\"min_similarity\"]:.4f}')
    print(f'  Max similarity: {stats[\"max_similarity\"]:.4f}')
    print('')
    print(f'High Similarity Pairs (>={stats[\"threshold_used\"]}): {stats[\"num_high_similarity_pairs\"]}')
    
    if high_sim_pairs:
        print('\\nTop 10 Most Similar Speaker Pairs:')
        sorted_pairs = sorted(high_sim_pairs, key=lambda x: x['similarity'], reverse=True)[:10]
        for i, pair in enumerate(sorted_pairs, 1):
            print(f'  {i}. {pair[\"speaker1\"]} <-> {pair[\"speaker2\"]}')
            print(f'     Similarity: {pair[\"similarity\"]:.4f}')
            print(f'     Dataset1: {pair[\"speaker1_info\"][\"dataset\"]}, ID1: {pair[\"speaker1_info\"][\"speaker_id\"]}')
            print(f'     Dataset2: {pair[\"speaker2_info\"][\"dataset\"]}, ID2: {pair[\"speaker2_info\"][\"speaker_id\"]}')
            print('')
    else:
        print('No speaker pairs found above the similarity threshold.')
    
    print('=== END OF REPORT ===\\n')
else:
    print('Similarity analysis was not performed or results not found.')
"
    else
        echo "Similarity computation was disabled. Skipping analysis report."
    fi
    
    # Show individual file statistics if available
    if [ "$SAVE_INDIVIDUAL" = true ] && [ -d "$FINAL_OUTPUT_DIR/embeddings_individual" ]; then
        echo ""
        echo "=== INDIVIDUAL FILE STATISTICS ==="
        
        individual_dir="$FINAL_OUTPUT_DIR/embeddings_individual"
        
        # Count files by dataset
        if [ -d "$individual_dir/utterances" ]; then
            echo "Utterance files by dataset:"
            for dataset_dir in "$individual_dir/utterances"/*; do
                if [ -d "$dataset_dir" ]; then
                    dataset_name=$(basename "$dataset_dir")
                    count=$(find "$dataset_dir" -name "*.pkl" | wc -l)
                    speaker_count=$(ls "$dataset_dir" | wc -l)
                    echo "  $dataset_name: $count utterances from $speaker_count speakers"
                fi
            done
        fi
        
        if [ -d "$individual_dir/speakers" ]; then
            echo "Speaker files by dataset:"
            for dataset_dir in "$individual_dir/speakers"/*; do
                if [ -d "$dataset_dir" ]; then
                    dataset_name=$(basename "$dataset_dir")
                    count=$(find "$dataset_dir" -name "*.pkl" | wc -l)
                    echo "  $dataset_name: $count speakers"
                fi
            done
        fi
        
        echo "================================="
    fi
    
    echo "Stage 3 completed."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Cleaning up intermediate files..."
    
    # Remove rank-specific directories to save space (optional)
    read -p "Do you want to remove intermediate rank-specific files to save space? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing rank-specific directories..."
        rm -rf "$OUTPUT_DIR"/rank_*
        echo "Intermediate files cleaned up."
    else
        echo "Keeping intermediate files."
    fi
    
    echo "Stage 4 completed."
fi

echo "=== Pipeline Completed ==="
echo "Final results are available in: $FINAL_OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - $FINAL_OUTPUT_DIR/utterance_embeddings.pkl    (utterance-level embeddings)"
echo "  - $FINAL_OUTPUT_DIR/speaker_embeddings.pkl      (speaker-level averaged embeddings)"
echo "  - $FINAL_OUTPUT_DIR/utterance_list.json         (utterance metadata)"
echo "  - $FINAL_OUTPUT_DIR/speaker_list.json           (speaker metadata)"
echo "  - $FINAL_OUTPUT_DIR/summary.json                (overall statistics)"

if [ "$SAVE_INDIVIDUAL" = true ] && [ "$MERGE_INDIVIDUAL" = true ]; then
    echo ""
    echo "Individual embedding files:"
    echo "  - $FINAL_OUTPUT_DIR/embeddings_individual/utterances/  (individual utterance embeddings)"
    echo "  - $FINAL_OUTPUT_DIR/embeddings_individual/speakers/    (individual speaker embeddings)"
    echo "  - $FINAL_OUTPUT_DIR/embeddings_individual/file_index.json  (file index for quick lookup)"
    echo ""
    echo "Directory structure for individual files:"
    echo "  utterances: dataset_name/speaker_id/utterance_id.pkl"
    echo "  speakers:   dataset_name/speaker_id.pkl"
fi

if [ "$COMPUTE_SIMILARITY" = true ]; then
    echo ""
    echo "Similarity analysis files:"
    echo "  - $FINAL_OUTPUT_DIR/speaker_similarity.json     (detailed similarity results)"
    echo "  - $FINAL_OUTPUT_DIR/similarity_matrix.npy       (similarity matrix as numpy array)"
    echo "  - $FINAL_OUTPUT_DIR/speaker_keys_mapping.json   (speaker key to index mapping)"
    echo "  - $FINAL_OUTPUT_DIR/similarity_heatmap.png      (similarity heatmap visualization)"
    echo "  - $FINAL_OUTPUT_DIR/similarity_analysis.png     (similarity distribution analysis)"
    if [ -f "$FINAL_OUTPUT_DIR/high_similarity_pairs.json" ]; then
        echo "  - $FINAL_OUTPUT_DIR/high_similarity_pairs.json  (high similarity pairs list)"
    fi
fi

echo ""
echo "You can now load the embeddings using:"
echo "  import pickle"
echo "  with open('$FINAL_OUTPUT_DIR/utterance_embeddings.pkl', 'rb') as f:"
echo "      utterance_embeddings = pickle.load(f)"
echo "  with open('$FINAL_OUTPUT_DIR/speaker_embeddings.pkl', 'rb') as f:"
echo "      speaker_embeddings = pickle.load(f)"

if [ "$SAVE_INDIVIDUAL" = true ]; then
    echo ""
    echo "To load individual embedding files:"
    echo "  import pickle, json"
    echo "  # Load file index"
    echo "  with open('$FINAL_OUTPUT_DIR/embeddings_individual/file_index.json', 'r') as f:"
    echo "      index = json.load(f)"
    echo "  # Load a specific utterance"
    echo "  file_path = index['utterances']['dataset_speaker_utterance']"
    echo "  with open(file_path, 'rb') as f:"
    echo "      data = pickle.load(f)"
    echo "      embedding = data['embedding']"
    echo "      info = data['info']"
fi

if [ "$COMPUTE_SIMILARITY" = true ]; then
    echo ""
    echo "To load similarity results:"
    echo "  import json"
    echo "  import numpy as np"
    echo "  with open('$FINAL_OUTPUT_DIR/speaker_similarity.json', 'r') as f:"
    echo "      similarity_results = json.load(f)"
    echo "  similarity_matrix = np.load('$FINAL_OUTPUT_DIR/similarity_matrix.npy')"
fi

echo "==========================" 