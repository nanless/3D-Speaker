#!/bin/bash
# Script to re-extract missing embeddings

DATA_ROOT="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments"
MODEL_PATH="/root/workspace/speaker_verification/mix_adult_kid/exp/eres2netv2_lm/models/CKPT-EPOCH-9-00/embedding_model.ckpt"
CONFIG_FILE="conf/eres2netv2_lm.yaml"
OUTPUT_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings"

echo "Re-extracting 17 missing embeddings..."

python extract_missing_embeddings.py \
    --missing_files_json "missing_files.json" \
    --model_path "$MODEL_PATH" \
    --config_file "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --use_gpu \
    --gpu 0

echo "Re-extraction completed."
