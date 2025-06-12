#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

stage=1
stop_stage=2

data=/root/workspace/speaker_verification/mix_adult_kid/data
exp=/root/workspace/speaker_verification/mix_adult_kid/exp
model_dir=$exp/modelscope_eres2netv2
model_id="iic/speech_eres2netv2_sv_zh-cn_16k-common"
model_name="speech_eres2netv2_sv_zh-cn_16k-common"
# model_id="iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
# model_name="speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
test_dataset_name=combined_datasets_test_fixratio
master_port=29501

gpus="0 1 2 3"

. utils/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Extract embeddings using modelscope model
  echo "Stage1: Downloading model and extracting embeddings..."
  torchrun --master_port $master_port --nproc_per_node=1 \
    speakerlab/bin/extract_from_modelscope_model.py \
    --model_id $model_id \
    --local_model_dir $model_dir \
    --data $data/${test_dataset_name}/wav.scp \
    --use_gpu \
    --gpu $gpus \
    --num_workers 4 \
    --test_dataset_name $test_dataset_name
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Compute evaluation metrics
  echo "Stage2: Computing evaluation metrics..."
  trials="$data/${test_dataset_name}/trials_mix_adult_kid"
  python speakerlab/bin/compute_score_metrics.py \
    --enrol_data $model_dir/$model_name/$test_dataset_name/embeddings \
    --test_data $model_dir/$model_name/$test_dataset_name/embeddings \
    --scores_dir $model_dir/$model_name/$test_dataset_name/scores \
    --trials $trials
fi
