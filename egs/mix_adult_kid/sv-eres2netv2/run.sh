#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

stage=4
stop_stage=4

data=/root/workspace/speaker_verification/mix_adult_kid/data
exp=/root/workspace/speaker_verification/mix_adult_kid/exp
exp_dir=$exp/eres2netv2
exp_lm_dir=$exp/eres2netv2_lm
test_dataset_name=combined_datasets_test_fixratio
master_port=29501

gpus="0 1 2 3"

. utils/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # In this stage we prepare the raw datasets.
  echo "Stage1: Preparing dataset..."
  ./local/prepare_data.sh --stage 3 --stop_stage 3 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # In this stage we prepare the data index files for training.
  echo "Stage2: Preparing training data index files..."
  python local/prepare_data_csv.py --data_dir $data/combined_datasets_train --nj 32
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Train the speaker embedding model.
  echo "Stage3: Training the speaker model..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  torchrun --nproc_per_node=$num_gpu --master_port $master_port speakerlab/bin/train.py --config conf/eres2netv2.yaml --gpu $gpus \
           --data $data/combined_datasets_train/train.csv --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Using large-margin-finetune strategy.
  echo "Stage4: finetune the model using large-margin"
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  # Change parameters in eres2net_lm.yaml.
  mkdir -p $exp_lm_dir/models/CKPT-EPOCH-0-00
  cp -r $exp_dir/models/CKPT-EPOCH-70-00/* $exp_lm_dir/models/CKPT-EPOCH-0-00/
  sed -i 's/70/0/g' $exp_lm_dir/models/CKPT-EPOCH-0-00/CKPT.yaml $exp_lm_dir/models/CKPT-EPOCH-0-00/epoch_counter.ckpt
  torchrun --nproc_per_node=$num_gpu --master_port $master_port speakerlab/bin/train.py --config conf/eres2netv2_lm.yaml --gpu $gpus \
           --data $data/combined_datasets_train/train.csv --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_lm_dir
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Extract embeddings of test datasets.
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  echo "Stage5: Extracting speaker embeddings..."
  torchrun --nproc_per_node=$num_gpu --master_port $master_port speakerlab/bin/extract.py --exp_dir $exp_lm_dir \
           --test_dataset_name $test_dataset_name --data $data/${test_dataset_name}/wav.scp --use_gpu --gpu $gpus
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Output score metrics.
  echo "Stage6: Computing score metrics..."
  trials="$data/${test_dataset_name}/trials_mix_adult_kid"
  python speakerlab/bin/compute_score_metrics.py --enrol_data $exp_lm_dir/$test_dataset_name/embeddings \
                                                 --test_data $exp_lm_dir/$test_dataset_name/embeddings \
                                                 --scores_dir $exp_lm_dir/$test_dataset_name/scores --trials $trials
fi
