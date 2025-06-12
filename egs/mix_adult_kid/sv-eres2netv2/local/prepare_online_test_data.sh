#!/bin/bash

. ./path.sh || exit 1

data=/root/workspace/speaker_verification/mix_adult_kid/data

. utils/parse_options.sh || exit 1

voxceleb1_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/voxceleb/vox1
online_data_raw_dir=/root/group-shared/voiceprint/data/test_data/speaker_verification/online_data_20250530_50kids
combined_test_dir=${data}/combined_datasets_test_online_20250530

echo "Prepare wav.scp for voxceleb1 and online_data_20250530_50kids datasets"
export LC_ALL=C 

# 1. VoxCeleb1
echo "Prepare wav.scp for voxceleb1 datasets"
mkdir -p ${data}/vox1
find ${voxceleb1_raw_dir}/wav -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/vox1/wav.scp
awk '{print $1}' ${data}/vox1/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vox1/utt2spk
./utils/utt2spk_to_spk2utt.pl ${data}/vox1/utt2spk >${data}/vox1/spk2utt


# 2. online_data_20250530_50kids
echo "Prepare wav.scp for online_data_20250530_50kids datasets"
mkdir -p ${data}/online_data_20250530_50kids
find ${online_data_raw_dir}/downloaded_audio -name "*for_register*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort > ${data}/online_data_20250530_50kids/wav.scp
awk '{print $1}' ${data}/online_data_20250530_50kids/wav.scp | awk -F "/" '{print $0,$1}' > ${data}/online_data_20250530_50kids/utt2spk
./utils/utt2spk_to_spk2utt.pl ${data}/online_data_20250530_50kids/utt2spk > ${data}/online_data_20250530_50kids/spk2utt

# Create combined dataset with unique dataset identifiers for speakers
echo "Creating combined dataset with prefixed speaker IDs..."
mkdir -p ${combined_test_dir}

# Create empty files for combined dataset
> ${combined_test_dir}/wav.scp
> ${combined_test_dir}/utt2spk

# Process voxceleb1 dataset and add prefix
echo "Adding voxceleb1 to combined dataset with VOX1_ prefix..."
awk '{sub(/\.wav$/,"",$1); print "VOX1_"$1, $2}' ${data}/vox1/wav.scp >> ${combined_test_dir}/wav.scp
awk '{sub(/\.wav$/,"",$1); print "VOX1_"$1, "VOX1_"$2}' ${data}/vox1/utt2spk >> ${combined_test_dir}/utt2spk

# Process online_data_20250530_50kids dataset and add prefix
echo "Adding online_data_20250530_50kids to combined dataset with Online_20250530_ prefix..."
awk '{sub(/\.wav$/,"",$1); print "Online_20250530_"$1, $2}' ${data}/online_data_20250530_50kids/wav.scp >> ${combined_test_dir}/wav.scp
awk '{sub(/\.wav$/,"",$1); print "Online_20250530_"$1, "Online_20250530_"$2}' ${data}/online_data_20250530_50kids/utt2spk >> ${combined_test_dir}/utt2spk

# Generate spk2utt file for combined dataset
echo "Generating spk2utt for combined dataset..."
./utils/utt2spk_to_spk2utt.pl ${combined_test_dir}/utt2spk > ${combined_test_dir}/spk2utt

# Count speakers and utterances in the combined dataset
spk_count=$(wc -l < ${combined_test_dir}/spk2utt)
utt_count=$(wc -l < ${combined_test_dir}/utt2spk)

echo "Combined dataset created with $spk_count speakers and $utt_count utterances"

# Create trial file
echo "Creating trial file..."
python ./local/create_online_trial.py --data-dir ${combined_test_dir} --output-trials ${combined_test_dir}/trials_mix_adult_kid

echo "Data Preparation Success !!!"

