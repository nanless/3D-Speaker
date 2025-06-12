#!/bin/bash

# Copyright (c) 2023 Yafeng Chen (chenyafeng.cyf@alibaba-inc.com)
#               2023 Luyao Cheng (shuli.cly@alibaba-inc.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

stage=4
stop_stage=4
data=/root/workspace/speaker_verification/mix_adult_kid/data

. utils/parse_options.sh || exit 1

# Define raw data directories
musan_raw_dir=/root/group-shared/voiceprint/data/noise/musan/musan
rirs_raw_dir=/root/group-shared/voiceprint/data/noise/rirs_noises
voxceleb2_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/voxceleb/vox2
cnceleb_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/cnceleb
threedspeaker_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/3dspeaker
ChildMandarin_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H
kingasr725_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid/King-ASR-725
combined_train_dir=${data}/combined_datasets_train

voxceleb1_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/voxceleb/vox1
kingasr612_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid/King-ASR-612
ChineseEnglishChildren_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children
speechocean762_raw_dir=/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762
combined_test_dir=${data}/combined_datasets_test_fixratio

# train data
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare wav.scp for musan, rirs, 3dspeaker, voxceleb2, cnceleb, ChildMandarin, kingasr725 datasets and mix adult and kid data"
  export LC_ALL=C # kaldi config

  # Create output directories
  mkdir -p ${data}/musan ${data}/rirs ${data}/3dspeaker ${data}/vox2 ${data}/cnceleb_train ${data}/ChildMandarin ${data}/kingasr725 ${combined_train_dir}
  
  # Prepare individual datasets
  
  # 1. Musan
  echo "Prepare wav.scp for musan datasets"
  find ${musan_raw_dir}/noise/free-sound -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
  
  # 2. RIRS
  echo "Prepare wav.scp for rirs datasets"
  awk '{print $5}' ${rirs_raw_dir}/RIRS_NOISES/real_rirs_isotropic_noises/rir_list | xargs -I {} echo {} ${rirs_raw_dir}/{} > ${data}/rirs/wav.scp
  
  # 3. VoxCeleb2
  echo "Prepare wav.scp utt2spk spk2utt for voxceleb2 datasets"
  find ${voxceleb2_raw_dir}/wav -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/vox2/wav.scp
  awk '{print $1}' ${data}/vox2/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vox2/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/vox2/utt2spk >${data}/vox2/spk2utt
  
  # 4. CN-Celeb
  echo "Prepare wav.scp utt2spk spk2utt for cnceleb datasets"
  for spk in `cat ${cnceleb_raw_dir}/CN-Celeb_flac/dev/dev.lst`; do
    find ${cnceleb_raw_dir}/CN-Celeb_wav/data/${spk} -name "*.wav" | \
      awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >> ${data}/cnceleb_train/wav.scp
  done
  for spk in `cat ${cnceleb_raw_dir}/CN-Celeb2_flac/spk.lst`; do
    find ${cnceleb_raw_dir}/CN-Celeb2_wav/data/${spk} -name "*.wav" | \
      awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >> ${data}/cnceleb_train/wav.scp
  done
  awk '{print $1}' ${data}/cnceleb_train/wav.scp | awk -F "/" '{print $0,$1}' > ${data}/cnceleb_train/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/cnceleb_train/utt2spk >${data}/cnceleb_train/spk2utt
  
  # 5. 3D-Speaker
  echo "Prepare wav.scp utt2spk spk2utt for 3dspeaker datasets"
  train_base_path=${data}/3dspeaker/train
  mkdir -p $train_base_path
  awk -v base_path="${threedspeaker_raw_dir}/" '{print $1" "base_path $2}' ${threedspeaker_raw_dir}/files/train_wav.scp > ${train_base_path}/all_wav.scp
  cp ${threedspeaker_raw_dir}/files/train_utt2info.csv ${train_base_path}/utt2info.csv
  cp ${threedspeaker_raw_dir}/files/train_utt2spk ${train_base_path}/all_utt2spk
  grep -v "Device09" ${train_base_path}/all_wav.scp > ${train_base_path}/wav.scp
  grep -v "Device09" ${train_base_path}/all_utt2spk > ${train_base_path}/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${train_base_path}/utt2spk > ${train_base_path}/spk2utt
  
  # 6. ChildMandarin
  echo "Prepare wav.scp utt2spk spk2utt for ChildMandarin datasets"
  find ${ChildMandarin_raw_dir} -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >${data}/ChildMandarin/wav.scp
  awk '{print $1}' ${data}/ChildMandarin/wav.scp | awk -F "/" '{print $0,$1}' >${data}/ChildMandarin/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/ChildMandarin/utt2spk >${data}/ChildMandarin/spk2utt
  
  # 7. King-ASR-725
  echo "Prepare wav.scp utt2spk spk2utt for kingasr725 datasets"
  find ${kingasr725_raw_dir}/DATA/CHANNEL0/WAVE -name "*.WAV" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/kingasr725/wav.scp
  awk '{print $1}' ${data}/kingasr725/wav.scp | awk -F "/" '{print $0,$1}' >${data}/kingasr725/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/kingasr725/utt2spk >${data}/kingasr725/spk2utt
  
  # Create combined dataset with unique dataset identifiers for speakers
  echo "Creating combined dataset with prefixed speaker IDs..."
  
  # Create empty files for combined dataset
  > ${combined_train_dir}/wav.scp
  > ${combined_train_dir}/utt2spk
  
  # Process voxceleb2 dataset and add prefix
  echo "Adding voxceleb2 to combined dataset with VOX2_ prefix..."
  awk '{sub(/\.wav$/,"",$1); print "VOX2_"$1, $2}' ${data}/vox2/wav.scp >> ${combined_train_dir}/wav.scp
  awk '{sub(/\.wav$/,"",$1); print "VOX2_"$1, "VOX2_"$2}' ${data}/vox2/utt2spk >> ${combined_train_dir}/utt2spk
  
  # Process cnceleb dataset and add prefix
  echo "Adding cnceleb to combined dataset with CNCE_ prefix..."
  awk '{sub(/\.wav$/,"",$1); print "CNCE_"$1, $2}' ${data}/cnceleb_train/wav.scp >> ${combined_train_dir}/wav.scp
  awk '{sub(/\.wav$/,"",$1); print "CNCE_"$1, "CNCE_"$2}' ${data}/cnceleb_train/utt2spk >> ${combined_train_dir}/utt2spk
  
  # Process 3dspeaker dataset and add prefix
  echo "Adding 3dspeaker to combined dataset with 3DSP_ prefix..."
  awk '{print "3DSP_"$1, $2}' ${train_base_path}/wav.scp >> ${combined_train_dir}/wav.scp
  awk '{print "3DSP_"$1, "3DSP_"$2}' ${train_base_path}/utt2spk >> ${combined_train_dir}/utt2spk
  
  # Process ChildMandarin dataset and add prefix
  echo "Adding ChildMandarin to combined dataset with CHMD_ prefix..."
  awk '{sub(/\.wav$/,"",$1); print "CHMD_"$1, $2}' ${data}/ChildMandarin/wav.scp >> ${combined_train_dir}/wav.scp
  awk '{sub(/\.wav$/,"",$1); print "CHMD_"$1, "CHMD_"$2}' ${data}/ChildMandarin/utt2spk >> ${combined_train_dir}/utt2spk
  
  # Process kingasr725 dataset and add prefix
  echo "Adding kingasr725 to combined dataset with KASR_ prefix..."
  awk '{sub(/\.WAV$/,"",$1); print "KASR_"$1, $2}' ${data}/kingasr725/wav.scp >> ${combined_train_dir}/wav.scp
  awk '{sub(/\.WAV$/,"",$1); print "KASR_"$1, "KASR_"$2}' ${data}/kingasr725/utt2spk >> ${combined_train_dir}/utt2spk
  
  # Generate spk2utt file for combined dataset
  echo "Generating spk2utt for combined dataset..."
  ./utils/utt2spk_to_spk2utt.pl ${combined_train_dir}/utt2spk > ${combined_train_dir}/spk2utt
  
  # Count speakers and utterances in the combined dataset
  spk_count=$(wc -l < ${combined_train_dir}/spk2utt)
  utt_count=$(wc -l < ${combined_train_dir}/utt2spk)
  echo "Combined dataset created with ${spk_count} speakers and ${utt_count} utterances"

  echo "Data Preparation Success !!!"
fi

# test data
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for voxceleb1, kingasr612, ChineseEnglishChildren, speechocean762 datasets"
  export LC_ALL=C # kaldi config

  # Create output directories
  mkdir -p ${data}/vox1 ${data}/kingasr612 ${data}/ChineseEnglishChildren ${data}/speechocean762 ${combined_test_dir}

  # 1. VoxCeleb1
  echo "Prepare wav.scp for voxceleb1 datasets"
  find ${voxceleb1_raw_dir}/wav -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/vox1/wav.scp
  awk '{print $1}' ${data}/vox1/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vox1/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/vox1/utt2spk >${data}/vox1/spk2utt

  # 2. King-ASR-612
  echo "Prepare wav.scp for kingasr612 datasets"
  find ${kingasr612_raw_dir}/DATA/CHANNEL0/WAVE -name "*.WAV" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/kingasr612/wav.scp
  awk '{print $1}' ${data}/kingasr612/wav.scp | awk -F "/" '{print $0,$1}' >${data}/kingasr612/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/kingasr612/utt2spk >${data}/kingasr612/spk2utt

  # 3. ChineseEnglishChildren
  echo "Prepare wav.scp for ChineseEnglishChildren datasets"
  find ${ChineseEnglishChildren_raw_dir}/WAV -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >${data}/ChineseEnglishChildren/wav.scp
  awk '{print $1}' ${data}/ChineseEnglishChildren/wav.scp | awk -F "/" '{print $0,$1}' >${data}/ChineseEnglishChildren/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/ChineseEnglishChildren/utt2spk >${data}/ChineseEnglishChildren/spk2utt

  # 4. speechocean762
  echo "Prepare wav.scp for speechocean762 datasets"
  find ${speechocean762_raw_dir}/WAVE -name "*.WAV" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >${data}/speechocean762/wav.scp
  awk '{print $1}' ${data}/speechocean762/wav.scp | awk -F "/" '{print $0,$1}' >${data}/speechocean762/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/speechocean762/utt2spk >${data}/speechocean762/spk2utt

  # Create combined dataset with unique dataset identifiers for speakers
  echo "Creating combined dataset with prefixed speaker IDs..."
  
  # Create empty files for combined dataset
  > ${combined_test_dir}/wav.scp
  > ${combined_test_dir}/utt2spk
  
  # Process voxceleb1 dataset and add prefix
  echo "Adding voxceleb1 to combined dataset with VOX1_ prefix..."
  awk '{sub(/\.wav$/,"",$1); print "VOX1_"$1, $2}' ${data}/vox1/wav.scp >> ${combined_test_dir}/wav.scp
  awk '{sub(/\.wav$/,"",$1); print "VOX1_"$1, "VOX1_"$2}' ${data}/vox1/utt2spk >> ${combined_test_dir}/utt2spk

  # Process kingasr612 dataset and add prefix
  echo "Adding kingasr612 to combined dataset with KASR_ prefix..."
  awk '{sub(/\.WAV$/,"",$1); print "KASR_"$1, $2}' ${data}/kingasr612/wav.scp >> ${combined_test_dir}/wav.scp
  awk '{sub(/\.WAV$/,"",$1); print "KASR_"$1, "KASR_"$2}' ${data}/kingasr612/utt2spk >> ${combined_test_dir}/utt2spk
  
  # Process ChineseEnglishChildren dataset and add prefix
  echo "Adding ChineseEnglishChildren to combined dataset with CEEC_ prefix..."
  awk '{sub(/\.wav$/,"",$1); print "CEEC_"$1, $2}' ${data}/ChineseEnglishChildren/wav.scp >> ${combined_test_dir}/wav.scp
  awk '{sub(/\.wav$/,"",$1); print "CEEC_"$1, "CEEC_"$2}' ${data}/ChineseEnglishChildren/utt2spk >> ${combined_test_dir}/utt2spk
  
  # Process speechocean762 dataset and add prefix
  echo "Adding speechocean762 to combined dataset with SPEE_ prefix..."
  awk '{sub(/\.wav$/,"",$1); print "SPEE_"$1, $2}' ${data}/speechocean762/wav.scp >> ${combined_test_dir}/wav.scp
  awk '{sub(/\.wav$/,"",$1); print "SPEE_"$1, "SPEE_"$2}' ${data}/speechocean762/utt2spk >> ${combined_test_dir}/utt2spk

  # Generate spk2utt file for combined dataset
  echo "Generating spk2utt for combined dataset..."
  ./utils/utt2spk_to_spk2utt.pl ${combined_test_dir}/utt2spk > ${combined_test_dir}/spk2utt

  # Count speakers and utterances in the combined dataset
  spk_count=$(wc -l < ${combined_test_dir}/spk2utt)
  utt_count=$(wc -l < ${combined_test_dir}/utt2spk)
  echo "Combined dataset created with ${spk_count} speakers and ${utt_count} utterances"

  # Create trial file
  echo "Creating trial file..."
  python ./local/create_trial.py --data-dir ${combined_test_dir} --output-trials ${combined_test_dir}/trials_mix_adult_kid

  echo "Data Preparation Success !!!"
fi