# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This is a speaker diarization inference script based on pretrained models.
Usages:
    1. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir]
    2. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token]
    3. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token] --nprocs [n]
"""

import os
import sys
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from scipy import optimize
import json
import time

import torch
import torch.multiprocessing as mp

try:
    from speakerlab.utils.config import Config
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(os.path.abspath(__file__)))
    from speakerlab.utils.config import Config

from speakerlab.utils.builder import build
from speakerlab.utils.utils import merge_vad, silent_print, download_model_from_modelscope, circle_pad
from speakerlab.utils.fileio import load_audio

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'
warnings.filterwarnings("ignore")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Delay pyannote import to avoid unnecessary heavy deps when overlap is disabled

parser = argparse.ArgumentParser(description='Speaker diarization inference.')
parser.add_argument('--wav', type=str, required=True, help='Input wavs')
parser.add_argument('--out_dir', type=str, required=True, help='Out results dir')
parser.add_argument('--out_type', choices=['rttm', 'json'], default='rttm', type=str, help='Results format, rttm or json')
parser.add_argument('--include_overlap', action='store_true', help='Include overlapping region')
parser.add_argument('--hf_access_token', type=str, help='hf_access_token for pyannote/segmentation-3.0 model. It\'s required if --include_overlap is specified')
parser.add_argument('--diable_progress_bar', action='store_true', help='Close the progress bar')
parser.add_argument('--nprocs', default=None, type=int, help='Num of procs')
parser.add_argument('--speaker_num', default=None, type=int, help='Oracle num of speaker')
parser.add_argument('--no_chunk_after_vad', action='store_true', 
                    help='Diarization mode: if set, extract one embedding per VAD segment (whole-segment mode); '
                         'if not set, split VAD segments into fixed-size subsegments with sliding window (sliding-window mode, default)')
parser.add_argument('--vad_min_speech_ms', default=200.0, type=float, help='VAD post-process: minimum speech segment duration (milliseconds)')
parser.add_argument('--vad_max_silence_ms', default=300.0, type=float, help='VAD post-process: maximum silence gap to fill (milliseconds)')
parser.add_argument('--vad_energy_threshold', default=0.05, type=float, help='VAD energy threshold for boundary refinement')
parser.add_argument('--vad_boundary_expansion_ms', default=10.0, type=float, help='VAD boundary expansion (milliseconds)')
parser.add_argument('--vad_boundary_energy_percentile', default=10.0, type=float, help='VAD boundary energy percentile')
parser.add_argument('--vad_threshold', default=0.5, type=float, help='VAD threshold for TenVad (default: 0.5)')
parser.add_argument('--cluster_mer_cos', default=0.3, type=float, help='Clustering merge cosine threshold (default: 0.3)')
parser.add_argument('--cluster_fix_cos_thr', default=0.3, type=float, help='Clustering fixed cosine threshold (default: 0.3)')
parser.add_argument('--cluster_min_cluster_size', default=0, type=int, help='Clustering minimum cluster size (default: 0)')
parser.add_argument('--chunk_dur', default=1.5, type=float, help='Chunk duration in seconds for sliding window mode (default: 1.5)')
parser.add_argument('--chunk_step', default=0.75, type=float, help='Chunk step size in seconds for sliding window mode (default: 0.75)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for embedding extraction (default: 64)')


def get_speaker_embedding_model(device:torch.device = None, cache_dir:str = None):
    conf = {
        'model_id': 'iic/speech_eres2netv2_sv_zh-cn_16k-common',
        'revision': 'v1.0.1',
        'model_ckpt': 'pretrained_eres2netv2.ckpt',
        'embedding_model': {
            'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'feature_extractor': {
            'obj': 'speakerlab.process.processor.FBank',
            'args': {
                'n_mels': 80,
                'sample_rate': 16000,
                'mean_nor': True,
                },
        }
    }

    cache_dir = download_model_from_modelscope(conf['model_id'], conf['revision'], cache_dir)
    pretrained_model_path = os.path.join(cache_dir, conf['model_ckpt'])
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    # load pretrained model
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    if device is not None:
        embedding_model.to(device)
    return embedding_model,  feature_extractor

def get_cluster_backend(mer_cos=0.3, fix_cos_thr=0.3, min_cluster_size=0):
    conf = {
        'cluster':{
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args':{
                'cluster_type': 'AHC',
                'mer_cos': mer_cos,
                'min_cluster_size': min_cluster_size,
                'fix_cos_thr': fix_cos_thr,
            }
        }
    }
    config = Config(conf)
    return build('cluster', config)

def get_voice_activity_detection_model(device: torch.device=None, cache_dir:str = None, threshold: float=0.5):
    """
    Use TenVad (as in dataset) to perform VAD locally. Return callable that
    takes a 1-D waveform tensor/ndarray and outputs [{'value': [[st_ms, ed_ms], ...]}].
    """
    try:
        from ten_vad import TenVad
    except ImportError:
        try:
            sys.path.append('/root/code/gitlab_repos/se_train')
            from ten_vad import TenVad  # type: ignore
        except Exception as e:
            raise ImportError('ten_vad is required for VAD. Please install/ensure it is available.') from e

    class TenVadWrapper:
        def __init__(self, sample_rate: int = 16000, frame_ms: float = 16.0, threshold: float = 0.5):
            self.sample_rate = sample_rate
            self.hop_size = int(frame_ms * sample_rate / 1000)
            self.engine = TenVad(self.hop_size, threshold)

        def __call__(self, wav_1d):
            import numpy as _np
            # to numpy float32 in [-1, 1]
            if hasattr(wav_1d, 'detach'):
                x = wav_1d.detach().cpu().numpy().astype(_np.float32)
            else:
                x = _np.asarray(wav_1d).astype(_np.float32)
            if x.size == 0:
                return [], x
            x = _np.clip(x, -1.0, 1.0)
            x_i16 = (x * 32767).astype(_np.int16)

            num_frames = len(x_i16) // self.hop_size
            flags = []
            for i in range(num_frames):
                frame = x_i16[i*self.hop_size:(i+1)*self.hop_size]
                if len(frame) == self.hop_size:
                    _, f = self.engine.process(frame)
                    flags.append(int(f))
                else:
                    flags.append(0)
            
            # Return raw flags for post-processing
            return flags, x

    # We use 16ms hop by default to match dataset settings
    return TenVadWrapper(sample_rate=16000, frame_ms=16.0, threshold=threshold)

def get_segmentation_model(use_auth_token, device: torch.device=None):
    from pyannote.audio import Inference, Model
    segmentation_params = {
        'segmentation':'pyannote/segmentation-3.0',
        'segmentation_batch_size':32,
        'use_auth_token':use_auth_token,
        }
    model = Model.from_pretrained(
        segmentation_params['segmentation'], 
        use_auth_token=segmentation_params['use_auth_token'], 
        strict=False,
        )
    segmentation = Inference(
        model,
        duration=model.specifications.duration,
        step=0.1 * model.specifications.duration,
        skip_aggregation=True,
        batch_size=segmentation_params['segmentation_batch_size'],
        device = device,
        )
    return segmentation


class Diarization3Dspeaker():
    """
    This class is designed to handle the speaker diarization process, 
    which involves identifying and segmenting audio by speaker identities. 
    Args:
        device (str, default=None): The device on which models will run. 
        include_overlap (bool, default=False): Indicates whether to include overlapping 
            speech segments in the diarization output. Overlapping speech occurs when multiple 
            speakers are talking simultaneously.
        hf_access_token (str, default=None): Access token for Hugging Face, required if 
            include_overlap is True. This token allows access to pynnote segmentation models 
            available on the Hugging Face that handles overlapping speech.
        speaker_num (int, default=None): Specify number of speakers.
        model_cache_dir (str, default=None): If specified, the pretrained model will be downloaded 
            to this directory; only pretrained from modelscope are supported.
    Usage:
        diarization_pipeline = Diarization3Dspeaker(device, include_overlap, hf_access_token)
        output = diarization_pipeline(input_audio) # input_audio can be a path to a WAV file, a NumPy array, or a PyTorch tensor
        print(output) # output: [[1.1, 2.2, 0], [3.1, 4.1, 1], ..., [st_n, ed_n, speaker_id]]
        diarization_pipeline.save_diar_output('audio.rttm') # or audio.json
    """
    def __init__(self, device=None, include_overlap=False, hf_access_token=None, speaker_num=None, model_cache_dir=None,
                 no_chunk_after_vad: bool=False,
                 vad_min_speech_ms: float=None, vad_max_silence_ms: float=None,
                 vad_energy_threshold: float=None, vad_boundary_expansion_ms: float=None,
                 vad_boundary_energy_percentile: float=None,
                 vad_threshold: float=0.5,
                 cluster_mer_cos: float=0.3, cluster_fix_cos_thr: float=0.3, cluster_min_cluster_size: int=0,
                 chunk_dur: float=1.5, chunk_step: float=0.75,
                 batch_size: int=64):
        if include_overlap and hf_access_token is None:
            raise ValueError("hf_access_token is required when include_overlap is True.")
 
        self.device = self.normalize_device(device)
        self.include_overlap = include_overlap

        self.embedding_model, self.feature_extractor = get_speaker_embedding_model(self.device, model_cache_dir)
        self.vad_model = get_voice_activity_detection_model(self.device, model_cache_dir, threshold=vad_threshold)
        self.cluster = get_cluster_backend(mer_cos=cluster_mer_cos, fix_cos_thr=cluster_fix_cos_thr, min_cluster_size=cluster_min_cluster_size)

        if include_overlap:
            self.segmentation_model = get_segmentation_model(hf_access_token, self.device)
        
        self.batchsize = batch_size
        self.chunk_dur = chunk_dur
        self.chunk_step = chunk_step
        self.fs = self.feature_extractor.sample_rate
        self.output_field_labels = None
        self.speaker_num = speaker_num
        self.no_chunk_after_vad = no_chunk_after_vad
        self.last_vad_time = None
        self.last_vad_time_raw = None
        self.last_vad_time_processed = None
        self.last_vad_masked_audio = None
        self.last_vad_refined_mask = None
        self.last_vad_processed_mask = None
        self.last_vad_processed_mask = None
        # VAD post-processing parameters
        self.vad_frame_size_ms = 16.0
        self.vad_min_speech_ms = float(vad_min_speech_ms) if vad_min_speech_ms is not None else 200.0
        self.vad_max_silence_ms = float(vad_max_silence_ms) if vad_max_silence_ms is not None else 300.0
        self.vad_energy_threshold = float(vad_energy_threshold) if vad_energy_threshold is not None else 0.05
        self.vad_boundary_expansion_ms = float(vad_boundary_expansion_ms) if vad_boundary_expansion_ms is not None else 10.0
        self.vad_boundary_energy_percentile = float(vad_boundary_energy_percentile) if vad_boundary_energy_percentile is not None else 10.0

    def __call__(self, wav, wav_fs=None, speaker_num=None):
        wav_data = load_audio(wav, wav_fs, self.fs)

        # stage 1-1: do vad (raw)
        speech_flags, wav_data_for_vad = self.do_vad(wav_data)
        # stage 1-1.5: VAD post-processing (returns processed_mask, refined_mask and vad_time)
        vad_processed_mask, vad_refined_mask, vad_time = self.postprocess_vad(speech_flags, wav_data_for_vad)
        # Keep raw flags for visualization (convert to intervals)
        vad_time_raw = self._flags_to_intervals(speech_flags, wav_data_for_vad)
        
        # Save masks for later use
        self.last_vad_processed_mask = vad_processed_mask
        self.last_vad_refined_mask = vad_refined_mask
        
        if self.include_overlap:
            # stage 1-2: do segmentation
            segmentations, count = self.do_segmentation(wav_data)
            valid_field = get_valid_field(count)
            vad_time = merge_vad(vad_time, valid_field)
            # Note: After merge_vad, we need to update refined_mask based on new vad_time
            # For now, we use vad_time to create chunks, which is derived from refined_mask

        # stage 2: prepare segments for embedding extraction
        # Use vad_time which is derived from refined_mask, so chunks correspond to refined_mask regions
        if self.no_chunk_after_vad:
            # one embedding per VAD segment
            chunks = [[st, ed] for (st, ed) in vad_time]
        else:
            # default behavior: split into fixed-size subsegments
            chunks = [c for (st, ed) in vad_time for c in self.chunk(st, ed)]

        # keep VAD for potential visualization
        self.last_vad_time_raw = vad_time_raw
        # Convert processed_mask to intervals for visualization
        self.last_vad_time_processed = self._mask_to_intervals(vad_processed_mask) if vad_processed_mask is not None else []
        self.last_vad_time = vad_time  # This is already from refined_mask
        
        # Generate VAD masked audio (mask out non-VAD regions using refined_mask)
        # Use refined_mask directly to ensure consistency
        self.last_vad_masked_audio = self._apply_vad_mask_from_mask(wav_data, vad_refined_mask)

        # if no valid chunks after filtering, return empty result
        if len(chunks) == 0:
            self.output_field_labels = []
            return []

        # stage 3: extract embeddings
        embeddings = self.do_emb_extraction(chunks, wav_data)

        # stage 4: clustering
        speaker_num, output_field_labels = self.do_clustering(chunks, embeddings, speaker_num)

        if self.include_overlap:
            # stage 5: include overlap results
            binary = self.post_process(output_field_labels, speaker_num, segmentations, count)
            timestamps = [count.sliding_window[i].middle for i in range(binary.shape[0])]
            output_field_labels = self.binary_to_segs(binary, timestamps)

        self.output_field_labels = output_field_labels
        return output_field_labels

    def do_vad(self, wav):
        # wav: [1, T]
        speech_flags, wav_data = self.vad_model(wav[0])
        return speech_flags, wav_data

    def postprocess_vad(self, speech_flags, wav_data):
        """
        Apply VAD post-processing using _post_process_speech_flags and _refine_vad_boundaries_with_energy
        Returns processed_mask, refined_mask, and vad_time intervals
        """
        import numpy as np
        
        # Convert flags to processed flags
        processed_flags = self._post_process_speech_flags(speech_flags)
        
        # Convert processed flags to mask (processed_mask)
        hop_size = int(self.vad_frame_size_ms * self.fs / 1000)
        processed_mask = np.zeros(len(wav_data), dtype=np.float32)
        for i, flag in enumerate(processed_flags):
            s = i * hop_size
            e = min((i + 1) * hop_size, len(wav_data))
            processed_mask[s:e] = flag
        
        # Refine boundaries with energy (refined_mask)
        refined_mask = self._refine_vad_boundaries_with_energy(wav_data, processed_mask)
        
        # Convert mask to time intervals
        vad_time = self._mask_to_intervals(refined_mask)
        return processed_mask, refined_mask, vad_time

    def _post_process_speech_flags(self, flags):
        """
        Smooth + morphological fill (simple implementation)
        Similar to monaural_dataset.py
        """
        import numpy as np
        flags = np.array(flags, dtype=np.float32)
        
        # Simple moving average smoothing
        win = 3
        pad = np.pad(flags, (win // 2, win // 2), mode='edge')
        smooth = np.convolve(pad, np.ones(win) / win, mode='valid')
        smooth = (smooth > 0.5).astype(np.float32)

        # Minimum speech segment / maximum silence segment constraints (frame-based)
        min_speech_frames = max(1, int(self.vad_min_speech_ms / self.vad_frame_size_ms))
        max_silence_frames = max(1, int(self.vad_max_silence_ms / self.vad_frame_size_ms))

        res = smooth.copy()
        # Fill short silence gaps
        count0 = 0
        for i in range(len(res)):
            if res[i] == 0:
                count0 += 1
            else:
                if 0 < count0 <= max_silence_frames:
                    res[i - count0 : i] = 1
                count0 = 0
        # Remove too-short speech segments
        count1 = 0
        for i in range(len(res)):
            if res[i] == 1:
                count1 += 1
            else:
                if 0 < count1 < min_speech_frames:
                    res[i - count1 : i] = 0
                count1 = 0
        return res.astype(np.float32)

    def _refine_vad_boundaries_with_energy(self, audio_data, vad_mask):
        """
        Refine VAD boundaries using energy-based method
        Similar to monaural_dataset.py
        """
        import numpy as np
        refined_mask = vad_mask.copy()
        window_size = int(0.02 * self.fs)  # 20ms
        hop_length = int(0.01 * self.fs)   # 10ms
        n_frames = (len(audio_data) - window_size) // hop_length + 1
        if n_frames <= 0:
            return refined_mask

        frame_energy = np.zeros(len(audio_data), dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_length
            e = min(s + window_size, len(audio_data))
            en = float(np.mean(audio_data[s:e] ** 2))
            frame_energy[s:e] = max(frame_energy[s:e].max(), en)

        vad_diff = np.diff(np.concatenate(([0], vad_mask, [0])))
        speech_starts = np.where(vad_diff > 0)[0]
        speech_ends = np.where(vad_diff < 0)[0]
        if len(speech_starts) == 0 or len(speech_ends) == 0:
            return refined_mask

        lookahead_frames = 10
        lookahead_samples = lookahead_frames * hop_length
        energy_floor = float(self.vad_energy_threshold)
        energy_percentile = float(self.vad_boundary_energy_percentile)
        boundary_expand_ms = float(self.vad_boundary_expansion_ms)
        boundary_expand_samples = int(boundary_expand_ms * self.fs / 1000.0)

        for start, end in zip(speech_starts, speech_ends):
            seg_energy = frame_energy[start:end]
            if len(seg_energy) == 0:
                continue
            dynamic_th = max(np.percentile(seg_energy, energy_percentile), energy_floor)
            
            # Step 1: Forward contraction - find new start after removing low energy
            new_start = start
            for i in range(start, min(end, start + lookahead_samples)):
                if frame_energy[i] < dynamic_th:
                    refined_mask[start:i] = 0
                    new_start = i  # Update new start position
                    break
            
            # Step 2: Backward contraction - find new end after removing low energy
            new_end = end
            for i in range(end - 1, max(new_start, end - lookahead_samples), -1):
                if frame_energy[i] < dynamic_th:
                    refined_mask[i:end] = 0
                    new_end = i + 1  # Update new end position (i+1 because range is exclusive)
                    break
            
            # Step 3: Expand boundaries from the contracted positions
            # Critical: expansion must be within original segment boundaries to prevent creating new segments
            # Expansion fills gaps created by contraction to ensure continuity
            if boundary_expand_samples > 0:
                # Expand start: extend backward from new_start, but stay within original segment
                # This fills any gap [start:new_start] created by contraction
                expand_start_begin = max(start, new_start - boundary_expand_samples)
                expand_start_end = new_start
                refined_mask[expand_start_begin:expand_start_end] = 1
                
                # Expand end: extend forward from new_end, but stay within original segment
                # This fills any gap [new_end:end] created by contraction
                expand_end_begin = new_end
                # Important: expand up to original end to fill any contraction gap, but not beyond
                expand_end_end = end  # Always fill to original end to ensure continuity
                refined_mask[expand_end_begin:expand_end_end] = 1
        return refined_mask.astype(np.float32)

    def _mask_to_intervals(self, mask):
        """
        Convert VAD mask to time intervals in seconds
        """
        import numpy as np
        if len(mask) == 0:
            return []
        
        # Find transitions
        diff = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.where(diff > 0)[0]
        ends = np.where(diff < 0)[0]
        
        if len(starts) == 0:
            return []
        
        intervals = []
        for s, e in zip(starts, ends):
            start_sec = float(s) / self.fs
            end_sec = float(e) / self.fs
            if end_sec > start_sec:
                intervals.append([start_sec, end_sec])
        
        return intervals

    def _flags_to_intervals(self, flags, wav_data):
        """
        Convert raw speech flags to time intervals in seconds
        """
        import numpy as np
        if len(flags) == 0:
            return []
        
        hop_size = int(self.vad_frame_size_ms * self.fs / 1000)
        intervals = []
        i = 0
        N = len(flags)
        while i < N:
            if flags[i]:
                j = i + 1
                while j < N and flags[j]:
                    j += 1
                start_sec = float(i * hop_size) / self.fs
                end_sec = float(min(j * hop_size, len(wav_data))) / self.fs
                if end_sec > start_sec:
                    intervals.append([start_sec, end_sec])
                i = j
            else:
                i += 1
        
        return intervals

    def _apply_vad_mask(self, wav_data, vad_time):
        """
        Apply VAD mask to audio: set non-VAD regions to zero
        Args:
            wav_data: audio data [1, T] or [T]
            vad_time: list of [start_sec, end_sec] intervals
        Returns:
            masked audio data with same shape as input
        """
        import numpy as np
        
        # Handle different input shapes
        if wav_data.ndim == 2:
            audio = wav_data[0].copy()
        else:
            audio = wav_data.copy()
        
        # Create mask (all zeros initially)
        mask = np.zeros(len(audio), dtype=np.float32)
        
        # Set VAD regions to 1
        for start_sec, end_sec in vad_time:
            start_sample = int(start_sec * self.fs)
            end_sample = int(end_sec * self.fs)
            start_sample = max(0, min(start_sample, len(audio)))
            end_sample = max(0, min(end_sample, len(audio)))
            if end_sample > start_sample:
                mask[start_sample:end_sample] = 1.0
        
        # Apply mask
        masked_audio = audio * mask
        
        # Return in same shape as input
        if wav_data.ndim == 2:
            return masked_audio.reshape(1, -1)
        else:
            return masked_audio

    def _apply_vad_mask_from_mask(self, wav_data, vad_mask):
        """
        Apply VAD mask to audio directly from mask array: set non-VAD regions to zero
        Args:
            wav_data: audio data [1, T] or [T] (can be Tensor or numpy array)
            vad_mask: mask array [T] with 1.0 for VAD regions, 0.0 for non-VAD
        Returns:
            masked audio data with same shape as input
        """
        import numpy as np
        
        # Convert Tensor to numpy if needed
        if hasattr(wav_data, 'detach'):
            # PyTorch Tensor
            wav_np = wav_data.detach().cpu().numpy()
        else:
            # numpy array
            wav_np = np.asarray(wav_data)
        
        # Handle different input shapes
        if wav_np.ndim == 2:
            audio = wav_np[0].copy()
        else:
            audio = wav_np.copy()
        
        # Ensure mask length matches audio length
        mask_len = min(len(vad_mask), len(audio))
        mask = vad_mask[:mask_len].copy()
        
        # Pad mask if needed
        if len(mask) < len(audio):
            mask = np.pad(mask, (0, len(audio) - len(mask)), mode='constant', constant_values=0.0)
        
        # Apply mask
        masked_audio = audio * mask
        
        # Return in same shape as input
        if wav_np.ndim == 2:
            return masked_audio.reshape(1, -1)
        else:
            return masked_audio

    def do_segmentation(self, wav):
        from pyannote.audio import Inference as _Inference
        segmentations = self.segmentation_model({'waveform':wav, 'sample_rate': self.fs})
        frame_windows = self.segmentation_model.model.receptive_field

        count = _Inference.aggregate(
            np.sum(segmentations, axis=-1, keepdims=True),
            frame_windows,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)
        return segmentations, count

    def chunk(self, st, ed):
        chunks = []
        if ed - st <= 0:
            return chunks
        subseg_st = st
        made = False
        while subseg_st + self.chunk_dur < ed + self.chunk_step:
            subseg_ed = min(subseg_st + self.chunk_dur, ed)
            chunks.append([subseg_st, subseg_ed])
            subseg_st += self.chunk_step
            made = True
        if not made:
            chunks.append([st, ed])
        return chunks

    def do_emb_extraction(self, chunks, wav):
        # chunks: [[st1, ed1]...]
        # wav: [1, T]
        wavs = [wav[0, int(st*self.fs):int(ed*self.fs)] for st, ed in chunks]
        max_len = max([x.shape[0] for x in wavs])
        wavs = [circle_pad(x, max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)

        embeddings = []
        batch_st = 0
        with torch.no_grad():
            while batch_st < len(chunks):
                wavs_batch = wavs[batch_st: batch_st+self.batchsize].to(self.device)
                feats_batch = torch.vmap(self.feature_extractor)(wavs_batch)
                embeddings_batch = self.embedding_model(feats_batch).cpu()
                embeddings.append(embeddings_batch)
                batch_st += self.batchsize
        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings

    def do_clustering(self, chunks, embeddings, speaker_num=None):
        cluster_labels = self.cluster(
            embeddings, 
            speaker_num = speaker_num if speaker_num is not None else self.speaker_num
        )
        speaker_num = cluster_labels.max()+1
        output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
        output_field_labels = compressed_seg(output_field_labels)
        return speaker_num, output_field_labels

    def post_process(self, output_field_labels, speaker_num, segmentations, count):
        num_frames = len(count)
        cluster_frames = np.zeros((num_frames, speaker_num))
        frame_windows = count.sliding_window
        for i in output_field_labels:
            cluster_frames[frame_windows.closest_frame(i[0]+frame_windows.duration/2)\
                :frame_windows.closest_frame(i[1]+frame_windows.duration/2)\
                    , i[2]] = 1.0

        activations = np.zeros((num_frames, speaker_num))
        num_chunks, num_frames_per_chunk, num_classes = segmentations.data.shape
        for i, (c, data) in enumerate(segmentations):
            # data: [num_frames_per_chunk, num_classes]
            # chunk_cluster_frames: [num_frames_per_chunk, speaker_num]
            start_frame = frame_windows.closest_frame(c.start+frame_windows.duration/2)
            end_frame = start_frame + num_frames_per_chunk
            chunk_cluster_frames = cluster_frames[start_frame:end_frame]
            align_chunk_cluster_frames = np.zeros((num_frames_per_chunk, speaker_num))

            # assign label to each dimension of "data" according to number of 
            # overlap frames between "data" and "chunk_cluster_frames"
            cost_matrix = []
            for j in range(num_classes):
                if sum(data[:, j])>0:
                    num_of_overlap_frames = [(data[:, j].astype('int') & d.astype('int')).sum() \
                        for d in chunk_cluster_frames.T]
                else:
                    num_of_overlap_frames = [-1]*speaker_num
                cost_matrix.append(num_of_overlap_frames)
            cost_matrix = np.array(cost_matrix) # (num_classes, speaker_num)
            row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
            for j in range(len(row_index)):
                r = row_index[j]
                c = col_index[j]
                if cost_matrix[r, c] > 0:
                    align_chunk_cluster_frames[:, c] = np.maximum(
                            data[:, r], align_chunk_cluster_frames[:, c]
                            )
            activations[start_frame:end_frame] += align_chunk_cluster_frames

        # correct activations according to count_data
        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations)
        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            cur_max_spk_num = min(speaker_num, c.item())
            for i in range(cur_max_spk_num):
                if activations[t, speakers[i]] > 0:
                    binary[t, speakers[i]] = 1.0

        supplement_field = (binary.sum(-1)==0) & (cluster_frames.sum(-1)!=0)
        binary[supplement_field] = cluster_frames[supplement_field]
        return binary

    def binary_to_segs(self, binary, timestamps, threshold=0.5):
        output_field_labels = []
        # binary: [num_frames, num_classes]
        # timestamps: [T_1, ..., T_num_frames]        
        for k, k_scores in enumerate(binary.T):
            start = timestamps[0]
            is_active = k_scores[0] > threshold

            for t, y in zip(timestamps[1:], k_scores[1:]):
                if is_active:
                    if y < threshold:
                        output_field_labels.append([round(start, 3), round(t, 3), k])
                        start = t
                        is_active = False
                else:
                    if y > threshold:
                        start = t
                        is_active = True

            if is_active:
                output_field_labels.append([round(start, 3), round(t, 3), k])
        return sorted(output_field_labels, key=lambda x : x[0])

    def save_diar_output(self, out_file, wav_id=None, output_field_labels=None):
        if output_field_labels is None and self.output_field_labels is None:
            raise ValueError('No results can be saved.')
        if output_field_labels is None:
            output_field_labels = self.output_field_labels

        wav_id = 'default' if wav_id is None else wav_id
        if out_file.endswith('rttm'):
            line_str ="SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
            with open(out_file, 'w') as f:
                for seg in output_field_labels:
                    seg_st, seg_ed, cluster_id = seg
                    f.write(line_str.format(wav_id, seg_st, seg_ed-seg_st, cluster_id))
        elif out_file.endswith('json'):
            out_json = {}
            for seg in output_field_labels:
                seg_st, seg_ed, cluster_id = seg
                item = {
                    'start': seg_st,
                    'stop': seg_ed,
                    'speaker': cluster_id,
                }
                segid = wav_id+'_'+str(round(seg_st, 3))+\
                    '_'+str(round(seg_ed, 3))
                out_json[segid] = item
            with open(out_file, mode='w') as f:
                json.dump(out_json, f, indent=2)
        else:
            raise ValueError('The supported output file formats are currently limited to RTTM and JSON.')

    def normalize_device(self, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            assert isinstance(device, torch.device)
        return device

def get_valid_field(count):
    valid_field = []
    start = None
    for i, (c, data) in enumerate(count):
        if data.item()==0 or i==len(count)-1:
            if start is not None:
                end = c.middle
                valid_field.append([start, end])
                start = None
        else:
            if start is None:
                start = c.middle
    return valid_field

def compressed_seg(seg_list):
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
        if i == 0:
            new_seg_list.append([seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][2]:
            if seg_st > new_seg_list[-1][1]:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][1] = seg_ed
        else:
            if seg_st < new_seg_list[-1][1]:
                p = (new_seg_list[-1][1]+seg_st) / 2
                new_seg_list[-1][1] = p
                seg_st = p
            new_seg_list.append([seg_st, seg_ed, cluster_id])
    return new_seg_list

def _save_vad_waveform_png(wav_path, fs, vad_time_raw, vad_time_processed, vad_time_refined, out_png):
    """
    Save a PNG showing waveform with raw, processed, and refined VAD-active intervals shaded.
    """
    try:
        import numpy as _np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        from speakerlab.utils.fileio import load_audio as _load_audio

        wav = _load_audio(wav_path, None, fs)
        if hasattr(wav, 'detach'):
            y = wav.detach().cpu().numpy()
        else:
            y = _np.asarray(wav)
        y = y[0] if y.ndim > 1 else y
        if y.size == 0:
            return
        t = _np.arange(y.shape[0], dtype=_np.float32) / float(fs)

        fig, axes = _plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # Top: RAW VAD
        ax = axes[0]
        ax.plot(t, y, color='#1f77b4', linewidth=0.5)
        for st, ed in (vad_time_raw or []):
            try:
                ax.axvspan(float(st), float(ed), color='crimson', alpha=0.25, label='Raw VAD')
            except Exception:
                continue
        ax.set_xlim(0, t[-1])
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform + Raw VAD')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Middle: PROCESSED VAD (after _post_process_speech_flags)
        ax2 = axes[1]
        ax2.plot(t, y, color='#1f77b4', linewidth=0.5)
        for st, ed in (vad_time_processed or []):
            try:
                ax2.axvspan(float(st), float(ed), color='orange', alpha=0.3, label='Processed VAD')
            except Exception:
                continue
        ax2.set_xlim(0, t[-1])
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Waveform + Processed VAD (after smoothing & morphological fill)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Bottom: REFINED VAD (after _refine_vad_boundaries_with_energy)
        ax3 = axes[2]
        ax3.plot(t, y, color='#1f77b4', linewidth=0.5)
        for st, ed in (vad_time_refined or []):
            try:
                ax3.axvspan(float(st), float(ed), color='green', alpha=0.3, label='Refined VAD')
            except Exception:
                continue
        ax3.set_xlim(0, t[-1])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Waveform + Refined VAD (after energy-based boundary refinement)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        _plt.close(fig)
    except Exception:
        # best-effort; skip plotting if any issue (e.g., matplotlib missing)
        pass

def _merge_and_filter_intervals(intervals, min_len_sec, max_gap_sec):
    """
    Filter out intervals shorter than min_len_sec, then merge intervals where
    the gap between consecutive intervals is <= max_gap_sec.
    """
    if not intervals:
        return []
    # sort and filter
    sorted_ints = sorted([[float(s), float(e)] for s, e in intervals if (e - s) > 0], key=lambda x: (x[0], x[1]))
    filtered = [[s, e] for s, e in sorted_ints if (e - s) >= float(min_len_sec)]
    if not filtered:
        return []
    # merge small gaps
    merged = []
    cur_s, cur_e = filtered[0]
    for s, e in filtered[1:]:
        gap = float(s) - float(cur_e)
        if gap <= float(max_gap_sec):
            cur_e = max(cur_e, e)
        else:
            merged.append([cur_s, cur_e])
            cur_s, cur_e = s, e
    merged.append([cur_s, cur_e])
    return merged

def main_process(rank, nprocs, args, wav_list):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        ngpus = torch.cuda.device_count()
        device = torch.device('cuda:%d'%(rank%ngpus))
    diarization = Diarization3Dspeaker(
        device,
        args.include_overlap,
        args.hf_access_token,
        args.speaker_num,
        None,
        args.no_chunk_after_vad,
        args.vad_min_speech_ms,
        args.vad_max_silence_ms,
        args.vad_energy_threshold,
        args.vad_boundary_expansion_ms,
        args.vad_boundary_energy_percentile,
        args.vad_threshold,
        args.cluster_mer_cos,
        args.cluster_fix_cos_thr,
        args.cluster_min_cluster_size,
        args.chunk_dur,
        args.chunk_step,
        args.batch_size,
    )
    
    wav_list = wav_list[rank::nprocs]
    if rank == 0 and (not args.diable_progress_bar):
        wav_list = tqdm(wav_list, desc=f"Rank 0 processing")
    for wav_path in wav_list:
        t0 = time.time()
        ouput = diarization(wav_path)
        elapsed = time.time() - t0

        # write diarization output
        wav_id = os.path.basename(wav_path).rsplit('.', 1)[0]
        if args.out_dir is not None:
            out_file = os.path.join(args.out_dir, wav_id+'.%s'%args.out_type)
        else:
            out_file = '%s.%s'%(wav_path.rsplit('.', 1)[0], args.out_type)
        diarization.save_diar_output(out_file, wav_id)

        # save waveform + VAD visualization
        try:
            png_path = os.path.join(os.path.dirname(out_file), f"{wav_id}.vad.png")
            _save_vad_waveform_png(
                wav_path,
                diarization.fs,
                diarization.last_vad_time_raw or [],
                diarization.last_vad_time_processed or [],
                diarization.last_vad_time or [],
                png_path
            )
        except Exception:
            pass

        # save VAD masked audio
        try:
            if diarization.last_vad_masked_audio is not None:
                masked_audio_path = os.path.join(os.path.dirname(out_file), f"{wav_id}.vad_masked.wav")
                import soundfile as sf
                masked_audio = diarization.last_vad_masked_audio
                # Handle different shapes
                if masked_audio.ndim == 2:
                    masked_audio = masked_audio[0]
                sf.write(masked_audio_path, masked_audio, diarization.fs)
        except Exception:
            pass

        # save VAD info (raw, processed, refined)
        try:
            vad_info = {
                'wav_path': wav_path,
                'sample_rate': diarization.fs,
                'vad_raw': {
                    'intervals': [[float(st), float(ed)] for st, ed in (diarization.last_vad_time_raw or [])],
                    'num_segments': len(diarization.last_vad_time_raw or []),
                    'total_duration': sum([float(ed) - float(st) for st, ed in (diarization.last_vad_time_raw or [])])
                },
                'vad_processed': {
                    'intervals': [[float(st), float(ed)] for st, ed in (diarization.last_vad_time_processed or [])],
                    'num_segments': len(diarization.last_vad_time_processed or []),
                    'total_duration': sum([float(ed) - float(st) for st, ed in (diarization.last_vad_time_processed or [])])
                },
                'vad_refined': {
                    'intervals': [[float(st), float(ed)] for st, ed in (diarization.last_vad_time or [])],
                    'num_segments': len(diarization.last_vad_time or []),
                    'total_duration': sum([float(ed) - float(st) for st, ed in (diarization.last_vad_time or [])])
                }
            }
            vad_info_file = os.path.join(os.path.dirname(out_file), f"{wav_id}.vad_info.json")
            with open(vad_info_file, 'w') as vf:
                json.dump(vad_info, vf, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # write sidecar meta: duration, processing time, rtf
        def _get_duration_seconds(path):
            try:
                import soundfile as sf  # type: ignore
                info = sf.info(path)
                if info.samplerate > 0:
                    return float(info.frames) / float(info.samplerate)
            except Exception:
                pass
            try:
                import wave
                with wave.open(path, 'rb') as w:
                    frames = w.getnframes()
                    rate = w.getframerate() or 0
                    return (float(frames) / float(rate)) if rate > 0 else None
            except Exception:
                pass
            return None

        duration_sec = _get_duration_seconds(wav_path)
        # compute pairwise cosine similarities between diarized segments
        pairwise_list = []
        pair_min = None
        pair_mean = None
        try:
            segs = diarization.output_field_labels or []
            seg_times = [[float(s[0]), float(s[1])] for s in segs]
            if len(seg_times) >= 2:
                wav_full = load_audio(wav_path, None, diarization.fs)
                embs = diarization.do_emb_extraction(seg_times, wav_full)
                if embs.shape[0] >= 2:
                    import numpy as _np
                    Z = embs / (_np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
                    S = Z @ Z.T
                    triu_idx = _np.triu_indices(S.shape[0], k=1)
                    vals = S[triu_idx]
                    if vals.size > 0:
                        pair_min = float(vals.min())
                        pair_mean = float(vals.mean())
                    for i in range(S.shape[0]):
                        for j in range(i+1, S.shape[0]):
                            pairwise_list.append({
                                'i': int(i),
                                'j': int(j),
                                'seg_i': {'start': float(segs[i][0]), 'stop': float(segs[i][1]), 'speaker': int(segs[i][2])},
                                'seg_j': {'start': float(segs[j][0]), 'stop': float(segs[j][1]), 'speaker': int(segs[j][2])},
                                'cosine': float(S[i, j]),
                            })
        except Exception:
            pass

        meta = {
            'wav_path': wav_path,
            'duration_sec': duration_sec,
            'processing_time_sec': elapsed,
            'rtf': (elapsed / duration_sec) if (duration_sec and duration_sec > 0) else None,
            'pairwise_min_cosine': pair_min,
            'pairwise_mean_cosine': pair_mean,
        }
        meta_file = os.path.join(os.path.dirname(out_file), f"{wav_id}.meta.json")
        try:
            with open(meta_file, 'w') as mf:
                json.dump(meta, mf, indent=2)
        except Exception:
            pass

        # write pairwise sidecar file
        try:
            pairs_file = os.path.join(os.path.dirname(out_file), f"{wav_id}.pairs.json")
            with open(pairs_file, 'w') as pf:
                json.dump({'pairs': pairwise_list}, pf, indent=2)
        except Exception:
            pass

def main():
    args = parser.parse_args()
    if args.include_overlap and args.hf_access_token is None:
        parser.error("--hf_access_token is required when --include_overlap is specified.")
    
    get_speaker_embedding_model()
    get_voice_activity_detection_model()
    get_cluster_backend()
    if args.include_overlap:
        get_segmentation_model(args.hf_access_token)
    print(f'[INFO]: Model downloaded successfully.')

    if args.wav.endswith('.wav'):
        # input is a wav file
        wav_list = [args.wav]
    else:
        try:
            # input should be a wav list
            with open(args.wav,'r') as f:
                wav_list = [i.strip() for i in f.readlines()]
        except:
            raise Exception('[ERROR]: Input should be a wav file or a wav list.')
    assert len(wav_list) > 0

    if args.nprocs is None:
        ngpus = torch.cuda.device_count()
        if ngpus > 0:
            print(f'[INFO]: Detected {ngpus} GPUs.')
            args.nprocs = ngpus
        else:
            print('[INFO]: No GPUs detected.')
            args.nprocs = 1

    args.nprocs = min(len(wav_list), args.nprocs)
    print(f'[INFO]: Set {args.nprocs} processes to extract embeddings.')

    # output dir (removed automatic 3D-Speaker_ prefix addition)
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    mp.spawn(main_process, nprocs=args.nprocs, args=(args.nprocs, args, wav_list))

if __name__ == '__main__':
    main()
