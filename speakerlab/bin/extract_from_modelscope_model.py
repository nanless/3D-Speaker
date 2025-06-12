#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import re
import pathlib
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from kaldiio import WriteHelper
import torch.multiprocessing as mp
import logging

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import
from speakerlab.utils.utils import get_logger
from speakerlab.utils.fileio import load_wav_scp

# Model configurations
CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2NetV2_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 26,
        'scale': 2,
        'expansion': 2,
    },
}

ERes2NetV2_w24s4ep4_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 24,
        'scale': 4,
        'expansion': 4,
    },
}

ECAPA_CNCeleb = {
    'obj': 'speakerlab.models.ecapa_tdnn.ECAPA_TDNN.ECAPA_TDNN',
    'args': {
        'input_size': 80,
        'lin_neurons': 192,
        'channels': [1024, 1024, 1024, 1024, 3072],
    },
}

supports = {
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'iic/speech_eres2netv2_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_COMMON,
        'model_pt': 'pretrained_eres2netv2.ckpt',
    },
    'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_w24s4ep4_COMMON,
        'model_pt': 'pretrained_eres2netv2w24s4ep4.ckpt',
    },
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    'iic/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'iic/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
}

def process_subset(args, device_id, data_subset, model_dir, save_dir, conf, queue):
    """Process a subset of data on a specific GPU"""
    try:
        device = torch.device(f'cuda:{device_id}' if device_id >= 0 else 'cpu')
        
        # Load model
        pretrained_model = model_dir / conf['model_pt']
        pretrained_state = torch.load(pretrained_model, map_location='cpu')
        model = conf['model']
        embedding_model = dynamic_import(model['obj'])(**model['args'])
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.to(device)
        embedding_model.eval()

        # Setup feature extractor
        feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

        # Create embedding directory for this process
        embedding_dir = os.path.join(save_dir, 'embeddings')
        os.makedirs(embedding_dir, exist_ok=True)
        
        emb_ark = os.path.join(embedding_dir, f'xvector_{device_id}.ark')
        emb_scp = os.path.join(embedding_dir, f'xvector_{device_id}.scp')

        with torch.no_grad():
            with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
                for k, wav_path in tqdm(data_subset.items(), 
                                      desc=f'GPU-{device_id}',
                                      position=device_id+1):
                    try:
                        wav, fs = torchaudio.load(wav_path)
                        if fs != 16000:
                            wav = torchaudio.functional.resample(wav, fs, 16000)
                        if wav.shape[0] > 1:
                            wav = wav[0, :].unsqueeze(0)
                        
                        feat = feature_extractor(wav)
                        feat = feat.unsqueeze(0)
                        feat = feat.to(device)
                        emb = embedding_model(feat).detach().cpu().numpy()
                        writer(k, emb)
                    except Exception as e:
                        logging.error(f"Error processing {wav_path}: {str(e)}")
                        continue
        
        queue.put((device_id, emb_ark, emb_scp))
    except Exception as e:
        queue.put((device_id, None, None, str(e)))

def setup_logger(save_dir):
    """Setup logging configuration"""
    log_file = os.path.join(save_dir, 'extract.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings using ModelScope models.')
    parser.add_argument('--model_id', required=True, type=str, help='Model id in modelscope')
    parser.add_argument('--data', required=True, type=str, help='Data directory containing wav.scp')
    parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model directory')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU or not')
    parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--test_dataset_name', type=str, default='', help='Test dataset name')
    args = parser.parse_args()
    
    # Validate model ID
    assert isinstance(args.model_id, str) and \
        is_official_hub_path(args.model_id), "Invalid modelscope model id."
    if args.model_id.startswith('damo/'):
        args.model_id = args.model_id.replace('damo/','iic/', 1)
    assert args.model_id in supports, "Model id not currently supported."

    # Setup paths and logger
    model_dir = os.path.join(args.local_model_dir, args.model_id.split('/')[1])
    save_dir = os.path.join(args.local_model_dir, args.model_id.split('/')[1], args.test_dataset_name)
    model_dir = pathlib.Path(model_dir)
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(save_dir)

    # Download model from ModelScope
    conf = supports[args.model_id]
    try:
        cache_dir = snapshot_download(
            args.model_id,
            revision=conf['revision'],
        )
        cache_dir = pathlib.Path(cache_dir)
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        sys.exit(1)

    # Link necessary files
    download_files = ['examples', conf['model_pt']]
    for src in cache_dir.glob('*'):
        if re.search('|'.join(download_files), src.name):
            dst = model_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)

    # Load data
    try:
        data = load_wav_scp(args.data)
    except Exception as e:
        logger.error(f"Failed to load wav.scp: {str(e)}")
        sys.exit(1)

    # Setup device and workers
    if args.use_gpu and torch.cuda.is_available():
        if args.gpu:
            devices = [int(g) for g in args.gpu]
        else:
            devices = list(range(torch.cuda.device_count()))
        num_workers = min(len(devices), args.num_workers)
    else:
        devices = [-1]  # CPU
        num_workers = 1
        if args.use_gpu:
            logger.warning('No cuda device detected. Using CPU.')

    # Split data among workers
    data_keys = list(data.keys())
    chunks = np.array_split(data_keys, num_workers)
    data_subsets = [{k: data[k] for k in chunk} for chunk in chunks]

    # Start parallel processing
    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()
    processes = []
    
    logger.info(f'Starting embedding extraction with {num_workers} workers')
    
    try:
        for i in range(num_workers):
            device_id = devices[i] if args.use_gpu else -1
            p = mp.Process(
                target=process_subset,
                args=(args, device_id, data_subsets[i], model_dir, save_dir, conf, queue)
            )
            p.start()
            processes.append(p)

        # Collect results
        successful_files = []
        failed_workers = []
        
        for _ in range(num_workers):
            result = queue.get()
            if len(result) == 3:
                successful_files.append(result)
            else:
                failed_workers.append(result)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Report any failures
        if failed_workers:
            for worker_id, _, _, error in failed_workers:
                logger.error(f"Worker {worker_id} failed: {error}")

        # # Merge results if we have successful extractions
        # if successful_files:
        #     final_ark, final_scp = merge_embeddings(save_dir, successful_files)
        #     logger.info(f'Successfully extracted embeddings to {final_ark} and {final_scp}')
        # else:
        #     logger.error("No embeddings were successfully extracted")
        #     sys.exit(1)

    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        for p in processes:
            p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main() 