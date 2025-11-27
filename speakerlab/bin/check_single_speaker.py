#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch

# Reuse existing pipeline pieces to ensure consistency
try:
    from speakerlab.bin.infer_diarization import (
        get_speaker_embedding_model,
        get_voice_activity_detection_model,
    )
except Exception:
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from speakerlab.bin.infer_diarization import (
        get_speaker_embedding_model,
        get_voice_activity_detection_model,
    )
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.utils import circle_pad


def extract_segments_vad(wav: torch.Tensor, fs: int) -> List[Tuple[float, float]]:
    vad = get_voice_activity_detection_model()
    vad_results = vad(wav[0])[0]
    segs = [(float(a) / 1000.0, float(b) / 1000.0) for a, b in vad_results["value"]]
    return segs


def extract_embeddings_for_segments(
    wav: torch.Tensor,
    fs: int,
    segments: List[Tuple[float, float]],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    if not segments:
        return np.zeros((0, 192), dtype=np.float32)

    embedding_model, feature_extractor = get_speaker_embedding_model(device)

    # Slice waveform; pad all to same length, compute embeddings in batches
    chunk_wavs = [wav[0, int(st * fs) : int(ed * fs)] for st, ed in segments]
    max_len = max(x.shape[0] for x in chunk_wavs) if chunk_wavs else 0
    if max_len == 0:
        return np.zeros((0, 192), dtype=np.float32)
    chunk_wavs = [circle_pad(x, max_len) for x in chunk_wavs]
    chunk_wavs = torch.stack(chunk_wavs).unsqueeze(1)

    embs = []
    with torch.no_grad():
        i = 0
        while i < len(segments):
            wavs_batch = chunk_wavs[i : i + batch_size].to(device)
            feats_batch = torch.vmap(feature_extractor)(wavs_batch)
            embs_batch = embedding_model(feats_batch).cpu()
            embs.append(embs_batch)
            i += batch_size
    embs = torch.cat(embs, dim=0).numpy()
    return embs


def pairwise_cosine_min_mean(embs: np.ndarray) -> Tuple[float, float]:
    if embs.shape[0] <= 1:
        return 1.0, 1.0
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    Z = embs / norms
    S = np.dot(Z, Z.T)
    # take upper triangle excluding diagonal
    triu_idx = np.triu_indices(S.shape[0], k=1)
    vals = S[triu_idx]
    return float(np.min(vals)), float(np.mean(vals))


def pairwise_cosine_pairs(embs: np.ndarray) -> List[Tuple[int, int, float]]:
    if embs.shape[0] <= 1:
        return []
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    Z = embs / norms
    S = np.dot(Z, Z.T)
    pairs: List[Tuple[int, int, float]] = []
    n = S.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(S[i, j])))
    return pairs


def check_single_speaker(
    wav_path: str,
    threshold: float = 0.8,
    device: torch.device | None = None,
) -> dict:
    # Build feature extractor to know target fs
    embedding_model, feature_extractor = get_speaker_embedding_model(device)
    fs = feature_extractor.sample_rate

    wav = load_audio(wav_path, None, fs)  # [1, T]

    segments = extract_segments_vad(wav, fs)
    embs = extract_embeddings_for_segments(wav, fs, segments, device or torch.device("cpu"))
    min_sim, mean_sim = pairwise_cosine_min_mean(embs)
    pairs = pairwise_cosine_pairs(embs)
    is_single = bool(min_sim >= threshold)

    result = {
        "wav_path": wav_path,
        "num_segments": len(segments),
        "segments": [{"start": float(s), "stop": float(e)} for s, e in segments],
        "threshold": float(threshold),
        "min_pairwise_cosine": float(min_sim),
        "mean_pairwise_cosine": float(mean_sim),
        "is_single_speaker": is_single,
    }
    # attach pairwise similarities with segment times
    if pairs and len(segments) == embs.shape[0]:
        seg_list = [{"start": float(s), "stop": float(e)} for s, e in segments]
        result["pairwise_similarities"] = [
            {
                "i": i,
                "j": j,
                "cosine": float(v),
                "seg_i": seg_list[i],
                "seg_j": seg_list[j],
            }
            for (i, j, v) in pairs
        ]
    else:
        result["pairwise_similarities"] = []
    return result


def main():
    parser = argparse.ArgumentParser(description="Check if utterances are single-speaker via VAD+embeddings; supports batch mode.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", help="Path to a WAV file or a text file (one path per line)")
    group.add_argument("--src_dir", help="Directory to scan for wavs (use with --pattern)")
    parser.add_argument("--pattern", default="*speech_estimate.wav", help="Filename pattern when using --src_dir")
    parser.add_argument("--threshold", type=float, default=0.8, help="Cosine similarity threshold for single-speaker decision")
    parser.add_argument("--out", default=None, help="JSON output path for single-file/list mode")
    parser.add_argument("--out_dir", default=None, help="Output directory for batch mode; per-file JSONs will be written here")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    paths: List[str] = []
    batch_mode = False
    if args.src_dir:
        from glob import glob
        batch_mode = True
        src_dir = os.path.abspath(args.src_dir)
        paths = sorted(glob(os.path.join(src_dir, args.pattern)))
        if not paths:
            raise SystemExit(f"No files matched pattern `{args.pattern}` under {src_dir}")
    else:
        if args.wav.endswith(".wav"):
            paths = [args.wav]
        else:
            with open(args.wav, "r") as f:
                paths = [line.strip() for line in f if line.strip()]

    if batch_mode:
        out_dir = args.out_dir or os.path.join(os.path.abspath(args.src_dir), "single_spk_check")
        os.makedirs(out_dir, exist_ok=True)
        for p in paths:
            res = check_single_speaker(p, threshold=args.threshold, device=device)
            base = os.path.splitext(os.path.basename(p))[0]
            out_file = os.path.join(out_dir, f"{base}.single_spk.json")
            with open(out_file, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote {len(paths)} results to {out_dir}")
    else:
        results = []
        for p in paths:
            res = check_single_speaker(p, threshold=args.threshold, device=device)
            results.append(res)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()




import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch

# Reuse existing pipeline pieces to ensure consistency
try:
    from speakerlab.bin.infer_diarization import (
        get_speaker_embedding_model,
        get_voice_activity_detection_model,
    )
except Exception:
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from speakerlab.bin.infer_diarization import (
        get_speaker_embedding_model,
        get_voice_activity_detection_model,
    )
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.utils import circle_pad


def extract_segments_vad(wav: torch.Tensor, fs: int) -> List[Tuple[float, float]]:
    vad = get_voice_activity_detection_model()
    vad_results = vad(wav[0])[0]
    segs = [(float(a) / 1000.0, float(b) / 1000.0) for a, b in vad_results["value"]]
    return segs


def extract_embeddings_for_segments(
    wav: torch.Tensor,
    fs: int,
    segments: List[Tuple[float, float]],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    if not segments:
        return np.zeros((0, 192), dtype=np.float32)

    embedding_model, feature_extractor = get_speaker_embedding_model(device)

    # Slice waveform; pad all to same length, compute embeddings in batches
    chunk_wavs = [wav[0, int(st * fs) : int(ed * fs)] for st, ed in segments]
    max_len = max(x.shape[0] for x in chunk_wavs) if chunk_wavs else 0
    if max_len == 0:
        return np.zeros((0, 192), dtype=np.float32)
    chunk_wavs = [circle_pad(x, max_len) for x in chunk_wavs]
    chunk_wavs = torch.stack(chunk_wavs).unsqueeze(1)

    embs = []
    with torch.no_grad():
        i = 0
        while i < len(segments):
            wavs_batch = chunk_wavs[i : i + batch_size].to(device)
            feats_batch = torch.vmap(feature_extractor)(wavs_batch)
            embs_batch = embedding_model(feats_batch).cpu()
            embs.append(embs_batch)
            i += batch_size
    embs = torch.cat(embs, dim=0).numpy()
    return embs


def pairwise_cosine_min_mean(embs: np.ndarray) -> Tuple[float, float]:
    if embs.shape[0] <= 1:
        return 1.0, 1.0
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    Z = embs / norms
    S = np.dot(Z, Z.T)
    # take upper triangle excluding diagonal
    triu_idx = np.triu_indices(S.shape[0], k=1)
    vals = S[triu_idx]
    return float(np.min(vals)), float(np.mean(vals))


def check_single_speaker(
    wav_path: str,
    threshold: float = 0.8,
    device: torch.device | None = None,
) -> dict:
    # Build feature extractor to know target fs
    embedding_model, feature_extractor = get_speaker_embedding_model(device)
    fs = feature_extractor.sample_rate

    wav = load_audio(wav_path, None, fs)  # [1, T]

    segments = extract_segments_vad(wav, fs)
    embs = extract_embeddings_for_segments(wav, fs, segments, device or torch.device("cpu"))
    min_sim, mean_sim = pairwise_cosine_min_mean(embs)
    is_single = bool(min_sim >= threshold)

    return {
        "wav_path": wav_path,
        "num_segments": len(segments),
        "segments": [{"start": float(s), "stop": float(e)} for s, e in segments],
        "threshold": float(threshold),
        "min_pairwise_cosine": float(min_sim),
        "mean_pairwise_cosine": float(mean_sim),
        "is_single_speaker": is_single,
    }


def main():
    parser = argparse.ArgumentParser(description="Check if an utterance is single-speaker via VAD+embeddings.")
    parser.add_argument("--wav", required=True, help="Path to a WAV file or a text file (one path per line)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Cosine similarity threshold for single-speaker decision")
    parser.add_argument("--out", default=None, help="Optional JSON output path. If omitted, prints to stdout")
    args = parser.parse_args()

    paths: List[str]
    if args.wav.endswith(".wav"):
        paths = [args.wav]
    else:
        with open(args.wav, "r") as f:
            paths = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    results = []
    for p in paths:
        res = check_single_speaker(p, threshold=args.threshold, device=device)
        results.append(res)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


