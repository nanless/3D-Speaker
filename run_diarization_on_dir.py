#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
import tempfile
from glob import glob


def find_wavs(src_dir: str, pattern: str) -> list:
    wavs = sorted(glob(os.path.join(src_dir, pattern)))
    return wavs


def run_diarization(infer_script: str, wav_list_path: str, out_dir: str, out_type: str,
                    include_overlap: bool, hf_access_token: str | None,
                    nprocs: int | None, speaker_num: int | None) -> None:
    cmd = [
        sys.executable,
        infer_script,
        "--wav", wav_list_path,
        "--out_dir", out_dir,
        "--out_type", out_type,
        "--diable_progress_bar",
    ]
    if include_overlap:
        cmd.append("--include_overlap")
        if hf_access_token:
            cmd.extend(["--hf_access_token", hf_access_token])
    if nprocs is not None:
        cmd.extend(["--nprocs", str(nprocs)])
    if speaker_num is not None:
        cmd.extend(["--speaker_num", str(speaker_num)])

    # Forward optional behavior controls if present in environment variables (handled by caller)
    # This function will be called with explicit flags in main below.

    subprocess.run(cmd, check=True)


def aggregate_results(wavs: list, out_dir: str, out_type: str, summary_out: str, per_sentence_reindex: bool) -> None:
    summary = []
    for w in wavs:
        base = os.path.splitext(os.path.basename(w))[0]
        out_file = os.path.join(out_dir, f"{base}.{out_type}")
        if not os.path.isfile(out_file):
            # Skip missing outputs to be robust
            continue
        with open(out_file, "r") as f:
            diar = json.load(f)
        # diar is a dict: segid -> {start, stop, speaker}
        segments = list(diar.values())

        # Optionally reindex speakers per file (treat each file as one sentence/utterance)
        if per_sentence_reindex and segments:
            # determine order by first occurrence in time
            segments_sorted = sorted(segments, key=lambda s: (float(s.get("start", 0.0)), float(s.get("stop", 0.0))))
            order = []
            seen = set()
            for s in segments_sorted:
                spk = int(s["speaker"])
                if spk not in seen:
                    seen.add(spk)
                    order.append(spk)
            remap = {old: new for new, old in enumerate(order)}
            # build reindexed diar and write alongside original
            diar_reindexed = {}
            for k, v in diar.items():
                spk_old = int(v["speaker"])
                v_new = dict(v)
                v_new["speaker"] = int(remap.get(spk_old, spk_old))
                diar_reindexed[k] = v_new
            re_file = os.path.join(out_dir, f"{base}.reindexed.{out_type}")
            with open(re_file, "w") as rf:
                json.dump(diar_reindexed, rf, indent=2, ensure_ascii=False)
            # replace for summary
            diar = diar_reindexed
            out_file = re_file
            segments = list(diar.values())

        speakers = sorted({int(s["speaker"]) for s in segments}) if segments else []
        # read meta (duration, processing, rtf) if exists
        meta_file = os.path.join(out_dir, f"{base}.meta.json")
        duration_sec = None
        processing_time_sec = None
        rtf = None
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, 'r') as mf:
                    meta = json.load(mf)
                duration_sec = meta.get('duration_sec', None)
                processing_time_sec = meta.get('processing_time_sec', None)
                rtf = meta.get('rtf', None)
            except Exception:
                pass
        summary.append({
            "wav_path": w,
            "result_file": out_file,
            "num_speakers": len(speakers),
            "speakers": speakers,
            "segments": segments,
            "duration_sec": duration_sec,
            "processing_time_sec": processing_time_sec,
            "rtf": rtf,
        })

    os.makedirs(os.path.dirname(summary_out), exist_ok=True)
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Batch diarization on a directory and aggregate results.")
    parser.add_argument("--src_dir", required=True, help="Source directory to scan")
    parser.add_argument("--pattern", default="*speech_estimate.wav", help="Filename pattern to match")
    parser.add_argument("--out_dir", default=None, help="Output directory for diarization JSON")
    parser.add_argument("--summary_out", default=None, help="Path to write aggregated JSON summary")
    parser.add_argument("--infer_script", default=os.path.join(os.path.dirname(__file__), "speakerlab", "bin", "infer_diarization.py"),
                        help="Path to infer_diarization.py")
    parser.add_argument("--include_overlap", action="store_true", help="Include overlapping speech")
    parser.add_argument("--hf_access_token", default=None, help="HF token for overlap model")
    parser.add_argument("--nprocs", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--speaker_num", type=int, default=None, help="Oracle speaker number if known")
    parser.add_argument("--out_type", choices=["json"], default="json", help="Output format (fixed json)")
    parser.add_argument("--per_sentence_reindex", action="store_true", help="Reindex speaker IDs per file from 0")
    parser.add_argument("--no_chunk_after_vad", action="store_true", help="Do not split VAD segments; one embedding per VAD segment")
    parser.add_argument("--vad_min_speech_ms", type=float, default=None, help="VAD post-process: minimum speech segment duration (milliseconds)")
    parser.add_argument("--vad_max_silence_ms", type=float, default=None, help="VAD post-process: maximum silence gap to fill (milliseconds)")
    parser.add_argument("--vad_energy_threshold", type=float, default=None, help="VAD energy threshold for boundary refinement")
    parser.add_argument("--vad_boundary_expansion_ms", type=float, default=None, help="VAD boundary expansion (milliseconds)")
    parser.add_argument("--vad_threshold", type=float, default=None, help="VAD threshold for TenVad (default: 0.5)")
    parser.add_argument("--cluster_mer_cos", type=float, default=None, help="Clustering merge cosine threshold (default: 0.3)")
    parser.add_argument("--cluster_fix_cos_thr", type=float, default=None, help="Clustering fixed cosine threshold (default: 0.3)")
    parser.add_argument("--cluster_min_cluster_size", type=int, default=None, help="Clustering minimum cluster size (default: 0)")
    parser.add_argument("--chunk_dur", type=float, default=None, help="Chunk duration in seconds for sliding window mode (default: 1.5)")
    parser.add_argument("--chunk_step", type=float, default=None, help="Chunk step size in seconds for sliding window mode (default: 0.75)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for embedding extraction (default: 64)")

    args = parser.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    # compute out_dir and enforce 3D-Speaker_ prefix on the last component
    _base_out = args.out_dir or os.path.join(src_dir, "diarization_out")
    _base_out_norm = os.path.normpath(_base_out)
    _parent = os.path.dirname(_base_out_norm)
    _name = os.path.basename(_base_out_norm)
    if not _name.startswith("3D-Speaker"):
        _name = f"3D-Speaker_{_name}"
    out_dir = os.path.join(_parent, _name)
    summary_out = args.summary_out or os.path.join(out_dir, "diarization_summary.json")

    wavs = find_wavs(src_dir, args.pattern)
    if len(wavs) == 0:
        raise SystemExit(f"No files matched pattern `{args.pattern}` under {src_dir}")

    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        wav_list_path = os.path.join(td, "wav_list.txt")
        with open(wav_list_path, "w") as f:
            for w in wavs:
                f.write(w + "\n")

        # Build command and append new flags here to ensure they are forwarded
        cmd = [
            sys.executable,
            args.infer_script,
            "--wav", wav_list_path,
            "--out_dir", out_dir,
            "--out_type", args.out_type,
            "--diable_progress_bar",
        ]
        if args.include_overlap:
            cmd.append("--include_overlap")
            if args.hf_access_token:
                cmd.extend(["--hf_access_token", args.hf_access_token])
        if args.nprocs is not None:
            cmd.extend(["--nprocs", str(args.nprocs)])
        if args.speaker_num is not None:
            cmd.extend(["--speaker_num", str(args.speaker_num)])
        if args.no_chunk_after_vad:
            cmd.append("--no_chunk_after_vad")
        if args.vad_min_speech_ms is not None:
            cmd.extend(["--vad_min_speech_ms", str(args.vad_min_speech_ms)])
        if args.vad_max_silence_ms is not None:
            cmd.extend(["--vad_max_silence_ms", str(args.vad_max_silence_ms)])
        if args.vad_energy_threshold is not None:
            cmd.extend(["--vad_energy_threshold", str(args.vad_energy_threshold)])
        if args.vad_boundary_expansion_ms is not None:
            cmd.extend(["--vad_boundary_expansion_ms", str(args.vad_boundary_expansion_ms)])
        if args.vad_threshold is not None:
            cmd.extend(["--vad_threshold", str(args.vad_threshold)])
        if args.cluster_mer_cos is not None:
            cmd.extend(["--cluster_mer_cos", str(args.cluster_mer_cos)])
        if args.cluster_fix_cos_thr is not None:
            cmd.extend(["--cluster_fix_cos_thr", str(args.cluster_fix_cos_thr)])
        if args.cluster_min_cluster_size is not None:
            cmd.extend(["--cluster_min_cluster_size", str(args.cluster_min_cluster_size)])
        if args.chunk_dur is not None:
            cmd.extend(["--chunk_dur", str(args.chunk_dur)])
        if args.chunk_step is not None:
            cmd.extend(["--chunk_step", str(args.chunk_step)])
        if args.batch_size is not None:
            cmd.extend(["--batch_size", str(args.batch_size)])

        subprocess.run(cmd, check=True)

    aggregate_results(wavs=wavs, out_dir=out_dir, out_type=args.out_type, summary_out=summary_out, per_sentence_reindex=args.per_sentence_reindex)
    print(f"[INFO] Finished. Aggregated summary saved to: {summary_out}")


if __name__ == "__main__":
    main()


