#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run diarization (VAD + Embedding + Clustering) on *_speech_estimate.wav files
in the original_audios_SC_CausalMelBandRNN_EDA_16k_resume3_variable_length_narrowgap_E0001_B030000 directory.
This script does NOT use segmentation model (no overlap detection).

Usage:
    python run_diarization_speech_estimate.py [--src_dir DIR] [--out_dir DIR] [--nprocs N] [--speaker_num N]
"""

import os
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Path to the inference script
INFER_SCRIPT = SCRIPT_DIR / "speakerlab" / "bin" / "infer_diarization.py"

# Default source directory
DEFAULT_SRC_DIR = "/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios_SC_CausalMelBandRNN_EDA_16k_resume3_variable_length_narrowgap_E0001_B030000"

def find_audio_files(src_dir, extensions=None):
    """Find all audio files in the directory."""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    audio_files = []
    src_path = Path(src_dir)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    
    for ext in extensions:
        audio_files.extend(src_path.glob(f"*{ext}"))
        audio_files.extend(src_path.glob(f"*{ext.upper()}"))
    
    return sorted(audio_files)

def main():
    parser = argparse.ArgumentParser(
        description="Run diarization (VAD + Embedding + Clustering) on *_speech_estimate.wav files. "
                    "Does NOT use segmentation model (no overlap detection)."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default=DEFAULT_SRC_DIR,
        help=f"Source directory containing audio files (default: {DEFAULT_SRC_DIR})"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for results (default: <src_dir_parent>/<basename>_3dspeaker_diarization)"
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=None,
        help="Number of processes to use (default: auto-detect based on GPU count)"
    )
    parser.add_argument(
        "--speaker_num",
        type=int,
        default=None,
        help="Oracle number of speakers if known (optional)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_speech_estimate.wav",
        help="File pattern to match (default: *_speech_estimate.wav)"
    )
    parser.add_argument(
        "--no_chunk_after_vad",
        action="store_true",
        help="Do not split VAD segments; one embedding per VAD segment"
    )
    parser.add_argument(
        "--vad_min_speech_ms",
        type=float,
        default=None,
        help="VAD post-process: minimum speech segment duration (milliseconds)"
    )
    parser.add_argument(
        "--vad_max_silence_ms",
        type=float,
        default=None,
        help="VAD post-process: maximum silence gap to fill (milliseconds)"
    )
    parser.add_argument(
        "--vad_energy_threshold",
        type=float,
        default=None,
        help="VAD energy threshold for boundary refinement"
    )
    parser.add_argument(
        "--vad_boundary_expansion_ms",
        type=float,
        default=None,
        help="VAD boundary expansion (milliseconds)"
    )
    parser.add_argument(
        "--vad_threshold",
        type=float,
        default=None,
        help="VAD threshold for TenVad (default: 0.5)"
    )
    parser.add_argument(
        "--cluster_mer_cos",
        type=float,
        default=None,
        help="Clustering merge cosine threshold (default: 0.3)"
    )
    parser.add_argument(
        "--cluster_fix_cos_thr",
        type=float,
        default=None,
        help="Clustering fixed cosine threshold (default: 0.3)"
    )
    parser.add_argument(
        "--cluster_min_cluster_size",
        type=int,
        default=None,
        help="Clustering minimum cluster size (default: 0)"
    )
    parser.add_argument(
        "--chunk_dur",
        type=float,
        default=None,
        help="Chunk duration in seconds for sliding window mode (default: 1.5)"
    )
    parser.add_argument(
        "--chunk_step",
        type=float,
        default=None,
        help="Chunk step size in seconds for sliding window mode (default: 0.75)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for embedding extraction (default: 64)"
    )
    
    args = parser.parse_args()
    
    src_dir = os.path.abspath(args.src_dir)
    
    # Set output directory
    if args.out_dir is None:
        # Output directory: original directory name + "_3dspeaker_diarization"
        # Note: infer_diarization.py will automatically add "3D-Speaker_" prefix to the
        # last path component if it doesn't start with "3D-Speaker", so the final
        # directory will be "3D-Speaker_<basename>_3dspeaker_diarization"
        base_name = os.path.basename(src_dir) + "_3dspeaker_diarization"
        out_dir = os.path.join(os.path.dirname(src_dir), base_name)
    else:
        out_dir = os.path.abspath(args.out_dir)
    
    print(f"[INFO] Starting diarization on directory: {src_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Pattern: {args.pattern}")
    
    # Find all audio files matching the pattern
    if args.pattern.startswith("*.") or "*" in args.pattern:
        # Use pattern-based search
        from glob import glob
        audio_files = sorted(glob(os.path.join(src_dir, args.pattern)))
        audio_files = [Path(f) for f in audio_files]
    else:
        audio_files = find_audio_files(src_dir)
    
    if len(audio_files) == 0:
        print(f"[ERROR] No audio files found in {src_dir} matching pattern {args.pattern}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(audio_files)} audio files")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Create temporary wav list file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        wav_list_path = f.name
        for audio_file in audio_files:
            f.write(str(audio_file) + "\n")
    
    try:
        # Build command
        cmd = [
            sys.executable,
            str(INFER_SCRIPT),
            "--wav", wav_list_path,
            "--out_dir", out_dir,
            "--out_type", "json",
            "--diable_progress_bar",
        ]
        
        # Add optional parameters
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
        
        # Note: We do NOT add --include_overlap, so segmentation model will NOT be used
        # This means only VAD + Embedding + Clustering will be performed
        
        print(f"[INFO] Running diarization (VAD + Embedding + Clustering only, no overlap detection)...")
        print(f"[INFO] Command: {' '.join(cmd)}")
        print()
        
        # Run diarization
        subprocess.run(cmd, check=True)
        
        print()
        print(f"[INFO] Diarization completed successfully!")
        print(f"[INFO] Results saved to: {out_dir}")
        print(f"[INFO] Each audio file has:")
        print(f"  - <filename>.json: Diarization results in JSON format")
        print(f"  - <filename>.meta.json: Metadata (duration, processing time, RTF)")
        print(f"  - <filename>.pairs.json: Pairwise cosine similarities")
        print(f"  - <filename>.vad.png: VAD visualization")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Diarization failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted by user")
        sys.exit(1)
    finally:
        # Clean up temporary file
        if os.path.exists(wav_list_path):
            os.unlink(wav_list_path)

if __name__ == "__main__":
    main()

