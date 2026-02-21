# ===========================================
# Song Plotting and Analysis Tool for Audio Files
# Supports MP3, FLAC and WAV format
# Bey - 2026-02-20
# ===========================================
#
# Usage examples:
#
# Basic analysis
# python spat.py --file="./data/songs/My_Song.mp3"
#
# Analysis + plots shown on screen
# python spspataaf.py --file="./data/songs/My_Song.mp3" --plot
#
# Analysis + plots saved to PNG
# python spat.py --file="./data/songs/My_Song.mp3" --plot --out="./data/reports/My_Song_report"
#
# Analysis + plots saved to lossless WEBP
# python spat.py --file="./data/songs/My_Song.mp3" --plot --out="./data/reports/My_Song_report" --format webp
#
# Single file + PNG + CSV
# python spat.py --file song.wav --plot --outdir reports --export csv
#
# Batch folder + WEBP + JSON
# python spat.py --batch ./dataset --plot --format webp --export json
#
# ===========================================

import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
from pathlib import Path

SUPPORTED_INPUT_FORMATS = {".mp3", ".flac", ".wav"}
SUPPORTED_OUTPUT_FORMATS = {"png", "webp"}

def analyze_audio(file_path, plot=False, out_path=None, out_format="png"):
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"Unsupported input format: {ext}. Supported: {SUPPORTED_INPUT_FORMATS}")

    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = len(y) / sr

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    ibi = np.diff(beats) * 1000
    
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    window = 8
    local_bpm = []
    for i in range(len(beats) - window):
        segment = beats[i:i + window]
        ibis = np.diff(segment)
        local_bpm.append(60.0 / np.mean(ibis))
    local_bpm = np.array(local_bpm)
    
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop)
    tempi = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop, sr=sr)

    valid_bins = np.isfinite(tempi) & (tempi >= 40) & (tempi <= 200)
    local_tempo = np.full(tempogram.shape[1], np.nan)
    for t in range(tempogram.shape[1]):
        frame_energy = tempogram[valid_bins, t]
        if np.max(frame_energy) > 0:
            local_tempo[t] = tempi[valid_bins][np.argmax(frame_energy)]

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onset_density = len(onsets) / duration

    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop)
    beat_grid = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    phase_error = beats - beat_grid
    
    tau = 5.0
    tempo_stability_score = float(np.exp(-np.std(local_bpm) / tau))
    
    results = {
        "file": os.path.basename(file_path),
        "duration_sec": duration,
        # "global_bpm": float(tempo),
        "global_bpm": float(tempo[0]) if tempo.size > 0 else 0.0,
        "local_bpm_min": float(np.min(local_bpm)) if len(local_bpm) else np.nan,
        "local_bpm_mean": float(np.mean(local_bpm)) if len(local_bpm) else np.nan,
        "local_bpm_max": float(np.max(local_bpm)) if len(local_bpm) else np.nan,
        "tempo_std_bpm": float(np.std(local_bpm)) if len(local_bpm) else np.nan,
        "tempo_stability_score": tempo_stability_score,
        "ibi_mean_ms": float(np.mean(ibi)),
        "ibi_std_ms": float(np.std(ibi)),
        "ibi_cv_percent": float(100 * np.std(ibi) / np.mean(ibi)),
        "onsets_per_sec": float(onset_density),
        "rms_min": float(rms.min()),
        "rms_mean": float(rms.mean()),
        "rms_max": float(rms.max()),
        "centroid_min_hz": float(centroid.min()),
        "centroid_mean_hz": float(centroid.mean()),
        "centroid_max_hz": float(centroid.max()),
        "beat_phase_error_std_ms": float(np.std(phase_error) * 1000)
    }
    
    if plot and out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1); plt.plot(local_bpm); plt.title("Local BPM")
        plt.subplot(2, 2, 2); plt.plot(rms); plt.title("RMS")
        plt.subplot(2, 2, 3); plt.plot(centroid); plt.title("Centroid")
        plt.subplot(2, 2, 4); plt.plot(local_tempo); plt.title("Tempogram BPM")
        plt.tight_layout()

        if out_format == "png":
            plt.savefig(out_path.with_suffix(".png"), dpi=150)
        else:
            plt.savefig(out_path.with_suffix(".webp"), dpi=150, format="webp", lossless=True)
        plt.close()

    return results

def write_csv(results_list, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    keys = results_list[0].keys()
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results_list:
            writer.writerow(row)

def write_json(results_list, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)

def collect_files(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return [f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_INPUT_FORMATS]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Objective audio analysis pipeline for AI music benchmarking/evaluation :: Bey 2026-02-20")
    parser.add_argument("--file", help="Single audio file (mp3, wav, flac)")
    parser.add_argument("--batch", help="Folder of audio files")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--outdir", default="./reports", help="Directory for images and reports")
    parser.add_argument("--format", choices=["png", "webp"], default="png", help="Plot image format")
    parser.add_argument("--export", choices=["csv", "json"], help="Export results to CSV or JSON")

    args = parser.parse_args()

    if not args.file and not args.batch:
        raise SystemExit("Must specify --file or --batch")

    files = collect_files(args.file or args.batch)
    results_all = []

    for f in files:
        out_img = Path(args.outdir) / f"{f.stem}_analysis"
        res = analyze_audio(str(f), plot=args.plot, out_path=out_img, out_format=args.format)
        results_all.append(res)

    if args.export == "csv":
        write_csv(results_all, Path(args.outdir) / "analysis_results.csv")
    elif args.export == "json":
        write_json(results_all, Path(args.outdir) / "analysis_results.json")

    print("\n=== ANALYSIS COMPLETE ===")
    for r in results_all:
        print(r)
