# spat

### Song Plotting and Analysis Tool for Audio Files

A deterministic, research-grade audio analysis pipeline designed for AI music benchmarking.
`spat` extracts objective temporal and spectral features to quantify the precision, stability,
and drift of audio generations.

## Key Features
- ✅ **Multi-Format:** Native support for `.mp3`, `.wav`, and `.flac`.
- ✅ **Temporal Metrics:** Global BPM, local BPM tracking, IBI (Inter-Beat Interval), and Tempo Stability.
- ✅ **Spectral Metrics:** RMS energy and Spectral Centroid tracking (min/mean/max).
- ✅ **Rhythmic Integrity:** Phase error calculation and onset density analysis.
- ✅ **Visual Validation:** Generates 4-pane diagnostic plots (Local BPM, RMS, Centroid, and Tempogram).
- ✅ **Pipeline Ready:** Batch process folders and export to `CSV` or `JSON` for Pandas/ML ingestion.

## Installation & Setup
```bash
# Clone the repository
git clone https://github.com/beyastard/spat
cd spat

# Install dependencies
pip install -r requirements.txt
```

## Usage
## Example Usage Patterns
### Single file + PNG + CSV
```bash
python spat.py --file song.wav --plot --outdir reports --export csv
```
### Batch folder + WEBP + JSON
```bash
python spat.py --batch ./dataset --plot --format webp --export json
```

## Technical Metrics Extracted
| Category | Metrics |
| :--- | :--- |
| **Tempo** | Global BPM, Local BPM (Min/Max/Mean/Std), Stability Score |
| **Rhythm** | IBI Mean/Std, IBI Coefficient of Variation (CV%), Onset Density |
| **Spectral** | Spectral Centroid (Hz), RMS Energy (Loudness) |
| **Precision** | Beat Phase Error (ms), Tempogram-based local tempo tracking |

## Output Visualizations
When `--plot` is enabled, `spat` generates a multi-pane report:
- **Local BPM:** Moving-window tempo calculation (8-beat window).
- **RMS:** Temporal energy profile of the signal.
- **Centroid:** "Spectral Brightness" over time.
- **Tempogram:** Rhythmic frequency energy used for local tempo estimation.

# Technical Notes
- **Local-First:** All processing is performed locally; no audio data is transmitted to external APIs.
- **Deterministic:** The tool uses standard signal processing (STFT/Onset Strength) rather than non-deterministic ML models for feature extraction.
- **Stability Score:** Calculated as $e^{-\sigma/\tau}$ where $\sigma$ is the standard deviation of local BPM and $\tau$ is the decay constant, providing a 0.0–1.0 score of rhythmic steadiness.

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
