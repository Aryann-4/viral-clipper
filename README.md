# Viral Clip Forge

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue)](https://aryann-4.github.io/viral-clipper/)
[![Open In Colab - Run Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aryann-4/viral-clipper/blob/main/Viral_Clipper_Demo.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Aryann-4/viral-clipper?quickstart=1)
[![Example Output](https://img.shields.io/badge/Example-Output%20Video-blue)](docs/demo.mp4)
[![GitHub](https://img.shields.io/badge/GitHub-Aryann--4%2Fviral--clipper-black)](https://github.com/Aryann-4/viral-clipper)

A fully local, free, open-source alternative to OpusClip for turning one long
video into several captioned, vertical 9:16 highlights.

## 🚀 Live Demo

**Open the project demo site:** <https://aryann-4.github.io/viral-clipper/>

It includes the example output and buttons to launch the app. There is no
permanently hosted Gradio instance; the interactive demo runs in your own
Colab or Codespace session:

- **[Run in Google Colab](https://colab.research.google.com/github/Aryann-4/viral-clipper/blob/main/Viral_Clipper_Demo.ipynb)** — run the cells, wait for the temporary Gradio URL, then upload a video or copy it into the Colab VM.
- **[Open in GitHub Codespaces](https://codespaces.new/Aryann-4/viral-clipper?quickstart=1)** — full dev environment in the browser; follow the setup below and bind to `0.0.0.0`.

**Example output:** [docs/demo.mp4](docs/demo.mp4) — a rendered 9:16 clip with reframing, burned-in captions, and active-word highlighting.

It runs entirely on your machine:

- **faster-whisper** (`base` or `small`) for word-level speech transcription
- **FFmpeg / FFprobe** for probing, trimming, reframing, encoding, and burning captions
- **OpenCV Haar cascades** for optional face-aware horizontal reframing
- **ASS subtitles** with short phrases, bold outlines, and an active-word highlight
- **NumPy** audio-energy analysis plus a transparent transcript scoring heuristic
- **Optional Ollama** local LLM reranking (disabled by default)
- **Gradio** local browser UI and a scriptable CLI

The app itself uses no paid APIs, accounts, tokens, or subscriptions. The
optional browser-run environments require a Google or GitHub account, and the
first Whisper run downloads a free model from Hugging Face once; after that the
app can run offline locally.

---

## Directory structure

```
viral-clipper/
├── app.py                      # Complete CLI + Gradio UI + processing pipeline
├── Viral_Clipper_Demo.ipynb    # Colab notebook — launches a live Gradio demo
├── requirements.txt            # Python dependencies (all free/open source)
├── README.md
├── .gitignore
├── docs/
│   ├── index.html              # GitHub Pages demo site
│   └── demo.mp4                # Example rendered output clip
└── output_clips/               # Generated MP4 clips, ASS captions, JSON manifest
    └── .gitkeep
```

At runtime the following local cache/output locations are also used:

```
~/.cache/viral-clipper/whisper/   # faster-whisper model cache
output_clips/                     # clips, captions and manifest
```

---

## Requirements

- Python **3.10+** (3.11 or 3.12 is the safest choice)
- FFmpeg **5.0+** with `libass` and `libx264`
- A CUDA GPU is optional. CPU-only operation is fully supported and automatic.

Check your FFmpeg build includes the subtitle renderer:

```bash
ffmpeg -hide_banner -filters | grep -E " ass | subtitles "
```

You should see both `ass` and `subtitles`. If not, install a standard/full
FFmpeg build from your OS package manager.

---

## Installation

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-venv python3-pip
cd viral-clipper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### macOS (Homebrew)

```bash
brew install ffmpeg python@3.12
cd viral-clipper
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
winget install Gyan.FFmpeg
winget install Python.Python.3.12
cd viral-clipper
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Verify the installation:

```bash
ffmpeg -version
ffprobe -version
python -c "import faster_whisper, gradio, cv2, numpy; print('Python dependencies ready')"
```

---

## Launch the local web UI

```bash
cd viral-clipper
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python app.py
```

Open <http://127.0.0.1:7860> if the browser does not open automatically.

Useful UI/launch options:

```bash
python app.py --host 127.0.0.1 --port 7861 --no-browser
python app.py --ffmpeg /usr/local/bin/ffmpeg --ffprobe /usr/local/bin/ffprobe
```

In GitHub Codespaces, install the system tool first and bind to all interfaces:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
pip install -r requirements.txt
python app.py --host 0.0.0.0 --no-browser
```

Then forward port `7860` in the Codespaces Ports panel.

The UI lets you:

1. Upload an MP4/MKV/MOV/WebM or paste a local path.
2. Pick a 30, 45, or 60 second target and 1–5 clips.
3. Choose face-aware or centered reframing.
4. Toggle burned-in captions.
5. Optionally enable local Ollama reranking.
6. Preview clips in the gallery and use the gallery download buttons, or
   download clips/captions/manifest from the files list.

---

## Command-line usage

```bash
# Basic: three ~45-second clips using the base Whisper model
python app.py --cli --input /path/to/podcast.mp4 --duration 45 --clips 3

# Faster CPU run with the smaller model and no face tracking
python app.py --cli \
  --input /path/to/video.mkv \
  --duration 30 \
  --clips 5 \
  --model base \
  --device cpu \
  --crop center

# Higher-accuracy transcription
python app.py --cli --input talk.mp4 --duration 60 --clips 3 --model small

# Optional local LLM reranking
ollama serve
ollama pull llama3.2
python app.py --cli \
  --input talk.mp4 \
  --duration 45 \
  --clips 3 \
  --ollama-model llama3.2

# Custom paths and output folder
python app.py --cli \
  --input ./videos/interview.mp4 \
  --duration 30 \
  --clips 3 \
  --crop face \
  --ffmpeg /usr/bin/ffmpeg \
  --ffprobe /usr/bin/ffprobe \
  --output-dir ./output_clips
```

The CLI prints progress and a JSON summary containing every output path, score,
score breakdown, transcript snippet, and the manifest location.

### CLI reference

| Option | Description | Default |
|---|---|---|
| `--cli` | Run the pipeline instead of the UI | UI |
| `--input PATH` | Local video file (required with `--cli`) | — |
| `--duration {30,45,60}` | Target highlight length | `45` |
| `--clips N` | Number of highlights (1–5) | `3` |
| `--model {base,small}` | faster-whisper model | `base` |
| `--language CODE` | Force a language, e.g. `en`, `hi`, `es`; omit to auto-detect | auto |
| `--crop {face,center}` | Reframing strategy | `face` |
| `--no-captions` | Do not burn captions | captions on |
| `--device {auto,cpu,cuda}` | Whisper device | `auto` |
| `--compute-type TYPE` | `auto`, `int8`, `float16`, `int8_float16`, `float32` | `auto` |
| `--ollama-model NAME` | Enable local Ollama reranking | off |
| `--ollama-url URL` | Local Ollama URL (localhost only) | `http://127.0.0.1:11434` |
| `--output-dir PATH` | Output folder | `./output_clips` |
| `--ffmpeg PATH` | Custom FFmpeg binary | PATH |
| `--ffprobe PATH` | Custom FFprobe binary | PATH |
| `--host`, `--port` | UI bind address and port | `127.0.0.1`, `7860` |
| `--no-browser` | Do not auto-open a browser | off |
| `--share` | Create a temporary public Gradio URL (for Colab/Kaggle) | off |
| `--verbose` | Debug logging | off |

---

## Output

Each run writes directly into the chosen output folder (default
`output_clips/`):

```
output_clips/
├── myvideo_20261126-071500-a1b2c3_clip01_0032s_0076s.mp4
├── myvideo_20261126-071500-a1b2c3_clip01_0032s_0076s.ass
├── myvideo_20261126-071500-a1b2c3_clip02_0104s_0149s.mp4
├── myvideo_20261126-071500-a1b2c3_clip02_0104s_0149s.ass
└── myvideo_20261126-071500-a1b2c3_manifest.json
```

- **MP4**: H.264/AAC, 1080×1920, 30 fps, fast-start enabled for social uploads.
- **ASS**: editable captions, useful for revisions or other editors.
- **JSON**: source metadata, settings, Whisper backend/language, sentence units,
  per-clip scores and breakdowns, face-tracking status, and warnings.

All clips are non-overlapping and start/end on sentence or clause boundaries,
with a small amount of padding so speech is not clipped.

---

## How highlights are selected

The pipeline does not rely on a black-box service. Each candidate receives a
transparent 0–100 score built from:

| Signal | What it measures |
|---|---|
| **Hook** | Questions, direct address, numbers, curiosity/emotion words in the opening |
| **Audio energy** | Mean and peak RMS energy from a 50 ms envelope, normalized to the source |
| **Keywords** | Density of high-value/emotion vocabulary |
| **Pace** | Words per minute near an engaging short-form range |
| **Clarity** | Penalizes filler-word density |
| **Completeness** | Ends on a complete sentence, not mid-thought |
| **Length** | Closeness to the requested 30/45/60 seconds |

Candidate windows are built from sentence units (with long monologues split at
pauses/clauses), then greedily selected highest-score-first with a 1.5-second
non-overlap gap. If fewer non-overlapping clips fit than requested, the app
warns you rather than silently reusing footage.

Optional Ollama reranking sends only the shortlisted transcript text to a
**local** model and blends its score 65/35 with the deterministic score. If
Ollama is unavailable the app logs a warning and continues with heuristics.

---

## Reframing and captions

- Sources are scaled with aspect-ratio preservation to cover 1080×1920 and
  center-cropped vertically.
- In **Face-aware** mode, frames are sampled at 2 fps, the dominant frontal face
  is detected with OpenCV, the path is median-smoothed and EMA-smoothed, and
  FFmpeg uses a piecewise-linear crop expression for horizontal tracking. If no
  reliable face is found, it falls back to a stable center crop.
- Vertical sources narrower than 9:16 are center-cropped vertically.
- Captions are generated as ASS at 1080×1920 with a bold font, thick black
  outline, bottom-center safe-zone margin, short phrases (up to 4 words / 20
  characters / 2.8 s), and the currently spoken word highlighted in yellow.
- The transcript is escaped before writing ASS so braces cannot inject styling.
- Captions are burned with FFmpeg's `ass`/libass filter.

Change the caption font with an environment variable:

```bash
VIRAL_CLIPPER_FONT="DejaVu Sans" python app.py
```

---

## CPU / GPU behavior

- `--device auto` queries CTranslate2 for CUDA devices.
- With CUDA: `float16` is used.
- Without CUDA: **CPU int8** is selected automatically.
- If a CUDA model load fails (for example missing cuDNN), the app retries with
  CPU int8 and records a warning in the manifest.
- Video encoding always uses the broadly compatible `libx264` CPU encoder, so
  the app works on any machine even when NVENC is unavailable.

Rendering is the slowest step on CPU. Use `base`, fewer clips, or a shorter target
first to validate quality.

---

## Error handling

The app fails with actionable messages for:

- Missing or invalid FFmpeg/FFprobe binaries, with install commands
- Unreadable/corrupted containers (FFprobe validation)
- Videos with no video stream, no audio stream, silent audio, or no speech
- FFmpeg build lacking `libass`
- Failed/partial renders (truncated files are removed; center-crop fallback if
  only the dynamic face crop fails)
- Unwritable output directories
- Failed first-time model downloads or incompatible compute types
- Unreachable Ollama server/model (non-fatal warning)
- Partial job failures (newly generated files from the failed job are cleaned up)

---

## Troubleshooting

**`FFmpeg/FFprobe was not found`**  
Install FFmpeg and confirm both binaries are on `PATH`. On Windows, reopen the
terminal after `winget install`.

**Captions are missing / `ass` filter error**  
Your FFmpeg build lacks libass. Install the distro `ffmpeg` package or a full
build from <https://ffmpeg.org/download.html>.

**First transcription is slow or fails**  
The first run downloads the model into `~/.cache/viral-clipper/whisper/`.
Confirm internet access and disk space. Subsequent runs use the cache.

**CUDA errors**  
The app automatically falls back to CPU. Force it with `--device cpu` to skip
the initial GPU attempt.

**Face tracking misses the subject**  
Haar cascades work best with a clear, front-facing face. Use `--crop center` for
screen recordings, gameplay, or non-human subjects.

**Port already in use**  
`python app.py --port 7861`

**No public Gradio URL with `--share`**
The local server still works; the public `gradio.live` link needs outbound
access to Gradio's tunnel service. Retry the Colab cell or use the forwarded
Codespaces port instead.

**Out of memory**  
Use `--model base`, close other GPU apps, or force `--device cpu`. Rendering
itself uses modest memory at 1080×1920.

---

## Privacy

- Uploaded/local videos are read directly from disk and never uploaded.
- Whisper inference is local.
- Gradio binds locally by default (`share=False`); the Colab notebook explicitly opts into a temporary public link with `--share`. Analytics/telemetry are disabled.
- Model downloads touch Hugging Face only on first use.
- Ollama reranking is optional, restricted to localhost URLs, and never
  required.
- All generated files remain in your chosen output folder.
