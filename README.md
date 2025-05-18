# YouTube Video Automation Pipeline

An automated pipeline for processing YouTube videos, featuring noise reduction, transcription, LLM-based editing, and timeline generation.

## Features

- 🎥 Video denoising using ARNNDN model
- 🎙️ Audio transcription using Whisper
- 🤖 LLM-powered transcript cleaning and optimization
- ✂️ Video segment editing based on cleaned transcripts
- 📊 Timeline generation for final video

## Prerequisites

- Python 3.8+
- FFmpeg
- Ollama (running on Windows host)
- WSL2 (if running on Windows)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd yt-automation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` to set:
- `OLLAMA_URL`: URL for Ollama server (default: `http://localhost:11434` for WSL)
- `OLLAMA_MODEL`: Model name (default: `llama3`)
- Other configuration options as needed

## Usage

Process a single video:
```bash
python main.py input/video.mp4
```

Process all videos in a directory:
```bash
python main.py input/
```

With custom output and log directories:
```bash
python main.py input/video.mp4 --output-dir custom_output --log-dir custom_logs
```

## Directory Structure

```
.
├── input/              # Input videos
├── output/             # Pipeline outputs
│   ├── denoised/      # Denoised audio/video
│   ├── transcripts/   # Raw transcripts
│   ├── edited/        # LLM-cleaned transcripts
│   ├── edited_segments/ # Edited video segments
│   └── timelines/     # Final timelines
├── data/
│   └── logs/          # Pipeline logs
└── models/            # Noise reduction models
```

## Troubleshooting

1. If Ollama connection fails in WSL:
   - Ensure Ollama is running on Windows
   - Check Windows firewall settings
   - Verify the correct IP in `.env` (use `cat /etc/resolv.conf | grep nameserver`)

2. For FFmpeg issues:
   - Ensure FFmpeg is installed and in PATH
   - Check input video format compatibility

