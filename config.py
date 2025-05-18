# config.py

import os
from typing import Dict

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Ollama Configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '300'))

# Pipeline Configuration
DEFAULT_OUTPUT_DIR = os.getenv('DEFAULT_OUTPUT_DIR', 'output')
DEFAULT_LOG_DIR = os.getenv('DEFAULT_LOG_DIR', 'data/logs')
DEFAULT_MODEL_PATH = os.getenv('DEFAULT_MODEL_PATH', 'models/arnndn/lq.rnnn')

# Output Directory Structure
OUTPUT_DIRS: Dict[str, str] = {
    "denoised": "denoised",
    "edited_segments": "edited_segments",
    "transcripts": "transcripts",
    "edited": "edited",
    "timelines": "timelines",
    "audio": "audio",  # Added for cleaned audio files
    "video": "video"   # Added for cleaned video files
}

# FFmpeg Configuration
FFMPEG_AUDIO_SETTINGS = {
    "sample_rate": "48000",
    "channels": "2",
    "bitrate": "192k"
}

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO') 