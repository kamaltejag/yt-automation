# agents/transcriber_agent.py

import os
import logging
import json
from typing import Dict, List, Optional, Tuple
from faster_whisper import WhisperModel

class TranscriberAgent:
    def __init__(self, audio_input_folder: str, output_folder: str, model_size: str = "base", log_base_dir: Optional[str] = None):
        self.audio_input_folder = audio_input_folder
        self.output_folder = output_folder
        self.model_size = model_size
        self.log_base_dir = log_base_dir

        os.makedirs(self.output_folder, exist_ok=True)
        if self.log_base_dir:
            os.makedirs(self.log_base_dir, exist_ok=True)
            # Use a file handler instead of basicConfig
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.log_base_dir, "transcribe.log"))
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
        else:
            self.logger = logging.getLogger(__name__)

        # Initialize Whisper model
        try:
            self.model = WhisperModel(self.model_size, compute_type="int8")
            self.logger.info(f"Initialized Whisper model with size: {model_size}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {str(e)}")
            raise

    def log_decision(self, file_name: str, reason: str) -> None:
        """Log a decision about file processing."""
        if not self.log_base_dir:
            return

        log_path = os.path.join(self.log_base_dir, f"{os.path.splitext(file_name)[0]}_transcribe.json")
        try:
            with open(log_path, "w") as f:
                json.dump({
                    "file": file_name,
                    "stage": "transcribe",
                    "decision": reason,
                    "timestamp": logging.Formatter("%(asctime)s").formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write decision log: {str(e)}")

    def validate_transcript(self, transcript: Dict) -> bool:
        """Validate the structure of the transcript."""
        if not isinstance(transcript, dict):
            self.logger.error("Transcript is not a dictionary")
            return False
        if "segments" not in transcript:
            self.logger.error("Transcript missing 'segments' key")
            return False
        if not isinstance(transcript["segments"], list):
            self.logger.error("Transcript segments is not a list")
            return False
        if not transcript["segments"]:
            self.logger.error("Transcript segments list is empty")
            return False

        for i, segment in enumerate(transcript["segments"]):
            if not all(key in segment for key in ["start", "end", "text"]):
                self.logger.error(f"Segment {i} missing required keys")
                return False
            if not isinstance(segment["start"], (int, float)) or not isinstance(segment["end"], (int, float)):
                self.logger.error(f"Segment {i} has invalid start/end types")
                return False
            if segment["end"] <= segment["start"]:
                self.logger.error(f"Segment {i} has invalid time range")
                return False
            if not isinstance(segment["text"], str) or not segment["text"].strip():
                self.logger.error(f"Segment {i} has invalid or empty text")
                return False
        return True

    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe the audio file and return the transcript."""
        try:
            self.logger.info(f"Starting transcription of {audio_path}")
            segments, _ = self.model.transcribe(audio_path, beam_size=5)

            # Parse into required format
            transcript = {"segments": []}
            for segment in segments:
                transcript["segments"].append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip()
                })

            if not self.validate_transcript(transcript):
                raise ValueError("Generated transcript failed validation")

            self.logger.info(f"Successfully transcribed {audio_path}")
            return transcript

        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise

    def process_file(self, file_name: str) -> None:
        """Process a single audio file."""
        base_name = os.path.splitext(file_name)[0]
        audio_path = os.path.join(self.audio_input_folder, f"{base_name}_clean.wav")
        output_path = os.path.join(self.output_folder, f"{base_name}_transcript.json")

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Cleaned audio file not found: {audio_path}")

            self.logger.info(f"Processing {file_name}")
            transcript = self.transcribe_audio(audio_path)
            
            # Save transcript to JSON file
            with open(output_path, "w") as f:
                json.dump(transcript, f, indent=2)
            
            self.log_decision(file_name, "success")
            self.logger.info(f"Successfully processed {file_name}")

        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            self.logger.error(f"Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)
        except ValueError as e:
            error_msg = f"Invalid data: {str(e)}"
            self.logger.error(f"Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)

    def run(self) -> None:
        """Process all audio files in the input folder."""
        try:
            wav_files = [f for f in os.listdir(self.audio_input_folder) if f.endswith("_clean.wav")]
            if not wav_files:
                self.logger.warning("No audio files found to process")
                return

            for wav_file in wav_files:
                original_name = wav_file.replace("_clean.wav", ".mp4")
                self.process_file(original_name)

        except Exception as e:
            self.logger.error(f"Failed to process audio folder: {str(e)}")
            raise
