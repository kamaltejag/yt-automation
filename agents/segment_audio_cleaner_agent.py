# agents/segment_audio_cleaner_agent.py

import os
import json
import logging
import subprocess
from typing import Dict, List
from pydub import AudioSegment
from config import LOG_FORMAT, LOG_LEVEL

class SegmentAudioCleaner:
    def __init__(self, audio_input_folder, transcript_folder, output_folder, log_base_dir=None):
        self.audio_input_folder = audio_input_folder
        self.transcript_folder = transcript_folder
        self.output_folder = output_folder
        self.log_base_dir = log_base_dir

        os.makedirs(self.output_folder, exist_ok=True)
        if self.log_base_dir:
            os.makedirs(self.log_base_dir, exist_ok=True)
            # Use a file handler instead of basicConfig
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(getattr(logging, LOG_LEVEL))
            handler = logging.FileHandler(os.path.join(self.log_base_dir, "audio_segment_clean.log"))
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
        else:
            self.logger = logging.getLogger(__name__)

    def log_decision(self, file_name, reason):
        if not self.log_base_dir:
            return

        log_path = os.path.join(self.log_base_dir, f"{os.path.splitext(file_name)[0]}_audio_clean.json")
        with open(log_path, "w") as f:
            json.dump({
                "file": file_name,
                "stage": "audio_segment_clean",
                "decision": reason
            }, f, indent=2)

    def validate_transcript(self, transcript: Dict) -> bool:
        """Validate the structure of the transcript."""
        if not isinstance(transcript, dict):
            return False
        if "segments" not in transcript:
            return False
        if not isinstance(transcript["segments"], list):
            return False
        for segment in transcript["segments"]:
            if not all(key in segment for key in ["start", "end", "text"]):
                return False
            if not isinstance(segment["start"], (int, float)) or not isinstance(segment["end"], (int, float)):
                return False
            if segment["end"] <= segment["start"]:
                return False
        return True

    def process_file(self, file_name):
        base_name = os.path.splitext(file_name)[0]
        audio_path = os.path.join(self.audio_input_folder, f"{base_name}_clean.wav")
        transcript_path = os.path.join(self.transcript_folder, f"{base_name}_llm_cleaned.json")
        output_path = os.path.join(self.output_folder, f"{base_name}_audio_clean.wav")

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            if not os.path.exists(transcript_path):
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

            # Load full audio
            audio = AudioSegment.from_wav(audio_path)

            with open(transcript_path, 'r') as f:
                transcript = json.load(f)

            # Validate transcript
            if not self.validate_transcript(transcript):
                raise ValueError("Invalid transcript format")

            # Create cleaned audio
            cleaned_audio = AudioSegment.empty()
            for segment in transcript["segments"]:
                start_ms = int(segment["start"] * 1000)
                end_ms = int(segment["end"] * 1000)
                
                # Validate segment timing
                if start_ms >= len(audio) or end_ms > len(audio):
                    logging.warning(f"Segment timing out of bounds for {file_name}: {start_ms}-{end_ms}ms")
                    continue
                
                cleaned_audio += audio[start_ms:end_ms]

            # Export with high quality settings
            cleaned_audio.export(
                output_path,
                format="wav",
                parameters=["-ar", "48000", "-ac", "2", "-b:a", "192k"]
            )

            self.log_decision(file_name, "success")
            self.logger.info(f"[INFO] Successfully exported cleaned audio for {file_name}")

        except Exception as e:
            error_msg = f"failed - {str(e)}"
            self.logger.error(f"[ERROR] Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)

    def run(self):
        files = [f for f in os.listdir(self.audio_input_folder) if f.endswith("_clean.wav")]
        for f in files:
            original_file = f.replace("_clean.wav", ".mp4")
            self.process_file(original_file)