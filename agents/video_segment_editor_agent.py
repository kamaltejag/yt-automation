# agents/video_segment_editor_agent.py

import os
import json
import logging
import subprocess
from typing import Dict, List
from config import LOG_FORMAT, LOG_LEVEL

class VideoSegmentEditor:
    def __init__(self, video_input_folder, transcript_folder, output_folder, log_base_dir=None):
        self.video_input_folder = video_input_folder
        self.transcript_folder = transcript_folder
        self.output_folder = output_folder
        self.log_base_dir = log_base_dir

        os.makedirs(self.output_folder, exist_ok=True)
        if self.log_base_dir:
            os.makedirs(self.log_base_dir, exist_ok=True)
            # Use a file handler instead of basicConfig
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(getattr(logging, LOG_LEVEL))
            handler = logging.FileHandler(os.path.join(self.log_base_dir, "video_segment_edit.log"))
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
        else:
            self.logger = logging.getLogger(__name__)

    def log_decision(self, file_name, reason):
        if not self.log_base_dir:
            return

        log_path = os.path.join(self.log_base_dir, f"{os.path.splitext(file_name)[0]}_video_edit.json")
        with open(log_path, "w") as f:
            json.dump({
                "file": file_name,
                "stage": "video_segment_edit",
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

    def run_ffmpeg_trim(self, input_path: str, output_path: str, segments: List[Dict]) -> None:
        """Trim and concatenate video segments using ffmpeg."""
        temp_dir = os.path.join(self.output_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        concat_file = os.path.join(temp_dir, "temp_concat.txt")

        try:
            # Create segment files
            segment_paths = []
            for idx, segment in enumerate(segments):
                start = segment["start"]
                duration = segment["end"] - segment["start"]
                segment_path = os.path.join(temp_dir, f"temp_segment_{idx}.mp4")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", input_path,
                    "-ss", str(start),
                    "-t", str(duration),
                    "-c:v", "copy", "-c:a", "copy",
                    segment_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg segment creation failed: {result.stderr}")
                
                segment_paths.append(segment_path)

            # Write concat file
            with open(concat_file, 'w') as f:
                for path in segment_paths:
                    f.write(f"file '{path}'\n")

            # Concat segments
            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path
            ]
            
            result = subprocess.run(cmd_concat, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg concatenation failed: {result.stderr}")

        finally:
            # Cleanup
            for path in segment_paths:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(concat_file):
                os.remove(concat_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def process_file(self, file_name):
        base_name = os.path.splitext(file_name)[0]
        video_path = os.path.join(self.video_input_folder, file_name)
        transcript_path = os.path.join(self.transcript_folder, f"{base_name}_llm_cleaned.json")
        output_path = os.path.join(self.output_folder, f"{base_name}_video_clean.mp4")

        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if not os.path.exists(transcript_path):
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

            with open(transcript_path, 'r') as f:
                transcript = json.load(f)

            # Validate transcript
            if not self.validate_transcript(transcript):
                raise ValueError("Invalid transcript format")

            self.run_ffmpeg_trim(video_path, output_path, transcript["segments"])
            
            self.log_decision(file_name, "success")
            self.logger.info(f"[INFO] Successfully exported cleaned video for {file_name}")

        except Exception as e:
            error_msg = f"failed - {str(e)}"
            self.logger.error(f"[ERROR] Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)

    def run(self):
        files = [f for f in os.listdir(self.video_input_folder) if f.endswith(".mp4")]
        for f in files:
            self.process_file(f)
