# agents/denoise_agent.py

import os
import subprocess
import logging
import json
from typing import Optional

class DenoiseAgent:
    def __init__(self, input_folder: str, audio_output: str, video_output: str, model_path: str, log_base_dir: Optional[str] = None):
        self.input_folder = input_folder
        self.audio_output = audio_output
        self.video_output = video_output
        self.model_path = model_path
        self.log_base_dir = log_base_dir

        os.makedirs(self.audio_output, exist_ok=True)
        os.makedirs(self.video_output, exist_ok=True)
        if self.log_base_dir:
            os.makedirs(self.log_base_dir, exist_ok=True)
            # Use a file handler instead of basicConfig
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.log_base_dir, "denoise.log"))
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
        else:
            self.logger = logging.getLogger(__name__)

    def log_decision(self, file_name: str, reason: str) -> None:
        """Log a decision about file processing."""
        if not self.log_base_dir:
            return

        log_path = os.path.join(self.log_base_dir, f"{os.path.splitext(file_name)[0]}_denoise.json")
        try:
            with open(log_path, "w") as f:
                json.dump({
                    "file": file_name,
                    "stage": "denoise",
                    "decision": reason,
                    "timestamp": logging.Formatter("%(asctime)s").formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write decision log: {str(e)}")

    def convert_mp4_to_wav(self, input_mp4: str, output_wav: str) -> None:
        """Convert MP4 to WAV format."""
        try:
            self.logger.info(f"Converting {input_mp4} to WAV format")
            subprocess.run([
                "ffmpeg", "-i", input_mp4,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "48000",  # 48kHz sample rate
                "-ac", "2",  # Stereo
                output_wav,
                "-y"
            ], check=True, capture_output=True, text=True)
            self.logger.info(f"Successfully converted to {output_wav}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg conversion failed: {e.stderr}")
            raise RuntimeError(f"Failed to convert MP4 to WAV: {str(e)}")

    def apply_noise_reduction(self, input_wav: str, output_wav: str) -> None:
        """Apply noise reduction to WAV file."""
        try:
            self.logger.info(f"Applying noise reduction to {input_wav}")
            self.logger.info(f"Using model path: {self.model_path}")
            
            # Ensure model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Noise reduction model not found at: {self.model_path}")
            
            subprocess.run([
                "ffmpeg", "-i", input_wav,
                "-af", f"arnndn=m={self.model_path}",  # Use m= prefix for model path
                output_wav,
                "-y"
            ], check=True, capture_output=True, text=True)
            self.logger.info(f"Successfully applied noise reduction to {output_wav}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg noise reduction failed: {e.stderr}")
            raise RuntimeError(f"Failed to apply noise reduction: {str(e)}")
        except FileNotFoundError as e:
            self.logger.error(f"Model not found: {str(e)}")
            raise RuntimeError(f"Failed to apply noise reduction: {str(e)}")

    def convert_wav_to_mp4(self, input_wav: str, output_mp4: str, original_video: str) -> None:
        """Convert WAV back to MP4 with original video."""
        try:
            self.logger.info(f"Converting {input_wav} back to MP4")
            subprocess.run([
                "ffmpeg", "-i", original_video,
                "-i", input_wav,
                "-map", "0:v:0",  # Use video from first input
                "-map", "1:a:0",  # Use audio from second input
                "-c:v", "copy",  # Copy video codec
                "-c:a", "aac",  # Use AAC for audio
                "-b:a", "192k",  # Audio bitrate
                "-ar", "48000",  # Audio sample rate
                "-ac", "2",  # Stereo
                output_mp4,
                "-y"
            ], check=True, capture_output=True, text=True)
            self.logger.info(f"Successfully created {output_mp4}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg conversion failed: {e.stderr}")
            raise RuntimeError(f"Failed to convert WAV to MP4: {str(e)}")

    def process_file(self, file_name: str) -> None:
        """Process a single video file."""
        input_path = os.path.join(self.input_folder, file_name)
        temp_wav = os.path.join(self.audio_output, f"{os.path.splitext(file_name)[0]}_raw.wav")
        cleaned_wav = os.path.join(self.audio_output, f"{os.path.splitext(file_name)[0]}_clean.wav")
        output_mp4 = os.path.join(self.video_output, file_name)

        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input video not found: {input_path}")

            self.logger.info(f"Processing {file_name}")
            
            # Convert to WAV
            self.convert_mp4_to_wav(input_path, temp_wav)
            
            # Apply noise reduction
            self.apply_noise_reduction(temp_wav, cleaned_wav)
            
            if not os.path.exists(cleaned_wav):
                raise FileNotFoundError("Noise-reduced audio not found")

            # Convert back to MP4
            self.convert_wav_to_mp4(cleaned_wav, output_mp4, input_path)
            
            self.log_decision(file_name, "success")
            self.logger.info(f"Successfully processed {file_name}")

        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            self.logger.error(f"Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)
        except RuntimeError as e:
            error_msg = f"Processing error: {str(e)}"
            self.logger.error(f"Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                    self.logger.debug(f"Removed temporary file: {temp_wav}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {temp_wav}: {str(e)}")

    def run(self) -> None:
        """Process all video files in the input folder."""
        try:
            mp4_files = [f for f in os.listdir(self.input_folder) if f.endswith(".mp4")]
            if not mp4_files:
                self.logger.warning("No MP4 files found to process")
                return

            for file in mp4_files:
                self.process_file(file)

        except Exception as e:
            self.logger.error(f"Failed to process input folder: {str(e)}")
            raise
