# agents/editor_agent.py

import os
import json
import logging
import requests
from typing import Dict, List, Optional
from time import sleep
from config import OLLAMA_URL

class LLMEditingAgent:
    def __init__(self, transcript_input_folder: str, output_folder: str, model_name: str = "llama3", 
                 log_base_dir: Optional[str] = None, timeout: int = 300, max_retries: int = 3):
        self.transcript_input_folder = transcript_input_folder
        self.output_folder = output_folder
        self.model_name = model_name
        self.log_base_dir = log_base_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.ollama_url = OLLAMA_URL

        os.makedirs(self.output_folder, exist_ok=True)
        if self.log_base_dir:
            os.makedirs(self.log_base_dir, exist_ok=True)
            # Use a file handler instead of basicConfig
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.log_base_dir, "llm_editing.log"))
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
        else:
            self.logger = logging.getLogger(__name__)

    def log_decision(self, file_name: str, reason: str) -> None:
        """Log a decision about file processing."""
        if not self.log_base_dir:
            return

        log_path = os.path.join(self.log_base_dir, f"{os.path.splitext(file_name)[0]}_llm_editing.json")
        try:
            with open(log_path, "w") as f:
                json.dump({
                    "file": file_name,
                    "stage": "llm_editing",
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
        return True

    
    def call_llm_api(self, prompt: str, retry_count: int = 0) -> str:
        """Call the LLM API with retry logic."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                self.logger.warning(f"API call failed, retrying ({retry_count + 1}/{self.max_retries}): {str(e)}")
                sleep(2 ** retry_count)  # Exponential backoff
                return self.call_llm_api(prompt, retry_count + 1)
            raise RuntimeError(f"Failed to call LLM API after {self.max_retries} retries: {str(e)}")

    def process_file(self, file_name: str) -> None:
        """Process a single transcript file."""
        base_name = os.path.splitext(file_name)[0]
        input_path = os.path.join(self.transcript_input_folder, f"{base_name}_transcript.json")
        output_path = os.path.join(self.output_folder, f"{base_name}_llm_cleaned.json")

        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Transcript file not found: {input_path}")

            with open(input_path, "r") as f:
                transcript = json.load(f)

            if not self.validate_transcript(transcript):
                raise ValueError("Invalid transcript format")

            # Process each segment
            for i, segment in enumerate(transcript["segments"]):
                prompt = f"Clean and improve the following transcript segment while preserving all visual markers and timing information:\n{segment['text']}"
                try:
                    cleaned_text = self.call_llm_api(prompt)
                    transcript["segments"][i]["text"] = cleaned_text
                    self.logger.info(f"Processed segment {i + 1}/{len(transcript['segments'])}")
                except Exception as e:
                    self.logger.error(f"Failed to process segment {i + 1}: {str(e)}")
                    raise

            # Save cleaned transcript
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
        """Process all transcript files in the input folder."""
        try:
            files = [f for f in os.listdir(self.transcript_input_folder) if f.endswith("_transcript.json")]
            if not files:
                self.logger.warning("No transcript files found to process")
                return

            for transcript_file in files:
                original_name = transcript_file.replace("_transcript.json", ".mp4")
                self.process_file(original_name)

        except Exception as e:
            self.logger.error(f"Failed to process transcript folder: {str(e)}")
            raise
