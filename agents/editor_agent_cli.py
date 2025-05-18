# agents/editor_agent.py

import os
import json
import logging
import subprocess
from typing import Dict, List
from config import LOG_FORMAT, LOG_LEVEL

class LLMEditingAgent:
    def __init__(self, transcript_input_folder, output_folder, model_name="llama3", log_base_dir=None, timeout=300):
        self.transcript_input_folder = transcript_input_folder
        self.output_folder = output_folder
        self.model_name = model_name  # Name used by Ollama, e.g., 'llama3'
        self.log_base_dir = log_base_dir
        self.timeout = timeout

        os.makedirs(self.output_folder, exist_ok=True)
        if self.log_base_dir:
            os.makedirs(self.log_base_dir, exist_ok=True)
            # Use a file handler instead of basicConfig
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(getattr(logging, LOG_LEVEL))
            handler = logging.FileHandler(os.path.join(self.log_base_dir, "llm_editing.log"))
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
        else:
            self.logger = logging.getLogger(__name__)

    def log_decision(self, file_name, reason):
        if not self.log_base_dir:
            return

        log_path = os.path.join(self.log_base_dir, f"{os.path.splitext(file_name)[0]}_llm_editing.json")
        with open(log_path, "w") as f:
            json.dump({
                "file": file_name,
                "stage": "llm_editing",
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
        return True

    def run_llm_editing(self, transcript: Dict) -> Dict:
        self.logger.info("Running LLM analysis on transcript...")

        # Structured prompt with clear instructions
        input_prompt = (
            "You are an AI video editor. Your task is to clean and optimize the transcript for video editing.\n\n"
            "Instructions:\n"
            "1. Remove duplicate phrases and redundant content\n"
            "2. Remove off-topic content and filler words\n"
            "3. Remove silent/empty space indicators\n"
            "4. Preserve segments containing 'start visual' and 'end visual'\n"
            "5. Maintain the original timing information\n"
            "6. Return the result in the same JSON format as input\n\n"
            f"Transcript to process: {json.dumps(transcript['segments'])}\n\n"
            "Return only the cleaned transcript in JSON format, maintaining the same structure."
        )

        try:
            # Call Ollama with the provided model
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=input_prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Ollama process failed: {result.stderr.decode('utf-8')}")

            try:
                cleaned_transcript = json.loads(result.stdout.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")

            # Validate the cleaned transcript
            if not self.validate_transcript(cleaned_transcript):
                raise ValueError("Cleaned transcript does not match expected format")

            return cleaned_transcript

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"LLM processing timed out after {self.timeout} seconds")
        except Exception as e:
            self.logger.error(f"LLM editing failed: {str(e)}")
            raise

    def process_file(self, file_name):
        base_name = os.path.splitext(file_name)[0]
        transcript_path = os.path.join(self.transcript_input_folder, f"{base_name}_transcript.json")
        output_path = os.path.join(self.output_folder, f"{base_name}_llm_cleaned.json")

        try:
            if not os.path.exists(transcript_path):
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

            with open(transcript_path, "r") as f:
                transcript = json.load(f)

            # Validate input transcript
            if not self.validate_transcript(transcript):
                raise ValueError("Input transcript does not match expected format")

            cleaned_transcript = self.run_llm_editing(transcript)

            with open(output_path, "w") as f:
                json.dump(cleaned_transcript, f, indent=2)

            self.log_decision(file_name, "success")
            self.logger.info(f"Successfully edited transcript for {file_name}")

        except Exception as e:
            error_msg = f"failed - {str(e)}"
            self.logger.error(f"[ERROR] Processing {file_name}: {error_msg}")
            self.log_decision(file_name, error_msg)

    def run(self):
        files = [f for f in os.listdir(self.transcript_input_folder) if f.endswith("_transcript.json")]
        for transcript_file in files:
            original_name = transcript_file.replace("_transcript.json", ".mp4")
            self.process_file(original_name)
