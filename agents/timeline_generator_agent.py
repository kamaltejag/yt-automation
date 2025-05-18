import os
import json
import uuid
import logging
import xml.etree.ElementTree as ET
import ffmpeg
from typing import Dict, List, Tuple, Optional

class TimelineGenerator:
    def __init__(self, transcript_folder: str, video_folder: str, output_folder: str, log_base_dir: Optional[str] = None):
        self.transcript_folder = transcript_folder
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.log_base_dir = log_base_dir

        os.makedirs(self.output_folder, exist_ok=True)
        if self.log_base_dir:
            os.makedirs(self.log_base_dir, exist_ok=True)
            # Use a file handler instead of basicConfig
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.log_base_dir, "timeline_generator.log"))
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
        else:
            self.logger = logging.getLogger(__name__)

    def log_decision(self, file_name: str, reason: str) -> None:
        """Log a decision about file processing."""
        if not self.log_base_dir:
            return

        log_path = os.path.join(self.log_base_dir, f"{os.path.splitext(file_name)[0]}_timeline.json")
        try:
            with open(log_path, "w") as f:
                json.dump({
                    "file": file_name,
                    "stage": "timeline_generation",
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

    def get_video_metadata(self, video_path: str) -> Tuple[int, int, float]:
        """Extract resolution and frame rate from video file using ffmpeg."""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if not video_stream:
                raise RuntimeError("No video stream found in file")

            width = int(video_stream['width'])
            height = int(video_stream['height'])

            r_frame_rate = video_stream['r_frame_rate']  # e.g. "25/1"
            num, denom = map(int, r_frame_rate.split('/'))
            frame_rate = num / denom

            return width, height, frame_rate
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract video metadata: {str(e)}")

    def create_fcpxml(self, transcript_segments: List[Dict], video_filename: str, output_path: str) -> None:
        """Create FCPXML timeline file."""
        try:
            width, height, frame_rate = self.get_video_metadata(video_filename)
            duration_factor = 1 / frame_rate
            frame_duration = f"{int(duration_factor * 100000)}/100000s"

            def time_to_seconds_string(seconds: float) -> str:
                return f"{seconds:.3f}s"

            media_id = str(uuid.uuid4())
            fcpxml = ET.Element("fcpxml", version="1.10")
            resources = ET.SubElement(fcpxml, "resources")

            # Add format
            ET.SubElement(resources, "format", {
                "id": "r1",
                "name": f"AutoFormat{width}x{height}@{int(frame_rate)}",
                "frameDuration": frame_duration,
                "width": str(width),
                "height": str(height)
            })

            # Add asset
            ET.SubElement(resources, "asset", {
                "id": media_id,
                "src": f"file://{os.path.abspath(video_filename)}",
                "start": "0s",
                "hasVideo": "1",
                "hasAudio": "1",
                "format": "r1"
            })

            # Create timeline structure
            library = ET.SubElement(fcpxml, "library")
            event = ET.SubElement(library, "event", {"name": "Auto Edited"})
            project = ET.SubElement(event, "project", {"name": "Cleaned Timeline"})
            sequence = ET.SubElement(project, "sequence", {
                "duration": "3600s",
                "format": "r1",
                "tcStart": "0s",
                "tcFormat": "NDF"
            })
            spine = ET.SubElement(sequence, "spine")

            # Add clips
            offset = 0.0
            for i, seg in enumerate(transcript_segments):
                start = seg["start"]
                end = seg["end"]
                dur = end - start

                ET.SubElement(spine, "clip", {
                    "name": f"Segment {i + 1}",
                    "ref": media_id,
                    "offset": time_to_seconds_string(offset),
                    "start": time_to_seconds_string(start),
                    "duration": time_to_seconds_string(dur)
                })

                offset += dur

            # Write XML file
            tree = ET.ElementTree(fcpxml)
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            self.logger.info(f"Timeline XML written to: {output_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to create FCPXML: {str(e)}")

    def process_file(self, file_name: str) -> None:
        """Process a single file to generate its timeline."""
        base_name = os.path.splitext(file_name)[0]
        transcript_path = os.path.join(self.transcript_folder, f"{base_name}_llm_cleaned.json")
        video_path = os.path.join(self.video_folder, file_name)
        output_path = os.path.join(self.output_folder, f"{base_name}_timeline.xml")

        try:
            if not os.path.exists(transcript_path):
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            with open(transcript_path, "r") as f:
                transcript = json.load(f)

            if not self.validate_transcript(transcript):
                raise ValueError("Invalid transcript format")

            self.create_fcpxml(transcript["segments"], video_path, output_path)
            
            self.log_decision(file_name, "success")
            self.logger.info(f"Successfully generated timeline for {file_name}")

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
        """Process all files in the transcript folder."""
        try:
            files = [f for f in os.listdir(self.transcript_folder) if f.endswith("_llm_cleaned.json")]
            if not files:
                self.logger.warning("No transcript files found to process")
                return

            for transcript_file in files:
                original_name = transcript_file.replace("_llm_cleaned.json", ".mp4")
                self.process_file(original_name)

        except Exception as e:
            self.logger.error(f"Failed to process transcript folder: {str(e)}")
            raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FCPXML timeline for Premiere Pro.")
    parser.add_argument("--transcript", required=True, help="Path to cleaned transcript JSON.")
    parser.add_argument("--video", required=True, help="Path to original video file.")
    parser.add_argument("--output", default="./output", help="Directory to save the XML file.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    generator = TimelineGenerator(args.transcript, args.video, args.output)
    generator.run()
