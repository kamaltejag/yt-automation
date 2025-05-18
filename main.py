# main.py

from agents.denoise_agent import DenoiseAgent
from agents.transcriber_agent import TranscriberAgent
from agents.editor_agent import LLMEditingAgent
# from agents.segment_audio_cleaner_agent import SegmentAudioCleaner  # Keep import but commented
from agents.video_segment_editor_agent import VideoSegmentEditor
from agents.timeline_generator_agent import TimelineGenerator
import argparse
import os
import logging
import sys
import glob
import shutil
import requests
from typing import Optional
from config import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    DEFAULT_OUTPUT_DIR, DEFAULT_LOG_DIR, DEFAULT_MODEL_PATH,
    OUTPUT_DIRS, LOG_FORMAT, LOG_LEVEL
)

def clean_directory(directory: str, preserve_structure: bool = False) -> None:
    """Clean all files in a directory.
    
    Args:
        directory: Path to the directory to clean
        preserve_structure: If True, keeps the directory structure and only removes files
                          If False, removes the entire directory and recreates it
    """
    if os.path.exists(directory):
        try:
            if preserve_structure:
                # Remove all files in the directory
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        os.makedirs(item_path)  # Recreate the empty directory
                print(f"Cleaned directory while preserving structure: {directory}")
            else:
                # Remove entire directory and recreate it
                shutil.rmtree(directory)
                os.makedirs(directory)
                print(f"Cleaned and recreated directory: {directory}")
        except Exception as e:
            print(f"Warning: Failed to clean directory {directory}: {str(e)}")
    os.makedirs(directory, exist_ok=True)

def preprocess_directories(output_dir: str, log_dir: str) -> None:
    """Preprocess all directories before starting the pipeline.
    
    This function:
    1. Cleans the logs directory (removes everything)
    2. Cleans all output directories (preserves structure)
    """
    # Clean logs directory (remove everything)
    clean_directory(log_dir, preserve_structure=False)
    
    # Clean output directories (preserve structure)
    for dir_name in OUTPUT_DIRS.values():
        dir_path = os.path.join(output_dir, dir_name)
        clean_directory(dir_path, preserve_structure=True)

def check_ollama_availability() -> bool:
    """Check if Ollama server is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    return logger

def create_output_dirs(base_dir: str) -> dict:
    """Create output directories for each stage."""
    dirs = {}
    for key, dir_name in OUTPUT_DIRS.items():
        dirs[key] = os.path.join(base_dir, dir_name)
        os.makedirs(dirs[key], exist_ok=True)
    return dirs

def process_video(video_path: str, output_dir: str, log_dir: str, model_path: str, logger: logging.Logger) -> None:
    """Process a single video file through the pipeline."""
    pipeline_success = True
    try:
        # Create output directories
        dirs = create_output_dirs(output_dir)
        
        # Initialize agents
        denoise_agent = DenoiseAgent(
            input_folder=os.path.dirname(video_path),  # Directory containing the video
            audio_output=dirs["denoised"],
            video_output=dirs["denoised"],
            model_path=os.path.abspath(model_path),  # Use absolute path for model
            log_base_dir=log_dir
        )
        
        transcriber = TranscriberAgent(
            audio_input_folder=dirs["denoised"],
            output_folder=dirs["transcripts"],
            log_base_dir=log_dir
        )
        
        editor = LLMEditingAgent(
            transcript_input_folder=dirs["transcripts"],
            output_folder=dirs["edited"],
            model_name=OLLAMA_MODEL,
            log_base_dir=log_dir,
            timeout=OLLAMA_TIMEOUT
        )
        
        video_editor = VideoSegmentEditor(
            video_input_folder=dirs["denoised"],  # Use denoised video directly
            transcript_folder=dirs["edited"],
            output_folder=dirs["edited_segments"],
            log_base_dir=log_dir
        )
        
        timeline_generator = TimelineGenerator(
            transcript_folder=dirs["edited"],
            video_folder=dirs["edited_segments"],
            output_folder=dirs["timelines"],
            log_base_dir=log_dir
        )
        
        # Run pipeline
        logger.info(f"Starting pipeline for {video_path}")
        
        # Step 1: Denoise the video
        logger.info("Step 1: Denoising video")
        video_filename = os.path.basename(video_path)  # Get just the filename
        logger.info(f"Processing video file: {video_filename}")
        logger.info(f"Using noise reduction model: {os.path.abspath(model_path)}")
        denoise_agent.process_file(video_filename)
        
        # Step 2: Transcribe the audio
        logger.info("Step 2: Transcribing audio")
        transcriber.run()
        
        # Step 3: Edit the transcript
        logger.info("Step 3: Editing transcript")
        if not check_ollama_availability():
            error_msg = f"Ollama server is not running at {OLLAMA_URL}. Please start Ollama before running the pipeline."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        editor.run()
        
        # Step 4: Edit video segments
        logger.info("Step 4: Editing video segments")
        video_editor.run()
        
        # Step 5: Generate timeline
        logger.info("Step 5: Generating timeline")
        timeline_generator.run()
        
        logger.info(f"Pipeline completed successfully for {video_path}")
        
    except Exception as e:
        pipeline_success = False
        logger.error(f"Pipeline failed for {video_path}: {str(e)}")
        raise
    finally:
        if not pipeline_success:
            logger.error("Pipeline failed - some steps may have completed but others failed")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Log directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to noise reduction model")
    args = parser.parse_args()
    
    # Preprocess all directories
    preprocess_directories(args.output_dir, args.log_dir)
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    try:
        if os.path.isfile(args.input):
            # Process single file
            if not args.input.endswith(".mp4"):
                logger.error("Input file must be an MP4 file")
                sys.exit(1)
            process_video(args.input, args.output_dir, args.log_dir, args.model_path, logger)
        else:
            # Process directory
            mp4_files = glob.glob(os.path.join(args.input, "*.mp4"))
            if not mp4_files:
                logger.error("No MP4 files found in input directory")
                sys.exit(1)
            
            for video_file in mp4_files:
                try:
                    process_video(video_file, args.output_dir, args.log_dir, args.model_path, logger)
                except Exception as e:
                    logger.error(f"Failed to process {video_file}: {str(e)}")
                    continue
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

    
