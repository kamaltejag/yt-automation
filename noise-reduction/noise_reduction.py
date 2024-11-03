import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def convert_mp4_to_wav(input_mp4, output_wav):
    """Convert MP4 file to WAV format using FFmpeg."""
    try:
        subprocess.run(["ffmpeg", "-i", input_mp4,
                       output_wav, "-y"], check=True)
        logging.info(f"Converted {input_mp4} to WAV format as {output_wav}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {input_mp4} to WAV: {e}")


def create_noise_profile_from_wav(noise_sample_path, profile_path):
    """Creates a noise profile from a WAV file using SoX."""
    try:
        subprocess.run(
            ["sox", noise_sample_path,
                "-n", "noiseprof", profile_path],
            check=True
        )
        logging.info(f"Noise profile created at {profile_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating noise profile from {
                      noise_sample_path}: {e}")


def create_noise_profile_from_wav_automatically(input_wav, profile_path, start_time, end_time):
    """Creates a noise profile from a specific segment of a WAV file using SoX."""
    try:
        # Extract the specified segment
        segment_wav = f"temp_segment.wav"
        subprocess.run(
            ["sox", input_wav, segment_wav, "trim", str(
                start_time), str(end_time - start_time)],
            check=True
        )
        logging.info(f"Extracted segment from {start_time}s to {
                     end_time}s as {segment_wav}")

        # Create a noise profile from the extracted segment
        subprocess.run(
            ["sox", segment_wav, "-n", "noiseprof", profile_path],
            check=True
        )
        logging.info(f"Noise profile created at {profile_path}")

        # Clean up the temporary segment
        os.remove(segment_wav)
        logging.info(f"Removed temporary segment file {segment_wav}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating noise profile from {input_wav}: {e}")


def apply_noise_reduction_sox(input_wav, output_wav, profile_path, reduction_level=0.21):
    """Apply noise reduction to a WAV file using SoX."""
    try:
        subprocess.run(["sox", input_wav, output_wav, "noisered",
                       profile_path, str(reduction_level)], check=True)
        logging.info(f"Applied noise reduction on {
                     input_wav}, saved as {output_wav}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying noise reduction on {input_wav}: {e}")


def apply_noise_reduction_arnndn(input_wav, output_wav, model_path):
    """Apply noise reduction to a WAV file using FFmpeg's arnndn filter."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_wav, "-af",
                f"arnndn={model_path}", output_wav, "-y"],
            check=True
        )
        print(f"Applied noise reduction using ARNNDN on {
              input_wav}, saved as {output_wav}")
    except subprocess.CalledProcessError as e:
        print(f"Error applying noise reduction on {input_wav}:", e)


def convert_wav_to_mp4(input_wav, output_mp4, original_video):
    """Converts a WAV file back to MP4 format by combining it with the original video stream."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", original_video, "-i", input_wav,
                "-map", "0:v:0", "-map", "1:a:0",  # Map video from original, audio from WAV
                "-c:v", "copy",                   # Copy the video stream without re-encoding
                "-c:a", "aac",                    # Encode the audio to AAC
                "-b:a", "192k",                   # Set audio bitrate
                "-ar", "48000",                   # Set audio sample rate
                "-ac", "2",                       # Set audio channels to stereo
                output_mp4, "-y"
            ],
            check=True
        )
        logging.info(
            f"Converted {input_wav} back to MP4 format as {output_mp4}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {input_wav} to MP4: {e}")


def process_mp4_files(input_folder, audio_output_folder, video_output_folder, noise_sample_path, start_time=0, end_time=5, use_noise_sample=True, reduction_level=0.21, cleanup=True):
    """Processes all .mp4 files in the input folder:
    - Converts them to WAV
    - Applies noise reduction using a dynamically created noise profile
    - Converts them back to MP4 with the original video stream
    """
    os.makedirs(audio_output_folder, exist_ok=True)
    os.makedirs(video_output_folder, exist_ok=True)

    # Process each MP4 file in the input folder
    mp4_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    total_files = len(mp4_files)

    for index, file_name in enumerate(mp4_files, start=1):
        input_mp4 = os.path.join(input_folder, file_name)
        temp_wav = os.path.join(
            audio_output_folder, f"{os.path.splitext(file_name)[0]}_temp.wav")
        output_wav = os.path.join(
            audio_output_folder, f"{os.path.splitext(file_name)[0]}_denoised.wav")
        output_mp4 = os.path.join(video_output_folder, file_name)
        noise_profile_path = os.path.join(
            audio_output_folder, f"{os.path.splitext(file_name)[0]}_noise.prof")

        logging.info(f"Processing file {index}/{total_files}: {file_name}")

        # Step 1: Convert MP4 to WAV
        convert_mp4_to_wav(input_mp4, temp_wav)

        # Step 2: Create a noise profile from the temporary WAV file
        if use_noise_sample:
            create_noise_profile_from_wav(
                noise_sample_path, noise_profile_path)
        else:
            create_noise_profile_from_wav_automatically(
                temp_wav, noise_profile_path, start_time, end_time)

        # Step 3: Apply noise reduction to the WAV file
        # apply_noise_reduction_sox(temp_wav, output_wav,
        #                           noise_profile_path, reduction_level)
        apply_noise_reduction_arnndn(
            temp_wav, output_wav, "noise-reduction/models/lq.rnnn")

        # Step 3.1: Verify that the noise-reduced WAV file was created
        if not os.path.exists(output_wav):
            logging.error(
                f"Error: Noise-reduced WAV file {output_wav} not found.")
            continue  # Skip to the next file if this one failed

        # Step 4: Convert back to MP4 with the original video stream
        convert_wav_to_mp4(output_wav, output_mp4, input_mp4)

        # Step 5: Cleanup temporary files if enabled
        if cleanup:
            os.remove(temp_wav)
            os.remove(output_wav)
            os.remove(noise_profile_path)
            logging.info(f"Cleaned up temporary files for {file_name}")


# Example usage
input_folder = "raw_videos"
audio_output_folder = "output/audio"
video_output_folder = "output/video"

# Replace with path to your noise sample
noise_sample_path = "noise-reduction/noise_sample/noise_sample.wav"

start_time = 2   # Start time in seconds
end_time = 7     # End time in seconds

process_mp4_files(input_folder, audio_output_folder, video_output_folder, noise_sample_path,
                  start_time, end_time)
