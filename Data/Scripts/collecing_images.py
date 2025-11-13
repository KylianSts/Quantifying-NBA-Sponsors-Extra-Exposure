"""
YouTube Video Frame Extractor (Parallel & Quality-Filtered Version)

This script automates the extraction of high-quality frames from YouTube videos
for use in machine learning datasets. It downloads videos, analyzes frame sharpness,
and extracts only the clearest frames to ensure dataset quality.

Key features:
- Downloads videos from YouTube URLs using yt-dlp
- Extracts random frames and filters by sharpness using Laplacian variance
- Processes multiple videos in parallel for faster execution
- Automatically cleans up downloaded videos to save disk space
- Saves frames as PNG files with unique identifiers
"""

import yt_dlp
import cv2
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def process_single_video(url: str, output_dir: str, num_frames: int, sharpness_threshold: float) -> str:
    """
    Download a single video and extract sharp frames from it (worker function for parallelization).
    Uses Laplacian variance to measure frame sharpness and only saves high-quality frames.
    
    Args:
        url: YouTube video URL to download
        output_dir: Directory where extracted frames will be saved
        num_frames: Number of sharp frames to extract from the video
        sharpness_threshold: Minimum Laplacian variance value for a frame to be considered sharp
                           (typical values: 100-200, higher = sharper)
    
    Returns:
        Status message indicating completion or error (e.g., "Completed: video_id" or "Error for url: ...")
    """
    try:
        # Configure yt-dlp to download highest quality video
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),  # Output template for filename
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Best video+audio quality
            'extractor_args': {'youtube': {'player_client': ['android']}},  # Use android client to avoid restrictions
           # 'cookiesfrombrowser': ('firefox',)  # Uncomment to use browser cookies if needed
        }
        
        # Download the video and get metadata
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            video_id = info.get('id', 'unknown')  # Extract video ID for frame naming
        
        # Open video file with OpenCV for frame extraction
        cap = cv2.VideoCapture(filename)

        # Get total number of frames in the video to enable random sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize counters for frame extraction loop
        saved_frames_count = 0  # Number of sharp frames successfully saved
        max_attempts = num_frames * 5  # Allow multiple attempts to find sharp frames (5x buffer)
        attempts = 0  # Current attempt counter

        # Extract random sharp frames until we get enough or reach max attempts
        while saved_frames_count < num_frames and attempts < max_attempts:
            # Select a random frame from the video (uniform distribution)
            frame_id = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # Seek to the random frame
            success, frame = cap.read()

            if success:
                # Convert frame to grayscale for sharpness analysis (Laplacian works on single channel)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate sharpness using Laplacian variance (higher = sharper)
                # Laplacian detects edges; high variance means many sharp edges
                laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
                
                # Only save frame if it meets the sharpness threshold (filters out blurry frames)
                if laplacian_var > sharpness_threshold:
                    saved_frames_count += 1
                    frame_filename = os.path.join(output_dir, f"{video_id}_frame_{saved_frames_count}.png")
                    cv2.imwrite(frame_filename, frame)  # Save as PNG (lossless quality)
            
            attempts += 1
    
        # Clean up resources
        cap.release()  # Release video file handle
        os.remove(filename)  # Delete video file after frame extraction to save space
        
        return f"Completed: {video_id}"
    
    except Exception as e:
        # Return error message if video processing fails
        return f"Error for {url}: {str(e)}"


def download_and_extract_frames(urls: list, output_dir: str, num_frames: int, sharpness_threshold: float = 150.0, max_workers: int = 20) -> None:
    """
    Download YouTube videos and extract random sharp frames in parallel.
    Videos are processed simultaneously to speed up extraction across multiple URLs.
    
    Args:
        urls: List of YouTube video URLs to process
        output_dir: Directory where extracted frames will be saved (created if doesn't exist)
        num_frames: Number of sharp frames to extract from each video
        sharpness_threshold: Minimum Laplacian variance value for frame sharpness (default: 150.0)
                           Higher values = stricter quality filter
        max_workers: Number of videos to process simultaneously (default: 20)
                    Higher values = faster but more CPU/network usage
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process videos in parallel using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video processing tasks to the thread pool
        futures = {
            executor.submit(process_single_video, url, output_dir, num_frames, sharpness_threshold): url 
            for url in urls
        }
        
        # Process results as they complete (not necessarily in submission order)
        for future in as_completed(futures):
            url = futures[future]
            try:
                # Retrieve result from completed task and print status
                result = future.result()
                print(result)
            except Exception as e:
                # Handle errors from failed video processing tasks
                print(f"Error for {url}: {str(e)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Load training video URLs from file (one URL per line)
    with open('Data/urls/train_urls.txt', 'r') as f:
        train_urls = [line.strip() for line in f.readlines()]

    # Extract frames from training videos (highlights from 2024/2025 seasons)
    download_and_extract_frames(urls=train_urls,
                                output_dir="Data/images/train_images",
                                num_frames=5,  # Extract 5 frames per video
                                sharpness_threshold=150,  # Minimum sharpness score (filters blurry frames)
                                max_workers=32)  # Number of videos processed simultaneously
    

    # Load test video URLs from file (separate dataset for model evaluation)
    with open('Data/urls/test_urls.txt', 'r') as f:
        test_urls = [line.strip() for line in f.readlines()]

    # Extract frames from test videos (same parameters as training for consistency)
    download_and_extract_frames(urls=test_urls,
                                output_dir="Data/images/test_images",
                                num_frames=5,
                                sharpness_threshold=150,
                                max_workers=32)