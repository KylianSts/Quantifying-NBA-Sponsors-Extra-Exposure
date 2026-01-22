"""
YouTube Video Frame Extractor (Final Robust Version)

This script is an enhanced version of the frame extractor that addresses common issues
encountered when downloading and processing YouTube videos at scale. It implements
robust error handling, quality preservation, and anti-ban measures.

Key improvements over the basic version:
- Robust file detection: Automatically handles .mkv, .webm, and .mp4 formats
- Quality preservation: Uses Safari client with cookie authentication for best quality
- Filename safety: Implements restrictfilenames to prevent Windows path errors
- Rate limiting: Sequential processing with random delays to avoid HTTP 403 bans
- Enhanced error handling: Proper resource cleanup and retry mechanisms

Use case:
This script is ideal for building large-scale machine learning datasets from YouTube
content while maintaining high video quality and avoiding platform restrictions.
"""

import yt_dlp
import cv2
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def process_single_video(url: str, output_dir: str, num_frames: int, sharpness_threshold: float) -> str:
    """
    Download a single video and extract sharp frames from it with robust error handling.
    
    This function implements several improvements over basic video processing:
    - Automatic file format detection (handles mkv/webm/mp4 variations)
    - Quality-first download strategy using Safari client + browser cookies
    - Safe filename handling for cross-platform compatibility
    - Random delay after processing to avoid rate limiting (anti-ban measure)
    
    Args:
        url: YouTube video URL to download and process
        output_dir: Directory where extracted frames will be saved
        num_frames: Number of sharp frames to extract from the video
        sharpness_threshold: Minimum Laplacian variance for frame quality
                           (typical values: 100-300, higher = stricter quality filter)
    
    Returns:
        Status message indicating success with frame count or detailed error information
    """
    filename = None  # Track downloaded file for cleanup
    cap = None       # Track video capture object for proper release
    
    try:
        # Configure yt-dlp with robust options for quality and compatibility
        ydl_opts = {
            # Output template for downloaded files
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            
            # Filename safety: Remove special characters to avoid Windows path errors
            # Converts characters like ":", "?", "*" to safe alternatives
            'restrictfilenames': True,
            
            # Quality prioritization: Always prefer highest resolution available
            # Tries to get best video+audio, falls back to best single stream
            'format': 'bestvideo+bestaudio/best',
            'format_sort': ['res:2160', 'res:1080', 'res:720', 'res'],  # Prefer 4K > 1080p > 720p
            
            # Output format: Use MKV container for better quality preservation
            # MKV supports high-quality video codecs better than MP4
            'merge_output_format': 'mkv',
            
            # Hybrid configuration: Combine cookies + Safari client for best results
            # Safari client helps bypass some YouTube restrictions
            # 'actual' player_js_version ensures we use the latest JavaScript player
            'extractor_args': {'youtube': {
                'player_client': ['default', 'web_safari'],
                'player_js_version': ['actual']
            }},
            
            # Cookie authentication: Use browser cookies to access higher quality streams
            # IMPORTANT: Close Firefox completely before running this script
            # Change 'firefox' to 'chrome', 'edge', or 'safari' if you use a different browser
            'cookiesfrombrowser': ('firefox',),
            
            # Suppress verbose output for cleaner logs
            'quiet': True,
            'no_warnings': True,
        }
        
        # Download video and extract metadata
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Download the video and get its metadata
            info = ydl.extract_info(url, download=True)
            
            # Get the theoretical filename (may not match actual file extension)
            temp_filename = ydl.prepare_filename(info)
            base_name = os.path.splitext(temp_filename)[0]  # Remove extension for searching
            
            # Robust file detection: yt-dlp may save as mkv/webm/mp4 depending on format
            # We need to find which extension was actually used
            found_file = None
            possible_extensions = ['mkv', 'webm', 'mp4', 'm4a']
            
            # Strategy 1: Check if the prepared filename exists exactly as returned
            if os.path.exists(temp_filename):
                found_file = temp_filename
            # Strategy 2: Try common video extensions to find the actual file
            else:
                for ext in possible_extensions:
                    test_path = f"{base_name}.{ext}"
                    if os.path.exists(test_path):
                        found_file = test_path
                        break
            
            # If file still not found, return detailed error for debugging
            if not found_file:
                return f"Error: File not found on disk after download. Base: {base_name}"
            
            filename = found_file
            video_id = info.get('id', 'unknown')  # Extract video ID for frame naming

        # Open video file with OpenCV for frame extraction
        cap = cv2.VideoCapture(filename)
        
        # Verify that OpenCV can open the video file
        if not cap.isOpened():
            return f"Error: Could not open video file {filename}"

        # Log video quality information for verification
        # This helps confirm we're getting the quality we requested
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Processing {video_id} | File: {os.path.basename(filename)} | Res: {width}x{height}")

        # Get total frame count for random sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return f"Error: Video has 0 frames {url}"

        # Frame extraction loop with increased retry budget
        saved_frames_count = 0  # Successful sharp frames saved
        max_attempts = num_frames * 15  # Allow 15x attempts to find sharp frames (more generous than 5x)
        attempts = 0  # Current attempt counter

        # Extract random sharp frames until quota met or max attempts reached
        while saved_frames_count < num_frames and attempts < max_attempts:
            # Select random frame from video
            frame_id = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # Seek to random position
            success, frame = cap.read()

            if success:
                # Convert to grayscale for sharpness analysis
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate sharpness using Laplacian variance
                # Higher variance indicates more edges = sharper image
                laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
                
                # Only save frame if it meets quality threshold
                if laplacian_var > sharpness_threshold:
                    saved_frames_count += 1
                    frame_filename = os.path.join(output_dir, f"{video_id}_frame_{saved_frames_count}.png")
                    cv2.imwrite(frame_filename, frame)  # Save as PNG for lossless quality
            
            attempts += 1
    
        # Resource cleanup
        cap.release()  # Release video file handle
        
        # Delete video file after extraction to save disk space
        # Comment out these lines if you want to keep the downloaded videos
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
            except PermissionError:
                # Windows sometimes locks files briefly after release
                time.sleep(1)  # Wait for file handle to be released
                try: 
                    os.remove(filename)
                except: 
                    pass  # If still locked, skip deletion
        
        # === ANTI-BAN MECHANISM ===
        # Random delay between 5-15 seconds to avoid triggering YouTube's rate limits
        # YouTube may block or throttle if too many requests come too quickly
        sleep_time = random.uniform(5, 15)
        print(f"Extraction completed for {video_id}. Pausing for {sleep_time:.1f}s...")
        time.sleep(sleep_time)

        return f"Completed: {video_id} ({saved_frames_count}/{num_frames} frames saved)"
    
    except Exception as e:
        # Cleanup in case of error
        if cap: 
            cap.release()
        
        # Attempt to remove partially downloaded file
        if filename and os.path.exists(filename):
            try: 
                os.remove(filename)
            except: 
                pass  # Ignore cleanup errors
        
        return f"Error for {url}: {str(e)}"


def download_and_extract_frames(urls: list, output_dir: str, num_frames: int, 
                                sharpness_threshold: float = 300.0, max_workers: int = 1) -> None:
    """
    Download YouTube videos and extract random sharp frames with rate limiting.
    
    Args:
        urls: List of YouTube video URLs to process
        output_dir: Directory where extracted frames will be saved (created if doesn't exist)
        num_frames: Number of sharp frames to extract from each video
        sharpness_threshold: Minimum Laplacian variance for frame quality (default: 300.0)
                           Higher values = stricter quality requirements
        max_workers: Number of videos to process simultaneously (default: 1)
    
    Returns:
        None (prints progress and results to console)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process videos sequentially or with limited parallelism
    # max_workers=1 is CRITICAL to avoid HTTP 403 errors when using cookies
    # YouTube's API detects multiple simultaneous authenticated requests as suspicious
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video processing tasks to the thread pool
        futures = {
            executor.submit(process_single_video, url, output_dir, num_frames, sharpness_threshold): url 
            for url in urls
        }
        
        # Process completed tasks and display results
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error for {url}: {str(e)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Define sharpness threshold for high-quality frames
    HIGH_QUALITY_THRESHOLD = 200.0 

    try:
        # === TRAINING SET EXTRACTION ===
        # Load training video URLs from file
        with open('Data/urls/train_urls.txt', 'r') as f:
            # Strip whitespace and filter out empty lines
            train_urls = [line.strip() for line in f.readlines() if line.strip()]    
        print(f"Starting extraction for {len(train_urls)} training videos...")
        
        download_and_extract_frames(
            urls=train_urls,
            output_dir="Data/images/train_images_quality_2",
            num_frames=50,  # Extract 10 frames per video
            sharpness_threshold=HIGH_QUALITY_THRESHOLD,
            max_workers=12  # Process videos in parallel
        ) 

        # === TEST SET EXTRACTION ===
        # Load test video URLs from separate file (for model evaluation)
        with open('Data/urls/test_urls.txt', 'r') as f:
            test_urls = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Starting extraction for {len(test_urls)} test videos...")
        download_and_extract_frames(
            urls=test_urls, 
            output_dir="Data/images/test_images_quality", 
            num_frames=10, 
            sharpness_threshold=HIGH_QUALITY_THRESHOLD, 
            max_workers=12
        )
                                    
    except FileNotFoundError:
        # Provide helpful error message if URL files are missing
        print("Error: URL files not found. Please verify the path 'Data/urls/train_urls.txt' exists")
        print("Expected file structure:")
        print("  Data/")
        print("    urls/")
        print("      train_urls.txt")
        print("      test_urls.txt")