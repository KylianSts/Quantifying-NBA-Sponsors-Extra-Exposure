"""
YouTube Channel Video URL Filter

This script extracts video URLs from a YouTube channel and filters them based on
upload date and duration criteria. It's optimized for speed using parallel processing
to check multiple videos simultaneously.

Key features:
- Extracts all video IDs from a YouTube channel
- Filters videos by upload date range (start date to end date)
- Filters videos by minimum duration (e.g., only videos longer than 8 minutes)
- Processes video metadata checks in parallel for faster execution
- Saves filtered URLs to text files for downstream processing
- Separate handling for training and test datasets
"""

import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_video_info(video_id: str, start_date: str, end_date: str, min_duration_secs: int) -> str | None:
    """
    Extract video information and check if it matches the filtering criteria.
    Fetches full metadata to validate upload date and duration without downloading the video.
    
    Args:
        video_id: YouTube video ID to check (e.g., 'dQw4w9WgXcQ')
        start_date: Start date for filtering in YYYYMMDD format (e.g., "20240101")
        end_date: End date for filtering in YYYYMMDD format (e.g., "20241231")
        min_duration_secs: Minimum video duration in seconds (e.g., 480 for 8 minutes)
    
    Returns:
        Video URL if it matches all criteria (date range and minimum duration)
        None if video doesn't match criteria or extraction fails
    """
    try:
        # Configure yt-dlp to extract full video metadata without downloading
        ydl_opts = {
            'quiet': True,  # Suppress output messages to avoid console clutter
            'ignoreerrors': True,  # Continue on download errors instead of stopping
            'extract_flat': False,  # Extract full metadata (needed for date and duration)
        }
        
        # Construct YouTube video URL from video ID
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Extract video information without downloading the actual video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if info:
                # Get upload date and duration from video metadata
                upload_date = info.get('upload_date', '')  # Format: YYYYMMDD (e.g., '20241115')
                duration = info.get('duration', 0)  # Duration in seconds
                
                # Check if video matches date range filter (inclusive range)
                if upload_date and start_date <= upload_date <= end_date:
                    # Check if video matches minimum duration filter
                    if duration and duration > min_duration_secs:
                        return video_url  # Video passed all filters
        
        return None  # Video didn't match criteria
    
    except Exception as e:
        # Log error and return None (don't stop processing other videos)
        print(f"Error for {video_id}: {str(e)}")
        return None


def get_video_urls(channel_url: str, outputdir: str, start_date: str, end_date: str, min_duration_secs: int, max_workers: int = 20) -> List[str]:
    """
    Retrieve video URLs from a channel that match date and duration filters.
    Two-phase approach: fast extraction of all video IDs, then parallel filtering.
    
    Args:
        channel_url: YouTube channel URL to extract videos from (e.g., 'https://www.youtube.com/@ChannelName/videos')
        outputdir: Path to text file where filtered URLs will be saved (one URL per line)
        start_date: Start date for filtering in YYYYMMDD format (e.g., "20240101")
        end_date: End date for filtering in YYYYMMDD format (e.g., "20241231")
        min_duration_secs: Minimum video duration in seconds (e.g., 480 for 8 minutes)
        max_workers: Number of videos to check simultaneously (default: 20)
                    Higher values = faster but more network/CPU usage
    
    Returns:
        List of video URLs that match all filtering criteria
    """
    # Phase 1: Configure yt-dlp to quickly get all video IDs from channel
    ydl_opts = {
        'quiet': False,  # Show progress messages for channel extraction
        'ignoreerrors': True,  # Continue on errors (skip unavailable videos)
        'extract_flat': True,  # Only get video IDs (fast extraction, no full metadata)
    }

    # Quickly extract all video IDs from the channel (no metadata fetching yet)
    video_ids = []
    print(f"Extracting video IDs from channel: {channel_url}")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(channel_url, download=False)
        
        # Parse video entries from channel data
        if 'entries' in info_dict:
            for video_entry in info_dict['entries']:
                # Only add videos with valid IDs (skip unavailable/private videos)
                if video_entry and 'id' in video_entry:
                    video_ids.append(video_entry['id'])
    
    print(f"Total videos found: {len(video_ids)}")
    print("Filtering videos by date and duration in parallel...")
    
    # Phase 2: Process videos in parallel to check filtering criteria
    video_urls = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video checking tasks to thread pool
        futures = {
            executor.submit(extract_video_info, video_id, start_date, end_date, min_duration_secs): video_id 
            for video_id in video_ids
        }
        
        # Collect results as they complete (not necessarily in submission order)
        for future in as_completed(futures):
            result = future.result()
            
            # Add URL to list if video passed all filters
            if result:
                video_urls.append(result)
                print(f"Valid video found: {len(video_urls)} total")

    # Save filtered URLs to text file (one URL per line)
    with open(outputdir, 'w') as f:
        for url in video_urls: 
            f.write(url + '\n')

    return video_urls 


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Source channel for NBA highlight videos
    CHANNEL_URL = "https://www.youtube.com/@TheGametimeHighlights/videos"
    
    # Minimum video duration filter (8 minutes = 480 seconds)
    # This filters out short clips and only keeps full highlight reels
    MINIMUM_DURATION = 8 * 60
    
    # Date range for training data: 2024/2025 NBA season highlights
    START_DATE_TRAIN = "20241101"  # November 1, 2024
    END_DATE_TRAIN = "20250501"    # May 1, 2025
    
    # Date range for test data: beginning of 2025/2026 NBA season
    START_DATE_TEST = "20251020"   # October 20, 2025
    END_DATE_TEST = "20251201"     # December 1, 2025
    
    print("="*60)
    print("COLLECTING TRAINING SET URLS")
    print("="*60)
    
    # Get training set URLs (videos from Nov 2024 to May 2025)
    train_urls = get_video_urls(
        channel_url=CHANNEL_URL,
        outputdir='Data/urls/train_urls.txt',
        start_date=START_DATE_TRAIN,
        end_date=END_DATE_TRAIN,
        min_duration_secs=MINIMUM_DURATION,
        max_workers=32  # Number of parallel workers for faster processing
    )
    
    print(f"\nTraining set complete: {len(train_urls)} URLs saved to 'Data/urls/train_urls.txt'\n")

    print("="*60)
    print("COLLECTING TEST SET URLS")
    print("="*60)
    
    # Get test set URLs (videos from Oct 2025 to Dec 2025)
    test_urls = get_video_urls(
        channel_url=CHANNEL_URL,
        outputdir='Data/urls/test_urls.txt',
        start_date=START_DATE_TEST,
        end_date=END_DATE_TEST,
        min_duration_secs=MINIMUM_DURATION,
        max_workers=32
    )
    
    print(f"\nTest set complete: {len(test_urls)} URLs saved to 'Data/urls/test_urls.txt'\n")
    
    # Display final summary
    print("="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"Training URLs: {len(train_urls)}")
    print(f"Test URLs: {len(test_urls)}")
    print(f"Total URLs: {len(train_urls) + len(test_urls)}")
    print("="*60)