import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_video_info(video_id: str, start_date: str, end_date: str, min_duration_secs: int) -> str | None:
    """
    Extract video information and check if it matches the filtering criteria.
    
    Args:
        video_id: YouTube video ID to check
        start_date: Start date for filtering in YYYYMMDD format (e.g., "20240101")
        end_date: End date for filtering in YYYYMMDD format (e.g., "20241231")
        min_duration_secs: Minimum video duration in seconds
    
    Returns:
        Video URL if it matches all criteria, None otherwise
    """
    try:
        # Configure yt-dlp to extract full video metadata without downloading
        ydl_opts = {
            'quiet': True,  # Suppress output messages
            'ignoreerrors': True,  # Continue on download errors
            'extract_flat': False,  # Extract full metadata (needed for date and duration)
        }
        
        # Construct YouTube video URL from video ID
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Extract video information without downloading
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if info:
                # Get upload date and duration from video metadata
                upload_date = info.get('upload_date', '')
                duration = info.get('duration', 0)
                
                # Check if video matches date range filter
                if upload_date and start_date <= upload_date <= end_date:
                    # Check if video matches minimum duration filter
                    if duration and duration > min_duration_secs:
                        return video_url
        
        return None
    
    except Exception as e:
        print(f"Error for {video_id}: {str(e)}")
        return None


def get_video_urls(channel_url: str, outputdir: str, start_date: str, end_date: str, min_duration_secs: int, max_workers: int = 20) -> List[str]:
    """
    Retrieve video URLs from a channel that match date and duration filters.
    Parallelized version for faster processing.
    
    Args:
        channel_url: YouTube channel URL to extract videos from
        outputdir: Path to text file where filtered URLs will be saved
        start_date: Start date for filtering in YYYYMMDD format (e.g., "20240101")
        end_date: End date for filtering in YYYYMMDD format (e.g., "20241231")
        min_duration_secs: Minimum video duration in seconds
        max_workers: Number of videos to check simultaneously (default: 20)
    
    Returns:
        List of video URLs that match all filtering criteria
    """
    # Configure yt-dlp to quickly get all video IDs from channel
    ydl_opts = {
        'quiet': False,  # Show progress messages
        'ignoreerrors': True,  # Continue on errors
        'extract_flat': True,  # Only get video IDs (fast extraction)
    }

    # Step 1: Quickly extract all video IDs from the channel
    video_ids = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(channel_url, download=False)
        if 'entries' in info_dict:
            for video_entry in info_dict['entries']:
                if video_entry and 'id' in video_entry:
                    video_ids.append(video_entry['id'])
    
    print(f"Total videos found: {len(video_ids)}")
    print("Filtering in parallel...")
    
    # Step 2: Process videos in parallel to check filtering criteria
    video_urls = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video checking tasks to thread pool
        futures = {
            executor.submit(extract_video_info, video_id, start_date, end_date, min_duration_secs): video_id 
            for video_id in video_ids
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result:
                video_urls.append(result)
                print(f"Valid video found: {len(video_urls)}")

    # Step 3: Save filtered URLs to text file
    with open(outputdir, 'w') as f:
        for url in video_urls: 
            f.write(url + '\n')

    return video_urls 


if __name__ == "__main__":
    
    # Configuration parameters
    CHANNEL_URL = "https://www.youtube.com/@TheGametimeHighlights/videos"
    MINIMUM_DURATION = 8 * 60  # The video has to be at least 8 minutes long
    
    # Train data are video from the 2024/2025 season
    START_DATE_TRAIN = "20241101"  
    END_DATE_TRAIN = "20250501"
    
    # Test data are video from the begening of the 2025/2026 season
    START_DATE_TEST = "20251020"  
    END_DATE_TEST = "20251201"  
    
    # Get training set URLs (videos from Nov 2024 to May 2025)
    train_urls = get_video_urls(
        channel_url=CHANNEL_URL,
        outputdir='Data/train_urls.txt',
        start_date=START_DATE_TRAIN,
        end_date=END_DATE_TRAIN,
        min_duration_secs=MINIMUM_DURATION,
        max_workers=32  # Number of parallel workers
    )
    
    print(f"Total: {len(train_urls)} URLs saved")

    # Get test set URLs (videos from Oct 2025 to Dec 2025)
    test_urls = get_video_urls(
        channel_url=CHANNEL_URL,
        outputdir='Data/test_urls.txt',
        start_date=START_DATE_TEST,
        end_date=END_DATE_TEST,
        min_duration_secs=MINIMUM_DURATION,
        max_workers=32
    )
    
    print(f"Total: {len(test_urls)} URLs saved")