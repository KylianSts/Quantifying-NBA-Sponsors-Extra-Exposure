"""
YouTube Video Collector for NBA Games (Parallel & Professional Version)

This script automates the collection of YouTube highlight video URLs for a given
list of NBA games. It operates in parallel to speed up the process and includes
robust validation checks to ensure the relevance of collected videos.

Key features:
- Generates search queries from a CSV of NBA games.
- Uses yt-dlp to search YouTube in parallel with multiple workers.
- Validates videos based on duration, channel blacklists, and date matching (game day or next day).
- Provides detailed debugging output for rejected videos.
- Saves the results in a clean, machine-readable CSV format.
"""

import pandas as pd
import yt_dlp
import os
import re
from typing import List, Dict, Tuple, Union
import time
from tqdm import tqdm
import concurrent.futures
from datetime import timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input CSV file containing NBA game data (game_id, teams, dates, etc.)
GAMES_CSV_INPUT = "Data/nba_games_2025_2026.csv"

# Output CSV file where valid video URLs will be saved
URLS_CSV_OUTPUT = "Data/urls/game_highlight_urls.csv"

# Maximum number of search results to retrieve per game query
MAX_RESULTS_PER_QUERY = 10

# Delay between searches per thread to avoid YouTube rate-limiting (in seconds)
SEARCH_DELAY = 0.5

# Number of parallel threads for simultaneous video searching
MAX_WORKERS = 32

# Enable detailed logging of rejected videos with reasons
DEBUG_REJECTIONS = True 

# Blacklist of channel names to exclude from results (channels known for low-quality content)
CHANNEL_BLACKLIST = {
    # Add channel names here in lowercase, e.g., 'spam_channel'
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def create_youtube_queries(df: pd.DataFrame) -> List[Dict]:
    """
    Generate YouTube search queries and validation metadata for each game in a DataFrame.
    Creates structured search strings optimized for finding NBA game highlights.
    
    Args:
        df: DataFrame with game data, requires columns: 'game_id', 'home_team_name',
            'visitor_team_name', 'game_date', 'home_team_abbreviation', 'visitor_team_abbreviation'
    
    Returns:
        List of dictionaries, where each dictionary contains:
            - game_id: Unique identifier for the game
            - query: Formatted YouTube search string
            - game_date_dt: Game date as datetime object
            - home_team: Full name of home team
            - visitor_team: Full name of visiting team
            - home_abbr: Home team abbreviation
            - visitor_abbr: Visitor team abbreviation
    """
    queries = []
    
    # Iterate through each game in the DataFrame
    for _, row in df.iterrows():
        # Skip rows with missing critical data
        if pd.isna(row['game_id']) or pd.isna(row['home_team_name']):
            continue
        
        # Convert game date string to datetime object for date validation
        game_date = pd.to_datetime(row['game_date'])
        
        # Create a dictionary with all game metadata needed for searching and validation
        query_dict = {
            'game_id': int(row['game_id']),
            'query': row['youtube_search_name'],
            'game_date_dt': game_date,
            'home_team': row['home_team_name'],
            'visitor_team': row['visitor_team_name'],
            'home_abbr': row['home_team_abbreviation'],
            'visitor_abbr': row['visitor_team_abbreviation']
        }
        queries.append(query_dict)
    
    return queries


def search_youtube_videos(query: str, max_results: int) -> List[Dict]:
    """
    Perform a YouTube search using yt-dlp and return video metadata without downloading.
    Uses flat extraction for faster results and reduced bandwidth.
    
    Args:
        query: The search string for YouTube (formatted with team names and date)
        max_results: The maximum number of video results to fetch from search
    
    Returns:
        List of dictionaries, each containing metadata for a found video:
            - video_id: Unique YouTube video identifier
            - title: Video title
            - channel: Channel name that uploaded the video
            - duration: Video duration in seconds
            - url: Full YouTube watch URL
    """
    try:
        # Configure yt-dlp for search-only mode (no downloads)
        ydl_opts = {
            'quiet': True,  # Suppress console output
            'no_warnings': True,  # Hide warning messages
            'extract_flat': True,  # Get metadata only, don't download
            'playlistend': max_results  # Limit number of results
        }
        
        # Format search string for yt-dlp (ytsearch prefix tells yt-dlp to search YouTube)
        search_string = f"ytsearch{max_results}:{query}"
        
        # Execute the YouTube search
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_string, download=False)
            
            # Check if search returned any results
            if not result or 'entries' not in result:
                return []
            
            # Parse video metadata from search results
            videos = []
            for entry in result.get('entries', []):
                # Only process valid entries with video IDs
                if entry and entry.get('id'):
                    duration = entry.get('duration')
                    videos.append({
                        'video_id': entry['id'],
                        'title': entry.get('title', ''),
                        'channel': entry.get('channel', ''),
                        'duration': duration if duration else 0,  # Default to 0 if duration missing
                        'url': f"https://www.youtube.com/watch?v={entry['id']}"
                    })
            
            return videos
    
    except Exception as e:
        # Re-raise exception to be handled by the caller
        raise e


def is_valid_video(video: Dict, game_info: Dict) -> Union[bool, str]:
    """
    Validate a single video against multiple criteria to ensure it's a legitimate game highlight.
    Checks include channel blacklist, duration limits, and date matching.
    
    Args:
        video: Dictionary of video metadata (title, channel, duration, url, etc.)
        game_info: Dictionary of the target game's metadata (teams, date, etc.)
    
    Returns:
        True if the video passes all validation checks
        String describing the rejection reason if the video fails any check
    """
    # Check if video is from a blacklisted channel (known spam or low-quality sources)
    channel_name = (video.get('channel') or '').lower()
    if channel_name in CHANNEL_BLACKLIST:
        return f"Rejected (Blacklisted Channel: {channel_name})"

    # Validate video duration (too short = clips/teasers, too long = full games)
    duration = video.get('duration', 0)
    if duration < 60:  # Less than 1 minute is likely a teaser or ad
        return f"Rejected (Too Short: {int(duration)}s)"
    if duration > 900:  # More than 15 minutes is likely a full game, not highlights
        return f"Rejected (Too Long: {int(duration)}s)"
    
    # Combine title and description for comprehensive text analysis
    text = (video.get('title', '') + ' ' + (video.get('description') or '')).lower()
    
    # Check if video mentions the game date (game day or next day upload)
    game_date = game_info['game_date_dt']
    next_day_date = game_date + timedelta(days=1)  # Highlights often uploaded day after
    
    # Build regex patterns for both full month names and abbreviations
    date_patterns = []
    for d in [game_date, next_day_date]:
        month_full = d.strftime('%B').lower()  # e.g., "January"
        month_abbr = d.strftime('%b').lower()  # e.g., "Jan"
        day = str(d.day)  # e.g., "15"
        
        # Create patterns for "Month Day" and "Day Month" formats
        date_patterns.extend([
            rf'({month_full}|{month_abbr})\s+{day}',  # "January 15" or "Jan 15"
            rf'{day}\s+({month_full}|{month_abbr})'   # "15 January" or "15 Jan"
        ])
    
    # Reject video if no date pattern matches
    if not any(re.search(pattern, text) for pattern in date_patterns):
        return f"Rejected (Date Mismatch: Expected {game_date.strftime('%b %d')} or {next_day_date.strftime('%b %d')})"
    
    # Video passed all validation checks
    return True


def process_single_game(game_info: Dict) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Worker function to process a single game query (designed for parallel execution).
    Searches YouTube for game highlights and validates each result.
    
    Args:
        game_info: Dictionary containing all info for one game (query, date, teams, etc.)
    
    Returns:
        Tuple containing three elements:
            - game_info: Original game information dictionary
            - valid_videos: List of videos that passed all validation checks
            - rejected_videos: List of videos that failed validation (with rejection reasons)
    """
    # Search YouTube for videos matching this game
    videos = search_youtube_videos(game_info['query'], MAX_RESULTS_PER_QUERY)
    
    # Initialize lists to categorize videos
    valid_videos = []
    rejected_videos = []

    # Validate each video found in the search
    for video in videos:
        validation_result = is_valid_video(video, game_info)
        
        if validation_result is True:
            # Video passed validation, add to valid list
            valid_videos.append(video)
        elif DEBUG_REJECTIONS:
            # Video failed validation, record the reason for debugging
            video['reason'] = validation_result
            rejected_videos.append(video)
    
    # Small delay to respect rate limits (per thread)
    time.sleep(SEARCH_DELAY) 
    
    return game_info, valid_videos, rejected_videos


def run_parallel_collection(queries: List[Dict]) -> List[Dict]:
    """
    Orchestrate the parallel collection of video data for multiple games.
    Distributes game queries across multiple worker threads for faster processing.
    
    Args:
        queries: List of game query dictionaries to process (from create_youtube_queries)
    
    Returns:
        List of dictionaries, each containing a valid video's data with its associated game_id
    """
    all_valid_videos = []
    
    # Create thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all game queries to the thread pool
        future_to_game = {executor.submit(process_single_game, q): q for q in queries}
        
        # Process results as they complete (with progress bar)
        for future in tqdm(concurrent.futures.as_completed(future_to_game), 
                          total=len(queries), 
                          desc="Collecting game videos"):
            original_game_info = future_to_game[future]
            
            try:
                # Retrieve results from completed task
                _, valid_videos, rejected_videos = future.result()
                
                # Log summary for this game query
                print(f"\n--- Result for search: \"{original_game_info['query']}\" ---")
                print(f"  > Found {len(valid_videos)} valid video(s).")
                
                # Log rejection details if debugging is enabled
                if DEBUG_REJECTIONS and rejected_videos:
                    print("  > Rejected videos:")
                    for v in rejected_videos:
                        print(f"    - {v['reason']}: {v['title']} ({v['url']})")
                
                # Add game_id to each valid video and append to final collection
                for video in valid_videos:
                    video['game_id'] = original_game_info['game_id']
                    all_valid_videos.append(video)

            except Exception as e:
                # Log errors for failed queries
                print(f"\n[ERROR] Task failed for query '{original_game_info['query']}': {e}")
    
    return all_valid_videos


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline for the YouTube highlight collector.
    Loads game data, generates queries, collects videos in parallel, and saves results.
    
    Returns:
        None
    """
    print("Starting the YouTube Highlight Collector script.")
    
    # Load game data from CSV file
    try:
        df = pd.read_csv(GAMES_CSV_INPUT)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{GAMES_CSV_INPUT}'. Aborting.")
        return

    # Convert game dates to datetime format for date validation
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['game_date_str'] = df['game_date'].dt.strftime('%b %d %Y')  # Format: "Jan 15 2025"
    
    # Create formatted YouTube search queries for each game
    df['youtube_search_name'] = (
        df['home_team_name'] + ' vs ' + df['visitor_team_name'] + 
        ' Full game highlights | ' + df['game_date_str'] + ' NBA Season'
    )
    
    # Generate list of search queries with metadata
    queries = create_youtube_queries(df)
    
    # Execute parallel collection across all games
    print(f"Starting parallel collection with {MAX_WORKERS} workers for {len(queries)} games...")
    collected_videos = run_parallel_collection(queries)
    
    # Save collected videos to output CSV file
    if collected_videos:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(URLS_CSV_OUTPUT)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert collected videos to DataFrame for CSV export
        results_df = pd.DataFrame(collected_videos)
        
        # Reorder columns for better readability
        results_df = results_df[['game_id', 'title', 'url', 'channel', 'duration', 'video_id']]
        
        # Save to CSV with UTF-8 encoding to handle special characters
        results_df.to_csv(URLS_CSV_OUTPUT, index=False, encoding='utf-8')
        
        print(f"\nSuccessfully saved {len(results_df)} video URLs to '{URLS_CSV_OUTPUT}'")
    else:
        print("\nNo valid videos were found. Output file was not created.")


if __name__ == "__main__":
    main()