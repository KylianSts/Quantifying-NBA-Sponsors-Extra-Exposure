"""
YouTube Video Collector for NBA Games

This script automates the collection of YouTube highlight video URLs for a given
list of NBA games. It operates in parallel to speed up the process and includes
robust validation checks to ensure the relevance of collected videos.

Key features:
- Generates search queries from a CSV of NBA games.
- Uses yt-dlp to search YouTube in parallel with multiple workers.
- Validates videos based on duration, channel blacklists, and date matching (game day or next day).
- Fetches detailed video metadata (views, likes, comments, duration).
- Includes team information (full names and abbreviations) in output.
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
GAMES_CSV_INPUT = "Data/nba_games_2025-26.csv"

# Output CSV file where valid video URLs will be saved
URLS_CSV_OUTPUT = "Data/urls/game_highlight_urls.csv"

# Maximum number of search results to retrieve per game query
MAX_RESULTS_PER_QUERY = 10

# Delay between searches per thread to avoid YouTube rate-limiting (in seconds)
SEARCH_DELAY = 0.1

# Number of parallel threads for simultaneous video searching
MAX_WORKERS = 32

# Enable detailed logging of rejected videos with reasons
DEBUG_REJECTIONS = True 

# Blacklist official channel names to exclude from results
CHANNEL_BLACKLIST = [
    "Prime Video Sport France",
    "NBA Europe",
    "NBA G League",
    "NBA Extra - beIN SPORTS France",
    "NBA",
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "LA Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards"
]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def create_youtube_queries(df: pd.DataFrame) -> List[Dict]:
    """
    Generate YouTube search queries and validation metadata for each game in a DataFrame.
    Creates structured search strings optimized for finding NBA game highlights.
    
    Args:
        df: DataFrame with game data, requires columns: 'GAME_ID', 'HOME_TEAM_NAME',
            'AWAY_TEAM_NAME', 'GAME_DATE', 'HOME_TEAM_ABBREVIATION', 'AWAY_TEAM_ABBREVIATION'
    
    Returns:
        List of dictionaries, where each dictionary contains:
            - game_id: Unique identifier for the game
            - query: Formatted YouTube search string
            - game_date_dt: Game date as datetime object
            - home_team: Full name of home team
            - away_team: Full name of away team
            - home_abbr: Home team abbreviation
            - away_abbr: Away team abbreviation
    """
    queries = []
    
    # Iterate through each game in the DataFrame
    for _, row in df.iterrows():
        # Skip rows with missing critical data
        if pd.isna(row['GAME_ID']) or pd.isna(row['HOME_TEAM_NAME']):
            continue
        
        # Convert game date string to datetime object for date validation
        game_date = pd.to_datetime(row['GAME_DATE'])
        
        # Create a dictionary with all game metadata needed for searching and validation
        query_dict = {
            'game_id': row['GAME_ID'],
            'query': row['youtube_search_name'],
            'game_date_dt': game_date,
            'home_team': row['HOME_TEAM_NAME'],
            'away_team': row['AWAY_TEAM_NAME'],
            'home_abbr': row['HOME_TEAM_ABBREVIATION'],
            'away_abbr': row['AWAY_TEAM_ABBREVIATION']
        }
        queries.append(query_dict)
    
    return queries


def fetch_video_metadata(video_id: str) -> Dict:
    """
    Fetch detailed metadata for a single video using yt-dlp.
    Retrieves views, likes, comments, duration, and channel information.
    
    Args:
        video_id: YouTube video ID (e.g., 'dQw4w9WgXcQ')
    
    Returns:
        Dictionary containing video metadata:
            - view_count: Number of views
            - like_count: Number of likes
            - comment_count: Number of comments
            - duration: Video duration in seconds
            - channel: Channel name
    """
    try:
        # Configure yt-dlp to extract full metadata (not flat)
        ydl_opts = {
            'quiet': True,  # Suppress console output
            'no_warnings': True,  # Hide warning messages
            'extract_flat': False,  # Extract full metadata for views/likes/comments
        }
        
        # Construct video URL
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Extract video metadata
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if info:
                return {
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'comment_count': info.get('comment_count', 0),
                    'duration': info.get('duration', 0),
                    'channel': info.get('channel', '')
                }
    
    except Exception as e:
        print(f"[WARNING] Failed to fetch metadata for {video_id}: {e}")
    
    # Return default values if extraction fails
    return {
        'view_count': 0,
        'like_count': 0,
        'comment_count': 0,
        'duration': 0,
        'channel': ''
    }


def search_youtube_videos(query: str, max_results: int) -> List[Dict]:
    """
    Perform a YouTube search using yt-dlp and return video metadata without downloading.
    Uses flat extraction for faster initial search results.
    
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
            'extract_flat': True,  # Get metadata only, don't download (fast)
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
    
    Supported date formats in video titles:
    - "November 2, 2025" or "Nov 2, 2025" (month day, year)
    - "November 2 2025" or "Nov 2 2025" (month day year without comma)
    - "2 November 2025" or "2 Nov 2025" (day month year)
    - "11/2/25" or "11/2/2025" (MM/DD/YY or MM/DD/YYYY)
    - "11-2-25" or "11-2-2025" (MM-DD-YY or MM-DD-YYYY)
    - "2/11/25" or "2/11/2025" (DD/MM/YY - European format)
    - "Nov 02" or "November 02" (month day without year)
    - "11.2.25" or "11.2.2025" (with dots)
    
    Args:
        video: Dictionary of video metadata (title, channel, duration, url, etc.)
        game_info: Dictionary of the target game's metadata (teams, date, etc.)
    
    Returns:
        True if the video passes all validation checks
        String describing the rejection reason if the video fails any check
    """
    # Check if video is from a blacklisted channel
    channel_name = (video.get('channel') or '').strip()
    if channel_name in CHANNEL_BLACKLIST:
        return f"Rejected (Blacklisted Channel: {channel_name})"

    # Validate video duration (too short = clips/teasers, too long = full games)
    duration = video.get('duration', 0)
    if duration < 60:  # Less than 1 minute is likely a teaser or ad
        return f"Rejected (Too Short: {int(duration)}s)"
    if duration > 900:  # More than 15 minutes is likely not highlights
        return f"Rejected (Too Long: {int(duration)}s)"
    
    # Combine title and description for comprehensive text analysis
    text = (video.get('title', '') + ' ' + (video.get('description') or '')).lower()
    
    # Get game date and next day (highlights often uploaded day after)
    game_date = game_info['game_date_dt']
    
    # Check both game day
    if _date_matches(text, game_date):
        return True  # Video passed all validation checks
    
    # Reject video if no date pattern matches
    return f"Rejected (Date Mismatch: Expected {game_date.strftime('%b %d')} or {next_day_date.strftime('%b %d')})"


def _date_matches(text: str, target_date) -> bool:
    """
    Check if the text contains the target date in any common format.
    
    Args:
        text: Lowercase text to search (title + description)
        target_date: Datetime object to match against
    
    Returns:
        True if date is found in text, False otherwise
    """
    # Extract date components
    month_full = target_date.strftime('%B').lower()  # "november"
    month_abbr = target_date.strftime('%b').lower()  # "nov"
    month_num = target_date.month  # 11
    day = target_date.day  # 2
    year_full = target_date.year  # 2025
    year_short = target_date.year % 100  # 25
    
    # Day with optional leading zero: matches both "2" and "02"
    day_pattern = rf'0?{day}'
    
    # Month number with optional leading zero: matches both "11" and "11"
    month_num_pattern = rf'0?{month_num}'
    
    # Year pattern: matches "2025", "25", or no year at all
    year_pattern = rf'({year_full}|{year_short})?'
    
    # Build comprehensive date patterns
    date_patterns = [
        # ===== Text-based month formats =====
        
        # "November 2, 2025" or "Nov 2, 2025" (with comma)
        rf'({month_full}|{month_abbr})\.?\s+{day_pattern}\s*,?\s*{year_pattern}',
        
        # "2 November 2025" or "2 Nov 2025" (day first)
        rf'{day_pattern}\s+({month_full}|{month_abbr})\.?\s*,?\s*{year_pattern}',
        
        # "2nd November" or "2nd of November" (ordinal)
        rf'{day}(st|nd|rd|th)?\s+(of\s+)?({month_full}|{month_abbr})',
        
        # ===== Numeric formats (US: MM/DD/YY) =====
        
        # "11/2/25" or "11/2/2025" or "11/02/25" (slash separator)
        rf'{month_num_pattern}/{day_pattern}/({year_full}|{year_short})',
        
        # "11/2" without year
        rf'{month_num_pattern}/{day_pattern}(?![/\d])',
        
        # "11-2-25" or "11-2-2025" (dash separator)
        rf'{month_num_pattern}-{day_pattern}-({year_full}|{year_short})',
        
        # "11.2.25" or "11.2.2025" (dot separator)
        rf'{month_num_pattern}\.{day_pattern}\.({year_full}|{year_short})',
        
        # ===== Numeric formats (European: DD/MM/YY) =====
        
        # "2/11/25" or "02/11/2025" (day/month/year)
        rf'{day_pattern}/{month_num_pattern}/({year_full}|{year_short})',
        
        # "2-11-25" or "02-11-2025" (day-month-year)
        rf'{day_pattern}-{month_num_pattern}-({year_full}|{year_short})',
        
        # ===== ISO format =====
        
        # "2025-11-02" (ISO format YYYY-MM-DD)
        rf'{year_full}-{month_num_pattern}-{day_pattern}',
    ]
    
    # Check if any pattern matches
    for pattern in date_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def process_single_game(game_info: Dict) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Worker function to process a single game query (designed for parallel execution).
    Searches YouTube for game highlights, validates each result, and fetches detailed metadata.
    
    Args:
        game_info: Dictionary containing all info for one game (query, date, teams, etc.)
    
    Returns:
        Tuple containing three elements:
            - game_info: Original game information dictionary
            - valid_videos: List of videos that passed all validation checks (with full metadata)
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
            # Video passed validation, fetch detailed metadata (views, likes, comments)
            metadata = fetch_video_metadata(video['video_id'])
            
            # Merge video info with detailed metadata
            video.update(metadata)
            
            # Add team information from game_info
            video['home_team_name'] = game_info['home_team']
            video['away_team_name'] = game_info['away_team']
            video['home_team_abbreviation'] = game_info['home_abbr']
            video['away_team_abbreviation'] = game_info['away_abbr']
            
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
        List of dictionaries, each containing a valid video's data with:
            - game_id: Associated NBA game identifier
            - Video metadata (title, url, channel, duration, video_id)
            - Engagement metrics (view_count, like_count, comment_count)
            - Team information (names and abbreviations)
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
    print("="*60)
    print("YOUTUBE NBA HIGHLIGHT COLLECTOR")
    print("="*60)
    
    # Load game data from CSV file
    try:
        df = pd.read_csv(GAMES_CSV_INPUT)
        print(f"Loaded {len(df)} games from '{GAMES_CSV_INPUT}'")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{GAMES_CSV_INPUT}'. Aborting.")
        return

    # Convert game dates to datetime format for date validation
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['game_date_str'] = df['GAME_DATE'].dt.strftime('%b %d %Y')  # Format: "Jan 15 2025"
    
    # Create formatted YouTube search queries for each game
    df['youtube_search_name'] = (
        df['HOME_TEAM_NAME'] + ' vs ' + df['AWAY_TEAM_NAME'] + 
        ' Full game highlights | ' + df['game_date_str'] + ' NBA Season'
    )
    
    # Generate list of search queries with metadata
    queries = create_youtube_queries(df)
    
    # Execute parallel collection across all games
    print(f"\nStarting parallel collection with {MAX_WORKERS} workers for {len(queries)} games...")
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
        # Video info -> Engagement metrics -> Team info -> Game ID
        column_order = [
            'game_id',
            'title',
            'url',
            'video_id',
            'channel',
            'duration',
            'view_count',
            'like_count',
            'comment_count',
            'home_team_name',
            'home_team_abbreviation',
            'away_team_name',
            'away_team_abbreviation'
        ]
        
        # Only include columns that exist (in case some are missing)
        existing_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[existing_columns]
        
        # Save to CSV with UTF-8 encoding to handle special characters
        results_df.to_csv(URLS_CSV_OUTPUT, index=False, encoding='utf-8')
        
        # Display summary
        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print("="*60)
        print(f"Total videos collected: {len(results_df)}")
        print(f"Output file: {URLS_CSV_OUTPUT}")
        print(f"\nColumns in output:")
        for col in existing_columns:
            print(f"  - {col}")
        print("="*60)
    else:
        print("\nNo valid videos were found. Output file was not created.")


if __name__ == "__main__":
    main()