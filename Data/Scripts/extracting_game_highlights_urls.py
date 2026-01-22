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
- Saves results incrementally and resumes from where it left off.
"""

import pandas as pd
import yt_dlp
import os
import re
from typing import List, Dict, Tuple, Union
import time
from tqdm import tqdm
import concurrent.futures
from datetime import datetime, timedelta
import csv
# ============================================================================
# CONFIGURATION
# ============================================================================

# Input CSV file containing NBA game data (game_id, teams, dates, etc.)
GAMES_CSV_INPUT = "Data/exposure_and_game_info/nba_games_2025-26.csv"

# Output CSV file where valid video URLs will be saved
URLS_CSV_OUTPUT = "Data/urls/game_highlight_urls_2025_26.csv"

# Maximum number of search results to retrieve per game query
MAX_RESULTS_PER_QUERY = 10

# Delay between searches per thread to avoid YouTube rate-limiting (in seconds)
SEARCH_DELAY = 0.3

# Number of parallel threads for simultaneous video searching
MAX_WORKERS = 32

# Enable detailed logging of rejected videos with reasons
DEBUG_REJECTIONS = True 

# Save frequency (number of games processed before saving)
SAVE_EVERY = 3

# Blacklist official channel names to exclude from results
CHANNEL_BLACKLIST = [
    "NBA on ESPN",
    "Sports On Prime",
    "Prime Video Sport France",
    "Amazon Prime Video Türkiye",
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
# RESUMABILITY FUNCTIONS
# ============================================================================

def load_existing_results(output_path: str) -> Tuple[pd.DataFrame, set]:
    """
    Load existing results CSV and return both the dataframe and a set of processed game_ids.
    
    Args:
        output_path: Path to the output CSV file
    
    Returns:
        Tuple containing:
            - DataFrame of existing results (empty if file doesn't exist)
            - Set of game_ids that have already been processed
    """
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            processed_game_ids = set(df['game_id'].unique())
            print(f"Found existing results with {len(processed_game_ids)} already processed games.")
            return df, processed_game_ids
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return pd.DataFrame(), set()
    return pd.DataFrame(), set()


def save_results_incremental(new_videos: List[Dict], output_path: str):
    """
    Append new video results to the CSV file (or create if doesn't exist).
    """
    if not new_videos:
        return
    
    new_df = pd.DataFrame(new_videos)
    
    # Reorder columns for better readability
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
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in new_df.columns]
    new_df = new_df[existing_columns]
    
    try:
        if os.path.exists(output_path):
            # Append to existing file
            # AJOUT DE: quoting=csv.QUOTE_NONNUMERIC
            new_df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
        else:
            # Create new file with header
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            # AJOUT DE: quoting=csv.QUOTE_NONNUMERIC
            new_df.to_csv(output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    except Exception as e:
        print(f"\n[SAVE ERROR] Could not save results: {e}")


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
        video_id: YouTube video ID 
    
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
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'cookiesfrombrowser': ('firefox',),
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
        List of dictionaries, each containing metadata for a found video
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'playlistend': max_results
        }
        
        search_string = f"ytsearch{max_results}:{query}"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_string, download=False)
            
            if not result or 'entries' not in result:
                return []
            
            videos = []
            for entry in result.get('entries', []):
                if entry and entry.get('id'):
                    duration = entry.get('duration')
                    view_count = entry.get('view_count')
                    
                    videos.append({
                        'video_id': entry['id'],
                        'title': entry.get('title', ''),
                        'channel': entry.get('channel', ''),
                        'duration': duration if duration else 0,
                        'view_count': view_count if view_count else 0,
                        'url': f"https://www.youtube.com/watch?v={entry['id']}",
                        'upload_date': entry.get('upload_date', '')
                    })
            
            return videos
    
    except Exception as e:
        raise e


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _team_matches(text: str, team_abbr: str, team_full_name: str) -> bool:
    """Check if the text contains any reference to the team."""
    if team_full_name.lower() in text:
        return True
    
    if re.search(rf'\b{team_abbr.lower()}\b', text):
        return True
    
    name_parts = team_full_name.lower().split()
    
    if len(name_parts) >= 2:
        nickname = name_parts[-1]
        
        if re.search(rf'\b{re.escape(nickname)}\b', text):
            return True
        
        city = ' '.join(name_parts[:-1])
        if re.search(rf'\b{re.escape(city)}\b', text):
            return True
        
        if len(name_parts) >= 3:
            nickname_two_words = ' '.join(name_parts[-2:])
            if re.search(rf'\b{re.escape(nickname_two_words)}\b', text):
                return True
    
    return False


def _date_matches(text: str, target_date) -> bool:
    """Check if the text contains the target date in any common format."""
    month_full = target_date.strftime('%B').lower()
    month_abbr = target_date.strftime('%b').lower()
    month_num = target_date.month
    day = target_date.day
    year_full = target_date.year
    year_short = target_date.year % 100
    
    day_str = str(day)
    day_padded = f'{day:02d}'
    
    month_str = str(month_num)
    month_padded = f'{month_num:02d}'
    
    year_short_str = str(year_short)
    year_full_str = str(year_full)
    
    date_patterns = [
        rf'({month_full}|{month_abbr})\.?\s*({day_str}|{day_padded})\s*,?\s*({year_full_str}|{year_short_str})?',
        rf'({day_str}|{day_padded})\s+({month_full}|{month_abbr})\.?\s*,?\s*({year_full_str}|{year_short_str})?',
        rf'{day_str}(st|nd|rd|th)?\s+(of\s+)?({month_full}|{month_abbr})',
        rf'({month_str}|{month_padded})/({day_str}|{day_padded})/({year_full_str}|{year_short_str})',
        rf'({month_str}|{month_padded})/({day_str}|{day_padded})(?!/|\d)',
        rf'({month_str}|{month_padded})-({day_str}|{day_padded})-({year_full_str}|{year_short_str})',
        rf'({month_str}|{month_padded})\.({day_str}|{day_padded})\.({year_full_str}|{year_short_str})',
        rf'({day_str}|{day_padded})/({month_str}|{month_padded})/({year_full_str}|{year_short_str})',
        rf'({day_str}|{day_padded})-({month_str}|{month_padded})-({year_full_str}|{year_short_str})',
        rf'{year_full_str}-({month_str}|{month_padded})-({day_str}|{day_padded})',
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, text):
            return True
    
    return False


# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def is_valid_video(video: Dict, game_info: Dict) -> Union[bool, str]:
    """
    Validate a single video against multiple criteria to ensure it's a legitimate game highlight.
    Returns True if valid, or a rejection reason string if invalid.
    """
    # Check 1: Channel Blacklist
    channel_name = (video.get('channel') or '').strip()
    if channel_name in CHANNEL_BLACKLIST:
        return f"Rejected (Blacklisted Channel: {channel_name})"

    # Check 2: Video Duration
    duration = video.get('duration', 0)
    if duration < 60:
        return f"Rejected (Too Short: {int(duration)}s)"
    if duration > 900:
        return f"Rejected (Too Long: {int(duration)}s)"
    
    # Check 3: Minimum View Count
    view_count = video.get('view_count', 0)
    if view_count < 1000:
        return f"Rejected (Insufficient Views: {view_count} views)"
    
    upload_date_str = video.get('upload_date')
    if upload_date_str:
        try:
            # Parse YYYYMMDD string to datetime
            upload_dt = datetime.strptime(upload_date_str, '%Y%m%d')
            # Calculate the cutoff date (7 days ago)
            cutoff_date = datetime.now() - timedelta(days=7)
            
            # If the video is newer (greater) than the cutoff, reject it
            if upload_dt > cutoff_date:
                 return f"Rejected (Too Recent: Uploaded {upload_dt.strftime('%Y-%m-%d')})"
        except ValueError:
            # If date format is missing or weird, we decide whether to pass or fail. 
            # Usually safe to pass and let other checks handle it, or log it.
            pass
    
    # Prepare text for analysis
    text = (video.get('title', '') + ' ' + (video.get('description') or '')).lower()
    
    # Check 4: Date Matching
    game_date = game_info['game_date_dt']
    
    if not _date_matches(text, game_date):
        return f"Rejected (Date Mismatch: Expected {game_date.strftime('%b %d')})"
    
    # Check 5: Team Matching
    home_team_found = _team_matches(text, game_info['home_abbr'], game_info['home_team'])
    away_team_found = _team_matches(text, game_info['away_abbr'], game_info['away_team'])
    
    if not home_team_found and not away_team_found:
        return f"Rejected (Team Mismatch: Expected {game_info['home_abbr']} or {game_info['away_abbr']})"
    
    return True


def process_single_game(game_info: Dict) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Worker function to process a single game query (designed for parallel execution).
    """
    videos = search_youtube_videos(game_info['query'], MAX_RESULTS_PER_QUERY)
    
    valid_videos = []
    rejected_videos = []

    for video in videos:
        validation_result = is_valid_video(video, game_info)
        
        if validation_result is True:
            metadata = fetch_video_metadata(video['video_id'])
            video.update(metadata)
            
            video['home_team_name'] = game_info['home_team']
            video['away_team_name'] = game_info['away_team']
            video['home_team_abbreviation'] = game_info['home_abbr']
            video['away_team_abbreviation'] = game_info['away_abbr']
            
            valid_videos.append(video)
            
        elif DEBUG_REJECTIONS:
            video['reason'] = validation_result
            rejected_videos.append(video)
    
    time.sleep(SEARCH_DELAY) 
    
    return game_info, valid_videos, rejected_videos


def run_parallel_collection(queries: List[Dict], output_path: str) -> int:
    """
    Orchestrate the parallel collection of video data with incremental saving.
    
    Args:
        queries: List of game query dictionaries to process
        output_path: Path to save results incrementally
    
    Returns:
        Total number of videos collected
    """
    pending_results = []
    processed_count = 0
    total_videos_collected = 0
    
    # Create thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all game queries to the thread pool
        future_to_game = {executor.submit(process_single_game, q): q for q in queries}
        
        # Process results as they complete (with progress bar)
        with tqdm(total=len(queries), desc="Collecting game videos") as pbar:
            for future in concurrent.futures.as_completed(future_to_game):
                original_game_info = future_to_game[future]
                
                try:
                    _, valid_videos, rejected_videos = future.result()
                    
                    print(f"\n--- Result for search: \"{original_game_info['query']}\" ---")
                    print(f"  > Found {len(valid_videos)} valid video(s).")
                    
                    if DEBUG_REJECTIONS and rejected_videos:
                        print("  > Rejected videos:")
                        for v in rejected_videos:
                            print(f"    - {v['reason']}: {v['title']} ({v['url']})")
                    
                    # Add game_id to each valid video
                    for video in valid_videos:
                        video['game_id'] = original_game_info['game_id']
                        pending_results.append(video)
                        total_videos_collected += 1

                except Exception as e:
                    print(f"\n[ERROR] Task failed for query '{original_game_info['query']}': {e}")
                
                processed_count += 1
                pbar.update(1)
                
                # Save incrementally
                if processed_count % SAVE_EVERY == 0 or processed_count == len(queries):
                    if pending_results:
                        save_results_incremental(pending_results, output_path)
                        pbar.set_description(f"✓ Saved {len(pending_results)} videos (Total: {total_videos_collected})")
                        pending_results = []
    
    # Final save for any remaining results
    if pending_results:
        save_results_incremental(pending_results, output_path)
        print(f"\n✓ Saved final {len(pending_results)} videos")
    
    return total_videos_collected


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline with resumability support.
    """
    start_time = time.time()
    
    print("="*60)
    print("YOUTUBE NBA HIGHLIGHT COLLECTOR (RESUMABLE)")
    print("="*60)
    
    # Load existing results and get processed game_ids
    existing_df, processed_game_ids = load_existing_results(URLS_CSV_OUTPUT)
    
    # Load game data from CSV file
    try:
        df = pd.read_csv(GAMES_CSV_INPUT)
        print(f"Loaded {len(df)} games from '{GAMES_CSV_INPUT}'")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{GAMES_CSV_INPUT}'. Aborting.")
        return

    # Convert game dates to datetime format
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['game_date_str'] = df['GAME_DATE'].dt.strftime('%b %d %Y')
    
    # Create formatted YouTube search queries
    df['youtube_search_name'] = (
        df['HOME_TEAM_NAME'] + ' vs ' + df['AWAY_TEAM_NAME'] + 
        ' Full game highlights | ' + df['game_date_str'] + ' NBA Season'
    )
    
    # Generate list of search queries
    all_queries = create_youtube_queries(df)
    
    # Filter out already processed games
    remaining_queries = [q for q in all_queries if q['game_id'] not in processed_game_ids]
    
    if len(remaining_queries) < len(all_queries):
        skipped = len(all_queries) - len(remaining_queries)
        print(f"Skipping {skipped} already processed games.")
    
    if len(remaining_queries) == 0:
        print("All games have already been processed!")
        return
    
    # Execute parallel collection
    print(f"\nStarting parallel collection with {MAX_WORKERS} workers for {len(remaining_queries)} games...")
    print(f"Save frequency: Every {SAVE_EVERY} games")
    print("="*60)
    
    videos_collected = run_parallel_collection(remaining_queries, URLS_CSV_OUTPUT)
    
    # Display final summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Videos collected (this session): {videos_collected}")
    print(f"Total videos in results: {len(processed_game_ids) + videos_collected}")
    print(f"Output file: {URLS_CSV_OUTPUT}")
    
    if hours > 0:
        print(f"Total time: {hours}h {minutes}m {seconds:.2f}s")
    elif minutes > 0:
        print(f"Total time: {minutes}m {seconds:.2f}s")
    else:
        print(f"Total time: {seconds:.2f}s")
    
    if len(remaining_queries) > 0:
        print(f"Average time per game: {elapsed_time/len(remaining_queries):.2f}s")
    print("="*60)


if __name__ == "__main__":
    main()