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

# --- Input/Output Files ---
GAMES_CSV_INPUT = "Data/nba_games_2025_2026.csv"
URLS_CSV_OUTPUT = "Data/urls/game_highlight_urls.csv"

# --- Search Parameters ---
MAX_RESULTS_PER_QUERY = 10
SEARCH_DELAY = 0.5  # Delay per thread to avoid rate-limiting
MAX_WORKERS = 32    # Number of parallel search threads

# --- Filtering Parameters ---
DEBUG_REJECTIONS = True 
CHANNEL_BLACKLIST = {

}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def create_youtube_queries(df: pd.DataFrame) -> List[Dict]:
    """
    Generate YouTube search queries and validation metadata for each game in a DataFrame.

    Args:
        df: DataFrame with game data, requires columns like 'game_id', 'home_team_name', etc.

    Returns:
        A list of dictionaries, where each dictionary represents a game and its search query.
    """
    queries = []
    for _, row in df.iterrows():
        if pd.isna(row['game_id']) or pd.isna(row['home_team_name']):
            continue
        
        game_date = pd.to_datetime(row['game_date'])
        
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
    Perform a YouTube search using yt-dlp and return video metadata.

    Args:
        query: The search string for YouTube.
        max_results: The maximum number of video results to fetch.

    Returns:
        A list of dictionaries, each containing metadata for a found video.
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
                    videos.append({
                        'video_id': entry['id'],
                        'title': entry.get('title', ''),
                        'channel': entry.get('channel', ''),
                        'duration': duration if duration else 0,
                        'url': f"https://www.youtube.com/watch?v={entry['id']}"
                    })
            return videos
    except Exception as e:
        raise e


def is_valid_video(video: Dict, game_info: Dict) -> Union[bool, str]:
    """
    Validate a single video against a set of criteria.

    Args:
        video: A dictionary of video metadata.
        game_info: A dictionary of the target game's metadata.

    Returns:
        True if the video is valid, otherwise a string explaining the rejection reason.
    """
    #Channel Blacklist Check
    channel_name = (video.get('channel') or '').lower()
    if channel_name in CHANNEL_BLACKLIST:
        return f"Rejected (Blacklisted Channel: {channel_name})"

    #Duration Check
    duration = video.get('duration', 0)
    if duration < 60: return f"Rejected (Too Short: {int(duration)}s)"
    if duration > 900: return f"Rejected (Too Long: {int(duration)}s)"
    
    text = (video.get('title', '') + ' ' + (video.get('description') or '')).lower()
    
    # Date Check (Game Day or Next Day)
    game_date = game_info['game_date_dt']
    next_day_date = game_date + timedelta(days=1)
    
    date_patterns = []
    for d in [game_date, next_day_date]:
        month_full = d.strftime('%B').lower()
        month_abbr = d.strftime('%b').lower()
        day = str(d.day)
        date_patterns.extend([rf'({month_full}|{month_abbr})\s+{day}', rf'{day}\s+({month_full}|{month_abbr})'])
        
    if not any(re.search(pattern, text) for pattern in date_patterns):
        return f"Rejected (Date Mismatch: Expected {game_date.strftime('%b %d')} or {next_day_date.strftime('%b %d')})"
        
    return True


def process_single_game(game_info: Dict) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Worker function to process a single game query. To be run in parallel.
    
    Args:
        game_info: Dictionary containing all info for one game.
        
    Returns:
        A tuple containing (game_info, list_of_valid_videos, list_of_rejected_videos).
    """
    videos = search_youtube_videos(game_info['query'], MAX_RESULTS_PER_QUERY)
    
    valid_videos = []
    rejected_videos = []

    for video in videos:
        validation_result = is_valid_video(video, game_info)
        if validation_result is True:
            valid_videos.append(video)
        elif DEBUG_REJECTIONS:
            video['reason'] = validation_result
            rejected_videos.append(video)
            
    time.sleep(SEARCH_DELAY) 
    return game_info, valid_videos, rejected_videos


def run_parallel_collection(queries: List[Dict]) -> List[Dict]:
    """
    Orchestrate the parallel collection of video data.

    Args:
        queries: A list of game query dictionaries to process.

    Returns:
        A list of dictionaries, each containing a valid video's data and its associated game_id.
    """
    all_valid_videos = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_game = {executor.submit(process_single_game, q): q for q in queries}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_game), total=len(queries), desc="Collecting game videos"):
            original_game_info = future_to_game[future]
            try:
                _, valid_videos, rejected_videos = future.result()
                
                # Log results for this query
                print(f"\n--- Result for search: \"{original_game_info['query']}\" ---")
                print(f"  > Found {len(valid_videos)} valid video(s).")
                
                if DEBUG_REJECTIONS and rejected_videos:
                    print("  > Rejected videos:")
                    for v in rejected_videos:
                        print(f"    - {v['reason']}: {v['title']} ({v['url']})")
                
                # Append game_id to each valid video and add to the final list
                for video in valid_videos:
                    video['game_id'] = original_game_info['game_id']
                    all_valid_videos.append(video)

            except Exception as e:
                print(f"\n[ERROR] Task failed for query '{original_game_info['query']}': {e}")
                
    return all_valid_videos

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline.
    """
    print("Starting the YouTube Highlight Collector script.")
    
    # Load and prepare game data
    try:
        df = pd.read_csv(GAMES_CSV_INPUT)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{GAMES_CSV_INPUT}'. Aborting.")
        return

    df['game_date'] = pd.to_datetime(df['game_date'])
    df['game_date_str'] = df['game_date'].dt.strftime('%b %d %Y')
    
    df['youtube_search_name'] = (
        df['home_team_name'] + ' vs ' + df['visitor_team_name'] + 
        ' Full game highlights | ' + df['game_date_str'] + ' NBA Season'
    )
    
    # Generate search queries
    queries = create_youtube_queries(df)
    
    # Run collection in parallel
    print(f"Starting parallel collection with {MAX_WORKERS} workers for {len(queries)} games...")
    collected_videos = run_parallel_collection(queries)
    
    # Save results to CSV
    if collected_videos:
        output_dir = os.path.dirname(URLS_CSV_OUTPUT)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results_df = pd.DataFrame(collected_videos)
        # Reorder columns for better readability
        results_df = results_df[['game_id', 'title', 'url', 'channel', 'duration', 'video_id']]
        results_df.to_csv(URLS_CSV_OUTPUT, index=False, encoding='utf-8')
        
        print(f"\nSuccessfully saved {len(results_df)} video URLs to '{URLS_CSV_OUTPUT}'")
    else:
        print("\nNo valid videos were found. Output file was not created.")

if __name__ == "__main__":
    main()