"""
Double Updater: YouTube Stats & Media Value Recalculation
=========================================================

This script performs two critical functions:
1. Updates the "Urls & Stats" CSV with the latest live view counts from YouTube.
2. Updates the "Exposure Results" CSV by mathematically adjusting the 'total_media_value'
   based on the view count increase, without re-running video processing.

Safety Features:
- If a video is banned/deleted, it keeps the old view count and value (does not zero them out).
- Threaded for speed.
- Incremental saving.
"""

import pandas as pd
import yt_dlp
import os
import time
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import csv
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# FILE 1: The list of URLs and View Counts
VIEWS_CSV_INPUT = "Data/urls/game_highlight_urls_2025_26.csv"
VIEWS_CSV_OUTPUT = "Data/urls/game_highlight_urls_2025_26_UPDATED.csv"

# FILE 2: The intense computer vision results
RESULTS_CSV_INPUT = "Data/exposure_and_game_info/exposure_results_2025_26.csv"
RESULTS_CSV_OUTPUT = "Data/exposure_and_game_info/exposure_results_2025_26_UPDATED.csv"

# Performance Settings
MAX_WORKERS = 32       # Number of simultaneous checks
REQUEST_DELAY = 0    # Delay to avoid rate limits
SAVE_EVERY = 50        # Save progress frequency

# ============================================================================
# PART 1: FETCH LATEST YOUTUBE STATS
# ============================================================================

def get_latest_stats(video_id: str):
    """
    Fetches current stats. Returns dict with status='Available' or 'Unavailable'.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False, # Need full info for live view counts
        'skip_download': True,
        'ignoreerrors': True,  # Critical: Don't crash on banned videos
        'cookiesfrombrowser': ('firefox',), # Optional: helps with restricted videos
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if info:
                return {
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'comment_count': info.get('comment_count', 0),
                    'status': 'Available'
                }
    except Exception:
        pass
    
    return {'status': 'Unavailable'}

def process_view_update(row_tuple):
    """Worker to update a single row of the Views CSV."""
    index, row = row_tuple
    
    # Copy original data to preserve it
    updated_data = row.to_dict()
    
    # Store the OLD view count for ratio calculation later
    # (Handle cases where 'view_count' might be NaN or string initially)
    try:
        old_views = float(row.get('view_count', 0))
        if pd.isna(old_views): old_views = 0
    except:
        old_views = 0
        
    updated_data['old_view_count_ref'] = old_views

    # Extract Video ID
    video_id = row.get('video_id')
    url = row.get('url')
    
    if pd.isna(video_id) and isinstance(url, str):
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
            
    updated_data['video_id'] = video_id # Ensure ID is set

    if video_id:
        stats = get_latest_stats(str(video_id))
        
        if stats['status'] == 'Available':
            # Video is Live: Update stats
            updated_data['view_count'] = stats['view_count']
            updated_data['like_count'] = stats['like_count']
            updated_data['comment_count'] = stats['comment_count']
            updated_data['video_status'] = 'Available'
        else:
            # Video Dead: Keep old stats, mark unavailable
            updated_data['video_status'] = 'Unavailable/Deleted'
            # We keep 'view_count' as is (old value)
    else:
        updated_data['video_status'] = 'No ID Found'

    updated_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time.sleep(REQUEST_DELAY)
    
    return updated_data

# ============================================================================
# PART 2: RECALCULATE MEDIA VALUE
# ============================================================================

def update_media_values(views_df, results_path_in, results_path_out):
    """
    Reads the Results CSV and updates 'total_media_value' using the ratio:
    NewValue = OldValue * (NewViews / OldViews)
    """
    print("\n" + "-"*60)
    print("STEP 2: UPDATING MEDIA VALUES")
    print("-"*60)
    
    if not os.path.exists(results_path_in):
        print(f"Skipping Step 2: Results file {results_path_in} not found.")
        return

    # 1. Create a lookup dictionary: video_id -> (old_views, new_views)
    # Filter views_df to only rows where we have valid IDs
    valid_views = views_df[views_df['video_id'].notna()]
    
    # Create mapping
    view_map = {}
    for _, row in valid_views.iterrows():
        vid = str(row['video_id'])
        old_v = float(row.get('old_view_count_ref', 0))
        new_v = float(row.get('view_count', 0))
        view_map[vid] = {'old': old_v, 'new': new_v}

    # 2. Load Results File
    print(f"Loading results from: {results_path_in}")
    results_df = pd.read_csv(results_path_in)
    
    # Ensure video_id is string for matching
    results_df['video_id'] = results_df['video_id'].astype(str)
    
    updated_count = 0
    unchanged_count = 0
    
    # 3. Update Logic
    def calculate_new_value(row):
        nonlocal updated_count, unchanged_count
        
        vid = row['video_id']
        current_value = row.get('total_media_value', 0)
        
        if vid in view_map:
            views = view_map[vid]
            
            # CASE A: Video was unavailable or views didn't change
            if views['new'] == views['old']:
                unchanged_count += 1
                return current_value
            
            # CASE B: Old views were 0 (Mathematically cannot scale up from 0)
            if views['old'] == 0:
                # If we have new views but old were 0, we can't calculate the ratio multiplier.
                # We return the old value (likely 0) to be safe.
                unchanged_count += 1
                return current_value
                
            # CASE C: Standard Update
            ratio = views['new'] / views['old']
            new_value = current_value * ratio
            updated_count += 1
            return new_value
            
        return current_value

    # Apply calculation
    results_df['total_media_value'] = results_df.apply(calculate_new_value, axis=1)
    
    # 4. Save
    results_df.to_csv(results_path_out, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Updated {updated_count} rows with new values.")
    print(f"Left {unchanged_count} rows unchanged (deleted videos or 0 views).")
    print(f"Saved to: {results_path_out}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("DUAL FILE UPDATER")
    print("="*60)

    # --- STEP 1: UPDATE VIEW COUNTS ---
    print("\nSTEP 1: FETCHING LATEST YOUTUBE STATS")
    
    if not os.path.exists(VIEWS_CSV_INPUT):
        print(f"Error: Views input file '{VIEWS_CSV_INPUT}' not found.")
        return

    views_df = pd.read_csv(VIEWS_CSV_INPUT)
    
    # Ensure columns exist
    for col in ['view_count', 'like_count', 'comment_count', 'video_status', 'last_updated']:
        if col not in views_df.columns:
            views_df[col] = None

    updated_rows = []
    
    print(f"Scanning {len(views_df)} videos with {MAX_WORKERS} threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(process_view_update, row): row for row in views_df.iterrows()}
        
        with tqdm(total=len(views_df), desc="Updating Views") as pbar:
            for i, future in enumerate(concurrent.futures.as_completed(future_to_row)):
                try:
                    result_row = future.result()
                    updated_rows.append(result_row)
                except Exception as e:
                    print(f"Error: {e}")
                    # Keep original if crash
                    updated_rows.append(future_to_row[future][1].to_dict())
                
                pbar.update(1)

                # Incremental Save of Views File
                if i % SAVE_EVERY == 0:
                    pd.DataFrame(updated_rows).to_csv(VIEWS_CSV_OUTPUT, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Final Save of Views File
    final_views_df = pd.DataFrame(updated_rows)
    if 'game_id' in final_views_df.columns:
        final_views_df = final_views_df.sort_values(by='game_id')
    
    final_views_df.to_csv(VIEWS_CSV_OUTPUT, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Views file saved to: {VIEWS_CSV_OUTPUT}")

    # --- STEP 2: UPDATE VALUE FILE ---
    update_media_values(final_views_df, RESULTS_CSV_INPUT, RESULTS_CSV_OUTPUT)

    print("\n" + "="*60)
    print("ALL UPDATES COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()