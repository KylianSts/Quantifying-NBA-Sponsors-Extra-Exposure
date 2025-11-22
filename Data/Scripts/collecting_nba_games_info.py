"""
NBA Games and Player Stats Collector (Parallel Version)

This script collects NBA game data and detailed player statistics from the official
NBA Stats API. It uses parallel processing to speed up data collection while
respecting API rate limits.

Key features:
- Retrieves all games for a specified NBA season
- Formats game data with clear Home vs Away team structure
- Collects detailed player statistics for each game (box scores)
- Uses parallel processing with ThreadPoolExecutor for faster data collection
- Includes API rate limiting to avoid being blocked
- Saves results to CSV files with UTF-8 encoding
- Provides progress tracking and error handling
"""

import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Optional
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Target NBA season to collect data for (format: "YYYY-YY")
TARGET_SEASON = "2025-26"

# Season type: "Regular Season", "Playoffs", "Pre Season", "All Star"
SEASON_TYPE = "Regular Season"

# Output file paths for collected data
GAMES_OUTPUT_FILE = f"Data/nba_games_{TARGET_SEASON}.csv"
PLAYERS_OUTPUT_FILE = f"Data/nba_players_stats_{TARGET_SEASON}.csv"

# API rate limiting configuration
API_DELAY = 0.2  # Delay between API calls in seconds (to avoid rate limiting)
MAX_WORKERS = 16  # Number of parallel workers (keep low to respect API limits)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_games_list(season: str, season_type: str) -> pd.DataFrame:
    """
    Retrieve the list of all games for a season and format the table
    to have one unique row per game (Home vs Away structure).
    
    The NBA API returns two rows per game (one for each team), so this function
    merges them into a single row with clear Home/Away prefixes.
    
    Args:
        season: NBA season in format "YYYY-YY" (e.g., "2025-26")
        season_type: Type of season ("Regular Season", "Playoffs", etc.)
    
    Returns:
        DataFrame with one row per game, containing:
            - GAME_ID: Unique game identifier
            - GAME_DATE: Date of the game
            - HOME_* columns: All stats for the home team
            - AWAY_* columns: All stats for the away team
    """
    print(f"\n--- Retrieving games for season {season} ({season_type}) ---")
    
    # Call NBA Stats API to get all games for the season
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type
    )
    games_df = gamefinder.get_data_frames()[0]
    
    # Remove unnecessary columns for the merge
    games_df = games_df.drop(columns=['SEASON_ID', 'WL'])
    
    # Separate Home and Away games based on MATCHUP format
    # 'vs.' indicates home team (e.g., "LAL vs. BOS")
    # '@' indicates away team (e.g., "BOS @ LAL")
    mask_home = games_df['MATCHUP'].str.contains(' vs. ')
    
    df_home = games_df[mask_home].copy()
    df_away = games_df[~mask_home].copy()

    # Define columns that should not be prefixed (common to both teams)
    common_columns = ['GAME_ID', 'GAME_DATE', 'MIN']
    
    # Remove MATCHUP column (redundant after identifying home/away)
    df_home = df_home.drop(columns=['MATCHUP'])
    df_away = df_away.drop(columns=['MATCHUP'])

    # Add HOME_ and AWAY_ prefixes to distinguish team-specific columns
    df_home.columns = ['HOME_' + col if col not in common_columns else col for col in df_home.columns]
    df_away.columns = ['AWAY_' + col if col not in common_columns else col for col in df_away.columns]

    # Merge home and away data on GAME_ID to create one row per game
    # Drop duplicate common columns from away DataFrame before merge
    final_games_df = pd.merge(
        df_home, 
        df_away.drop(columns=common_columns[1:]),  # Keep only GAME_ID for merge
        on="GAME_ID"
    )
    
    print(f"Number of unique games found: {len(final_games_df)}")
    
    return final_games_df


def fetch_single_game_stats(game_id: str) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Worker function to fetch player statistics for a single game.
    Designed for parallel execution with ThreadPoolExecutor.
    
    Args:
        game_id: Unique NBA game identifier (e.g., "0022500001")
    
    Returns:
        Tuple of (game_id, DataFrame with player stats or None if failed)
    """
    try:
        # Call NBA Stats API for box score data
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        stats_df = boxscore.player_stats.get_data_frame()
        
        # Small delay to respect API rate limits
        time.sleep(API_DELAY)
        
        return game_id, stats_df
    
    except Exception as e:
        # Return None for failed requests (will be logged)
        return game_id, None


def get_players_stats_parallel(game_ids: List[str], max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    """
    Retrieve detailed player statistics for multiple games using parallel processing.
    Uses ThreadPoolExecutor to speed up data collection while respecting API limits.
    
    Args:
        game_ids: List of unique NBA game identifiers to process
        max_workers: Number of parallel threads (default: 8, keep low for API limits)
    
    Returns:
        DataFrame containing player statistics for all successfully fetched games,
        with columns including GAME_ID, PLAYER_ID, PLAYER_NAME, and all box score stats
    """
    print(f"\n--- Retrieving player stats for {len(game_ids)} games ---")
    print(f"Using {max_workers} parallel workers")
    print("Note: This may take some time due to API rate limiting.\n")
    
    # Lists to track results and errors
    all_player_stats = []
    failed_games = []
    
    # Process games in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all game fetch tasks to the thread pool
        futures = {
            executor.submit(fetch_single_game_stats, game_id): game_id 
            for game_id in game_ids
        }
        
        # Process results as they complete (with progress bar)
        for future in tqdm(as_completed(futures), total=len(game_ids), desc="Fetching player stats"):
            game_id = futures[future]
            
            try:
                # Get result from completed task
                returned_game_id, stats_df = future.result()
                
                if stats_df is not None and not stats_df.empty:
                    # Successfully retrieved stats, add to collection
                    all_player_stats.append(stats_df)
                else:
                    # API returned empty data
                    failed_games.append(returned_game_id)
                    
            except Exception as e:
                # Task raised an exception
                failed_games.append(game_id)
                print(f"\n[ERROR] Failed to fetch game {game_id}: {e}")
    
    # Log summary of failed games
    if failed_games:
        print(f"\n[WARNING] Failed to retrieve {len(failed_games)} games")
        print(f"Failed game IDs: {failed_games[:10]}...")  # Show first 10
    
    # Concatenate all successful results into single DataFrame
    if all_player_stats:
        combined_df = pd.concat(all_player_stats, ignore_index=True)
        print(f"\nSuccessfully retrieved stats for {len(all_player_stats)} games")
        print(f"Total player records: {len(combined_df)}")
        return combined_df
    else:
        # Return empty DataFrame if all requests failed
        print("\n[ERROR] No player data was retrieved")
        return pd.DataFrame()


def save_dataframe_to_csv(df: pd.DataFrame, filepath: str, description: str) -> bool:
    """
    Save a DataFrame to CSV file with error handling.
    Creates output directory if it doesn't exist.
    
    Args:
        df: DataFrame to save
        filepath: Path where CSV file will be saved
        description: Human-readable description for logging (e.g., "games data")
    
    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV with UTF-8 encoding
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"{description} saved to: {filepath}")
        return True
    
    except Exception as e:
        print(f"Failed to save {description}: {e}")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution pipeline for NBA data collection.
    
    Workflow:
    1. Retrieve all games for the target season
    2. Save games data to CSV
    3. Retrieve detailed player statistics for each game (in parallel)
    4. Save player statistics to CSV
    5. Display summary and column information
    
    Returns:
        None
    """
    print("="*60)
    print("NBA DATA COLLECTION PIPELINE")
    print("="*60)
    print(f"Target Season: {TARGET_SEASON}")
    print(f"Season Type: {SEASON_TYPE}")
    print("="*60)
    
    # ========================================================================
    # STEP 1: Retrieve and save game data
    # ========================================================================
    print("\n[STEP 1/2] Collecting Games Data")
    print("-"*60)
    
    df_games = get_games_list(TARGET_SEASON, SEASON_TYPE)
    
    # Check if games were found
    if df_games.empty:
        print("No games found. Please verify the target season.")
        print("The season may not have started yet or the format may be incorrect.")
        return
    
    # Save games data to CSV
    import os
    save_dataframe_to_csv(df_games, GAMES_OUTPUT_FILE, "Games data")
    
    # Display games DataFrame columns
    print(f"\nGames data columns ({len(df_games.columns)} total):")
    print(df_games.columns.tolist())
    
    # ========================================================================
    # STEP 2: Retrieve and save player statistics
    # ========================================================================
    print("\n[STEP 2/2] Collecting Player Statistics")
    print("-"*60)
    
    # Get unique game IDs from games DataFrame
    unique_game_ids = df_games['GAME_ID'].unique().tolist()
    print(f"Processing {len(unique_game_ids)} unique games")
    
    # Uncomment the line below to test with only 5 games
    # unique_game_ids = unique_game_ids[:5]
    
    # Retrieve player statistics in parallel
    df_players = get_players_stats_parallel(unique_game_ids, max_workers=MAX_WORKERS)
    
    # Save player statistics if data was retrieved
    if not df_players.empty:
        save_dataframe_to_csv(df_players, PLAYERS_OUTPUT_FILE, "Player stats")
        
        # Display player stats DataFrame columns
        print(f"\nPlayer stats columns ({len(df_players.columns)} total):")
        print(df_players.columns.tolist())
    else:
        print("No player data retrieved. Check for API errors above.")
    
    # ========================================================================
    # Display completion summary
    # ========================================================================
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"Games collected: {len(df_games)}")
    print(f"Player records collected: {len(df_players) if not df_players.empty else 0}")
    print(f"Games output: {GAMES_OUTPUT_FILE}")
    print(f"Players output: {PLAYERS_OUTPUT_FILE}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()