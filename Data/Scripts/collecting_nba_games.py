"""
NBA Games Data Collector for Season 2025-2026

This script collects NBA game information via the NBA Stats API and manages
incremental updates. It handles multiple runs by only adding new games to
the existing CSV file, avoiding duplicates.

The script covers the complete 2025-2026 season from regular season through playoffs.

Key features:
- Fetches game data from official NBA Stats API
- Incremental collection: only adds new games on subsequent runs
- Handles both completed and upcoming games
- Includes team names, abbreviations, and scores
- Manages API rate limiting with automatic delays
"""

import requests
import time
from datetime import datetime, timedelta
import os
import pandas as pd
from typing import List, Dict, Tuple, Set

# ============================================================================
# CONFIGURATION
# ============================================================================

# NBA Stats API endpoint and required headers for successful requests
NBA_API_BASE = "https://stats.nba.com/stats"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',  # Required for API acceptance
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',  # API checks referer
    'Origin': 'https://www.nba.com'
}

# NBA Teams mapping (team_id -> team info)
# Complete dictionary of all 30 NBA teams with official IDs
NBA_TEAMS = {
    1610612737: {'name': 'Atlanta Hawks', 'abbreviation': 'ATL'},
    1610612738: {'name': 'Boston Celtics', 'abbreviation': 'BOS'},
    1610612739: {'name': 'Cleveland Cavaliers', 'abbreviation': 'CLE'},
    1610612740: {'name': 'New Orleans Pelicans', 'abbreviation': 'NOP'},
    1610612741: {'name': 'Chicago Bulls', 'abbreviation': 'CHI'},
    1610612742: {'name': 'Dallas Mavericks', 'abbreviation': 'DAL'},
    1610612743: {'name': 'Denver Nuggets', 'abbreviation': 'DEN'},
    1610612744: {'name': 'Golden State Warriors', 'abbreviation': 'GSW'},
    1610612745: {'name': 'Houston Rockets', 'abbreviation': 'HOU'},
    1610612746: {'name': 'LA Clippers', 'abbreviation': 'LAC'},
    1610612747: {'name': 'Los Angeles Lakers', 'abbreviation': 'LAL'},
    1610612748: {'name': 'Miami Heat', 'abbreviation': 'MIA'},
    1610612749: {'name': 'Milwaukee Bucks', 'abbreviation': 'MIL'},
    1610612750: {'name': 'Minnesota Timberwolves', 'abbreviation': 'MIN'},
    1610612751: {'name': 'Brooklyn Nets', 'abbreviation': 'BKN'},
    1610612752: {'name': 'New York Knicks', 'abbreviation': 'NYK'},
    1610612753: {'name': 'Orlando Magic', 'abbreviation': 'ORL'},
    1610612754: {'name': 'Indiana Pacers', 'abbreviation': 'IND'},
    1610612755: {'name': 'Philadelphia 76ers', 'abbreviation': 'PHI'},
    1610612756: {'name': 'Phoenix Suns', 'abbreviation': 'PHX'},
    1610612757: {'name': 'Portland Trail Blazers', 'abbreviation': 'POR'},
    1610612758: {'name': 'Sacramento Kings', 'abbreviation': 'SAC'},
    1610612759: {'name': 'San Antonio Spurs', 'abbreviation': 'SAS'},
    1610612760: {'name': 'Oklahoma City Thunder', 'abbreviation': 'OKC'},
    1610612761: {'name': 'Toronto Raptors', 'abbreviation': 'TOR'},
    1610612762: {'name': 'Utah Jazz', 'abbreviation': 'UTA'},
    1610612763: {'name': 'Memphis Grizzlies', 'abbreviation': 'MEM'},
    1610612764: {'name': 'Washington Wizards', 'abbreviation': 'WAS'},
    1610612765: {'name': 'Detroit Pistons', 'abbreviation': 'DET'},
    1610612766: {'name': 'Charlotte Hornets', 'abbreviation': 'CHA'}
}

# Season date range for 2025-2026 (regular season through playoffs)
SEASON_START = "2025-10-15"  # Approximate regular season start date
SEASON_END = "2026-06-20"    # Approximate playoffs end date (Finals)

# Output file path for collected game data
CSV_FILE = "Data/nba_games_2025_2026.csv"

# API delay to prevent rate limiting (seconds between requests)
API_DELAY = 0.6

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_games_by_date(game_date: str) -> List[Dict]:
    """
    Fetch all NBA games for a specific date from the NBA Stats API.
    Retrieves game metadata including teams, scores, and status.
    
    Args:
        game_date: Date string in 'YYYY-MM-DD' format (e.g., '2025-10-15')
    
    Returns:
        List of dictionaries containing game information for the specified date.
        Each dictionary includes game_id, teams, scores, and metadata.
        Returns empty list if request fails or no games found.
    """
    # Construct API endpoint for scoreboard data
    endpoint = f"{NBA_API_BASE}/scoreboardv2"
    params = {
        'GameDate': game_date,  # Target date for games
        'LeagueID': '00',  # NBA league identifier
        'DayOffset': '0'  # No offset from target date
    }
    
    try:
        # Make HTTP request to NBA Stats API
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        games = []
        
        # Extract game header data (game IDs, teams, status)
        game_header = data['resultSets'][0]['rowSet']
        
        # Extract line score data (individual team scores)
        line_score = data['resultSets'][1]['rowSet']
        
        # Build a dictionary mapping game_id to team scores for efficient lookup
        scores_dict = {}
        for score in line_score:
            game_id = score[1]  # Game identifier
            team_id = score[3]  # Team identifier
            
            # Initialize game entry if not exists
            if game_id not in scores_dict:
                scores_dict[game_id] = {}
            
            # Store team score data
            scores_dict[game_id][team_id] = {
                'team_abbreviation': score[4],  # e.g., 'LAL'
                'team_name': score[5],  # e.g., 'Los Angeles Lakers'
                'pts': score[22]  # Final score
            }
        
        # Process each game and extract relevant information
        for game in game_header:
            game_id = game[2]  # Unique game identifier
            home_team_id = game[6]  # Home team ID
            visitor_team_id = game[7]  # Visiting team ID
            
            # Get team information from the mapping (fallback to 'Unknown' if not found)
            home_team_info = NBA_TEAMS.get(home_team_id, {'name': 'Unknown', 'abbreviation': 'UNK'})
            visitor_team_info = NBA_TEAMS.get(visitor_team_id, {'name': 'Unknown', 'abbreviation': 'UNK'})
            
            # Create structured game information dictionary
            game_info = {
                'game_id': game_id,
                'game_date': game_date,
                'season': game[8],  # Season identifier (e.g., '2025-26')
                'home_team_id': home_team_id,
                'visitor_team_id': visitor_team_id,
                'game_status': game[9],  # Status code (1=scheduled, 2=in-progress, 3=final)
                'home_team_name': home_team_info['name'],
                'visitor_team_name': visitor_team_info['name'],
                'home_team_abbreviation': home_team_info['abbreviation'],
                'visitor_team_abbreviation': visitor_team_info['abbreviation'],
                'home_team_score': None,  # Will be populated if game completed
                'visitor_team_score': None  # Will be populated if game completed
            }
            
            # Populate scores if game has been played (scores available in line_score)
            if game_id in scores_dict:
                if home_team_id in scores_dict[game_id]:
                    game_info['home_team_score'] = scores_dict[game_id][home_team_id]['pts']
                
                if visitor_team_id in scores_dict[game_id]:
                    game_info['visitor_team_score'] = scores_dict[game_id][visitor_team_id]['pts']
            
            games.append(game_info)
        
        return games
    
    except requests.exceptions.RequestException as e:
        # Return empty list on request failure (network errors, timeouts, etc.)
        return []


def load_existing_games() -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load previously collected games from the CSV file.
    Reads existing data to enable incremental updates without duplicates.
    
    Returns:
        Tuple containing:
        - DataFrame with existing game data (empty DataFrame if file doesn't exist)
        - Set of game IDs already collected (for efficient duplicate checking)
    """
    # Check if CSV file exists from previous runs
    if os.path.exists(CSV_FILE):
        try:
            # Read existing game data
            df = pd.read_csv(CSV_FILE)
            
            # Extract game IDs and convert to set for O(1) lookup
            existing_ids = set(df['game_id'].astype(str).tolist())
            
            return df, existing_ids
        
        except Exception as e:
            # Return empty data if file is corrupted or unreadable
            return pd.DataFrame(), set()
    else:
        # No existing data - first run of the script
        return pd.DataFrame(), set()


def collect_new_games(existing_ids: Set[str]) -> List[Dict]:
    """
    Collect new NBA games from the 2025-2026 season that aren't already in the dataset.
    Iterates through each day from season start to current date (or season end).
    
    Args:
        existing_ids: Set of game IDs already collected to avoid duplicates
    
    Returns:
        List of dictionaries containing information for newly found games
    """
    new_games = []
    
    # Parse season start date
    current_date = datetime.strptime(SEASON_START, '%Y-%m-%d')
    
    # Collect up to current date (don't fetch future games beyond today)
    end_date = datetime.now()
    season_end = datetime.strptime(SEASON_END, '%Y-%m-%d')
    
    # Don't search beyond the season end date
    if end_date > season_end:
        end_date = season_end
    
    # Calculate total days to process for progress tracking
    total_days = (end_date - current_date).days + 1
    days_processed = 0
    
    # Iterate through each day of the date range
    while current_date <= end_date:
        # Format current date for API request
        date_str = current_date.strftime('%Y-%m-%d')
        days_processed += 1
        
        # Fetch games for current date
        games = get_games_by_date(date_str)
        
        # Filter out games that already exist in the dataset (incremental update)
        for game in games:
            if str(game['game_id']) not in existing_ids:
                new_games.append(game)
        
        # Pause to avoid overwhelming the API (rate limiting prevention)
        time.sleep(API_DELAY)
        
        # Move to next day
        current_date += timedelta(days=1)
    
    return new_games


def save_games_to_csv(new_games_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Save games to CSV file by combining new data with existing data.
    Creates the Data directory if it doesn't exist.
    
    Args:
        new_games_df: DataFrame containing newly collected games
        existing_df: DataFrame containing previously collected games
    
    Returns:
        Combined DataFrame with all games (existing + new)
    """
    # Create output directory if needed
    os.makedirs('Data', exist_ok=True)
    
    # Combine existing and new data
    if not existing_df.empty:
        # Append new games to existing dataset
        combined_df = pd.concat([existing_df, new_games_df], ignore_index=True)
    else:
        # First run - use only new games
        combined_df = new_games_df
    
    # Sort by date and game ID for better readability and chronological order
    combined_df = combined_df.sort_values(['game_date', 'game_id'])
    
    # Save to CSV with UTF-8 encoding to handle special characters
    combined_df.to_csv(CSV_FILE, index=False, encoding='utf-8')
    
    return combined_df


def display_summary(df: pd.DataFrame) -> None:
    """
    Display a summary of the collected data including statistics and date range.
    Shows total games, completed vs upcoming games, and date coverage.
    
    Args:
        df: DataFrame containing all collected games
    
    Returns:
        None
    """
    # Check if DataFrame is empty
    if df.empty:
        print("No games collected yet.")
        return
    
    # Separate completed games from upcoming games (based on score availability)
    finished = df[df['home_team_score'].notna()]
    upcoming = df[df['home_team_score'].isna()]
    
    # Calculate summary statistics
    total_games = len(df)
    completed_games = len(finished)
    upcoming_games = len(upcoming)
    first_date = df['game_date'].min()  # Earliest game in dataset
    last_date = df['game_date'].max()  # Latest game in dataset
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Display summary report
    print("\n" + "="*60)
    print("NBA GAMES DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Total games collected: {total_games}")
    print(f"  - Completed games: {completed_games}")
    print(f"  - Upcoming games: {upcoming_games}")
    print(f"Date range: {first_date} to {last_date}")
    print(f"Last updated: {last_updated}")
    print(f"Output file: {CSV_FILE}")
    print("="*60 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution function.
    Orchestrates the complete data collection workflow:
    1. Load existing data
    2. Collect new games
    3. Save updated dataset
    4. Display summary statistics
    
    Returns:
        None
    """
    print("Starting NBA Games Data Collection for 2025-2026 Season...")
    
    # Load previously collected games (if any)
    existing_df, existing_ids = load_existing_games()
    
    print(f"Found {len(existing_ids)} existing games in dataset.")
    print("Collecting new games from NBA Stats API...")
    
    # Collect new games not in the dataset
    new_games = collect_new_games(existing_ids)
    
    # Process and save results
    if new_games:
        print(f"Found {len(new_games)} new games to add.")
        
        # Convert list to DataFrame
        new_games_df = pd.DataFrame(new_games)
        
        # Save combined dataset
        combined_df = save_games_to_csv(new_games_df, existing_df)
        
        # Display collection summary
        display_summary(combined_df)
    else:
        print("No new games found. Dataset is up to date.")
        
        # Display summary of existing data
        display_summary(existing_df)


if __name__ == "__main__":
    main()