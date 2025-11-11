"""
NBA Games Data Collector for Season 2025-2026

This script collects NBA game information via the NBA Stats API and manages
incremental updates. It handles multiple runs by only adding new games to
the existing CSV file, avoiding duplicates.

The script covers the complete 2025-2026 season from regular season through playoffs.
"""

import requests
import json
import time
from datetime import datetime, timedelta
import os
import pandas as pd
from typing import List, Dict, Tuple, Set

# API Configuration
NBA_API_BASE = "https://stats.nba.com/stats"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com'
}

# NBA Teams mapping (team_id -> team info)
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

# Season Configuration
SEASON_START = "2025-10-15"  # Approximate regular season start date
SEASON_END = "2026-06-20"    # Approximate playoffs end date (Finals)
CSV_FILE = "Data/nba_games_2025_2026.csv"
API_DELAY = 0.6  # Delay between API calls in seconds to avoid rate limiting


def get_games_by_date(game_date: str) -> List[Dict]:
    """
    Fetch all NBA games for a specific date from the NBA Stats API.
    
    Args:
        game_date: Date string in 'YYYY-MM-DD' format
    
    Returns:
        List of dictionaries containing game information for the specified date.
        Returns empty list if request fails or no games found.
    """
    endpoint = f"{NBA_API_BASE}/scoreboardv2"
    params = {
        'GameDate': game_date,
        'LeagueID': '00',
        'DayOffset': '0'
    }
    
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        games = []
        game_header = data['resultSets'][0]['rowSet']
        line_score = data['resultSets'][1]['rowSet']
        
        # Build a dictionary mapping game_id to team scores for efficient lookup
        scores_dict = {}
        for score in line_score:
            game_id = score[1]
            team_id = score[3]
            if game_id not in scores_dict:
                scores_dict[game_id] = {}
            scores_dict[game_id][team_id] = {
                'team_abbreviation': score[4],
                'team_name': score[5],
                'pts': score[22]
            }
        
        # Process each game and extract relevant information
        for game in game_header:
            game_id = game[2]
            home_team_id = game[6]
            visitor_team_id = game[7]
            
            # Get team information from the mapping
            home_team_info = NBA_TEAMS.get(home_team_id, {'name': 'Unknown', 'abbreviation': 'UNK'})
            visitor_team_info = NBA_TEAMS.get(visitor_team_id, {'name': 'Unknown', 'abbreviation': 'UNK'})
            
            game_info = {
                'game_id': game_id,
                'game_date': game_date,
                'season': game[8],
                'home_team_id': home_team_id,
                'visitor_team_id': visitor_team_id,
                'game_status': game[9],
                'home_team_name': home_team_info['name'],
                'visitor_team_name': visitor_team_info['name'],
                'home_team_abbreviation': home_team_info['abbreviation'],
                'visitor_team_abbreviation': visitor_team_info['abbreviation'],
                'home_team_score': None,
                'visitor_team_score': None
            }
            
            # Populate scores if game has been played
            if game_id in scores_dict:
                if home_team_id in scores_dict[game_id]:
                    game_info['home_team_score'] = scores_dict[game_id][home_team_id]['pts']
                
                if visitor_team_id in scores_dict[game_id]:
                    game_info['visitor_team_score'] = scores_dict[game_id][visitor_team_id]['pts']
            
            games.append(game_info)
        
        return games
    
    except requests.exceptions.RequestException as e:
        return []


def load_existing_games() -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load previously collected games from the CSV file.
    
    Returns:
        Tuple containing:
        - DataFrame with existing game data
        - Set of game IDs already collected (for efficient duplicate checking)
    """
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            existing_ids = set(df['game_id'].astype(str).tolist())
            return df, existing_ids
        except Exception as e:
            return pd.DataFrame(), set()
    else:
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
    current_date = datetime.strptime(SEASON_START, '%Y-%m-%d')
    end_date = datetime.now()  # Collect up to current date
    season_end = datetime.strptime(SEASON_END, '%Y-%m-%d')
    
    # Don't search beyond the season end date
    if end_date > season_end:
        end_date = season_end
    
    total_days = (end_date - current_date).days + 1
    days_processed = 0
    
    # Iterate through each day of the date range
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        days_processed += 1
        
        # Fetch games for current date
        games = get_games_by_date(date_str)
        
        # Filter out games that already exist in the dataset
        for game in games:
            if str(game['game_id']) not in existing_ids:
                new_games.append(game)
        
        # Pause to avoid overwhelming the API
        time.sleep(API_DELAY)
        
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
        combined_df = pd.concat([existing_df, new_games_df], ignore_index=True)
    else:
        combined_df = new_games_df
    
    # Sort by date and game ID for better readability
    combined_df = combined_df.sort_values(['game_date', 'game_id'])
    
    # Save to CSV with UTF-8 encoding
    combined_df.to_csv(CSV_FILE, index=False, encoding='utf-8')
    
    return combined_df


def display_summary(df: pd.DataFrame) -> None:
    """
    Display a summary of the collected data including statistics and date range.
    
    Args:
        df: DataFrame containing all collected games
    
    Returns:
        None
    """
    if df.empty:
        return
    
    # Separate completed games from upcoming games
    finished = df[df['home_team_score'].notna()]
    upcoming = df[df['home_team_score'].isna()]
    
    # Calculate summary statistics
    total_games = len(df)
    completed_games = len(finished)
    upcoming_games = len(upcoming)
    first_date = df['game_date'].min()
    last_date = df['game_date'].max()
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


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
    # Load previously collected games
    existing_df, existing_ids = load_existing_games()
    
    # Collect new games not in the dataset
    new_games = collect_new_games(existing_ids)
    
    if new_games:
        # Convert list to DataFrame
        new_games_df = pd.DataFrame(new_games)
        
        # Save combined dataset
        combined_df = save_games_to_csv(new_games_df, existing_df)
        
        # Display collection summary
        display_summary(combined_df)
    else:
        display_summary(existing_df)


if __name__ == "__main__":
    main()