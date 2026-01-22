import matplotlib
# Use the 'Agg' backend for Matplotlib to save plots to files without a display interface
# This is essential when running on servers or in headless environments.
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================
warnings.filterwarnings("ignore")

# Define where models and visual artifacts will be saved
OUTPUT_DIR = "Models/models_results/value_prediction_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================================
# 1. DATA LOADING & MERGING
# ============================================================================
def load_and_merge_data():
    """
    Loads raw data sources (metadata, exposure, game info), aggregates target variables,
    and merges them into a single dataframe.
    
    Returns:
        pd.DataFrame: A cleaned dataframe containing game schedules and target values 
                      (media value & views), sorted chronologically.
    """
    print("Loading data...")
    
    # 1. Load Metadata (Contains View Counts)
    # Group by Game ID to get total views across all highlight clips for that game
    df_metadata = pd.read_csv("Data/urls/game_highlight_urls_2025_26_UPDATED.csv")
    view_per_game = df_metadata.groupby('game_id')['view_count'].sum()

    # 2. Load Exposure Data (Contains Calculated Media Value)
    # Group by Game ID to get total media valuation
    df_exposure = pd.read_csv("Data/exposure_and_game_info/exposure_results_2025_26_UPDATED.csv")
    value_per_game = df_exposure.groupby('game_id')['total_media_value'].sum()

    # 3. Load Base Game Schedule/Info
    df_game_info = pd.read_csv("Data/exposure_and_game_info/nba_games_2025-26.csv")

    # 4. Map Targets to Game Info
    # This aligns the target variables (y) with the game features (X)
    df_game_info['view_per_game'] = df_game_info['GAME_ID'].map(view_per_game)
    df_game_info['media_value_per_game'] = df_game_info['GAME_ID'].map(value_per_game)

    # Remove rows with missing targets (likely G-League or preseason games not tracked)
    df_game_info = df_game_info.dropna()

    # 5. Drop Post-Match Box Score Statistics
    # These stats (FG%, Rebounds, etc.) are only known AFTER the game. 
    # We drop them to prevent "Look-Ahead Bias" / Data Leakage.
    stats_to_drop  = ['MIN', 'HOME_FGM', 'HOME_FGA', 'HOME_FG_PCT',
                      'HOME_FG3M', 'HOME_FG3A', 'HOME_FG3_PCT', 'HOME_FTM',
                      'HOME_FTA', 'HOME_FT_PCT', 'HOME_REB', 'HOME_PF',
                      'AWAY_FGM', 'AWAY_FGA', 'AWAY_FG_PCT',
                      'AWAY_FG3M', 'AWAY_FG3A', 'AWAY_FG3_PCT', 'AWAY_FTM',
                      'AWAY_FTA', 'AWAY_FT_PCT', 'AWAY_REB', 'AWAY_PF']
    
    # Safely drop columns only if they exist
    cols_to_drop = [c for c in stats_to_drop if c in df_game_info.columns]
    df_game_info = df_game_info.drop(columns=cols_to_drop)

    # 6. Chronological Sorting
    # Essential for time-series modeling (past predicts future)
    df_game_info['GAME_DATE'] = pd.to_datetime(df_game_info['GAME_DATE'])
    df_game_info = df_game_info.sort_values(by='GAME_DATE', ascending=True).reset_index(drop=True)

    return df_game_info

# ============================================================================
# 2. FEATURE ENGINEERING: HISTORICAL STATE (PRE-MATCH)
# ============================================================================
def calculate_pre_match_stats(df_game_info):
    """
    Iteratively calculates team states (wins, losses, streaks) AS THEY STOOD
    before each match began.
    
    CRITICAL: This must be done iteratively or with careful windowing to ensure
    we don't use the result of Game N to predict Game N.
    """
    print("Calculating pre-match statistics...")
    
    # Helper: Determine actual winners for updating state *after* prediction point
    df_game_info['HOME_WIN'] = (df_game_info['HOME_PTS'] > df_game_info['AWAY_PTS']).astype(int)
    df_game_info['AWAY_WIN'] = (df_game_info['AWAY_PTS'] > df_game_info['HOME_PTS']).astype(int)

    # Initialize columns for features "known at kickoff"
    for col in ['HOME_TEAM_wins_before', 'HOME_TEAM_losses_before', 
                'AWAY_TEAM_wins_before', 'AWAY_TEAM_losses_before',
                'HOME_TEAM_streak_before', 'AWAY_TEAM_streak_before']:
        df_game_info[col] = 0

    # Define East Conference IDs (Hardcoded lookup)
    east_teams = {
        1610612738, 1610612751, 1610612752, 1610612755, 1610612761, 
        1610612741, 1610612739, 1610612765, 1610612754, 1610612749, 
        1610612737, 1610612766, 1610612748, 1610612753, 1610612764
    }
    df_game_info['HOME_TEAM_is_east'] = df_game_info['HOME_TEAM_ID'].isin(east_teams).astype(int)
    df_game_info['AWAY_TEAM_is_east'] = df_game_info['AWAY_TEAM_ID'].isin(east_teams).astype(int)

    # Iterative State Update Loop
    # We walk through the schedule chronologically.
    team_stats = {}

    for idx, row in df_game_info.iterrows():
        home_team = row['HOME_TEAM_ID']
        away_team = row['AWAY_TEAM_ID']
        
        # Initialize team if seen for first time
        if home_team not in team_stats: team_stats[home_team] = {'wins': 0, 'losses': 0, 'streak': 0}
        if away_team not in team_stats: team_stats[away_team] = {'wins': 0, 'losses': 0, 'streak': 0}
        
        # 1. READ State: Assign current stats to the dataframe (Features for Model)
        df_game_info.at[idx, 'HOME_TEAM_wins_before'] = team_stats[home_team]['wins']
        df_game_info.at[idx, 'HOME_TEAM_losses_before'] = team_stats[home_team]['losses']
        df_game_info.at[idx, 'AWAY_TEAM_wins_before'] = team_stats[away_team]['wins']
        df_game_info.at[idx, 'AWAY_TEAM_losses_before'] = team_stats[away_team]['losses']
        df_game_info.at[idx, 'HOME_TEAM_streak_before'] = team_stats[home_team]['streak']
        df_game_info.at[idx, 'AWAY_TEAM_streak_before'] = team_stats[away_team]['streak']
        
        # 2. UPDATE State: Update stats based on this game's result (For NEXT Model)
        if row['HOME_WIN'] == 1:
            team_stats[home_team]['wins'] += 1
            team_stats[away_team]['losses'] += 1
            # Update streaks (Positive for Win streak, Negative for Loss streak)
            team_stats[home_team]['streak'] = team_stats[home_team]['streak'] + 1 if team_stats[home_team]['streak'] >= 0 else 1
            team_stats[away_team]['streak'] = team_stats[away_team]['streak'] - 1 if team_stats[away_team]['streak'] <= 0 else -1
        else:
            team_stats[home_team]['losses'] += 1
            team_stats[away_team]['wins'] += 1
            # Update streaks
            team_stats[home_team]['streak'] = team_stats[home_team]['streak'] - 1 if team_stats[home_team]['streak'] <= 0 else -1
            team_stats[away_team]['streak'] = team_stats[away_team]['streak'] + 1 if team_stats[away_team]['streak'] >= 0 else 1

    # Calculate Win Percentages
    df_game_info['HOME_TEAM_games_before'] = df_game_info['HOME_TEAM_wins_before'] + df_game_info['HOME_TEAM_losses_before']
    df_game_info['AWAY_TEAM_games_before'] = df_game_info['AWAY_TEAM_wins_before'] + df_game_info['AWAY_TEAM_losses_before']

    # Handle division by zero for the first game of the season (default to 0.5)
    df_game_info['HOME_TEAM_winpct_before'] = df_game_info.apply(
        lambda x: x['HOME_TEAM_wins_before'] / x['HOME_TEAM_games_before'] if x['HOME_TEAM_games_before'] > 0 else 0.5, axis=1
    )
    df_game_info['AWAY_TEAM_winpct_before'] = df_game_info.apply(
        lambda x: x['AWAY_TEAM_wins_before'] / x['AWAY_TEAM_games_before'] if x['AWAY_TEAM_games_before'] > 0 else 0.5, axis=1
    )
    
    return df_game_info

# ============================================================================
# 3. FEATURE ENGINEERING: ROLLING AVERAGES
# ============================================================================
def calculate_rolling_averages(df_game_info, window=5):
    """
    Transforms data to Long Format to calculate 'Last 5 Games' statistics for each team,
    then merges these features back into the Match (Wide) dataframe.
    """
    
    print(f"Calculating rolling averages (Window={window})...")
    
    # A. Transform to Long Format (1 row per team per game)
    # This makes grouping by Team ID and rolling much easier.
    df_unique = df_game_info.drop_duplicates(['GAME_ID']).copy()
    
    # Extract Home Data
    cols_home = [c for c in df_unique.columns if "HOME_" in c]
    df_home = df_unique[['GAME_ID', 'GAME_DATE', 'view_per_game', 'media_value_per_game'] + cols_home].copy()
    df_home = df_home.rename(columns={c: c.replace('HOME_', '') for c in cols_home})
    if 'TEAM_ID' not in df_home.columns and 'HOME_TEAM_ID' in df_unique.columns:
        df_home['TEAM_ID'] = df_unique['HOME_TEAM_ID']
    
    # Extract Away Data
    cols_away = [c for c in df_unique.columns if "AWAY_" in c]
    df_away = df_unique[['GAME_ID', 'GAME_DATE', 'view_per_game', 'media_value_per_game'] + cols_away].copy()
    df_away = df_away.rename(columns={c: c.replace('AWAY_', '') for c in cols_away})
    if 'TEAM_ID' not in df_away.columns and 'AWAY_TEAM_ID' in df_unique.columns:
        df_away['TEAM_ID'] = df_unique['AWAY_TEAM_ID']
    
    # Concatenate to get a single timeline per team
    df_team_stats = pd.concat([df_home, df_away], ignore_index=True)
    df_team_stats = df_team_stats.sort_values(by=['TEAM_ID', 'GAME_DATE'])

    # B. Define Columns to Roll
    exclude_cols = [
        'TEAM_wins_before', 'TEAM_losses_before', 'TEAM_games_before', 'TEAM_winpct_before',
        'TEAM_conf_rank_before', 'TEAM_league_rank_before', 'TEAM_streak_before', 'WIN', 
        'TEAM_ABBREVIATION', 'TEAM_is_east', 'TEAM_ID', 'GAME_ID', 'TEAM_NAME', 'GAME_DATE'
    ]
    cols_to_roll = [c for c in df_team_stats.select_dtypes(include='number').columns if c not in exclude_cols]

    # C. Calculate Rolling Stats
    # CRITICAL: We use .shift(1) to ensure we only use PAST games.
    # Without shift(1), the rolling mean would include the CURRENT game's stats, causing leakage.
    df_rolling = df_team_stats.groupby('TEAM_ID')[cols_to_roll].apply(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    
    df_rolling = df_rolling.add_suffix('_last5')
    
    # D. Merge Back to Main DataFrame
    df_stats_calculated = pd.concat([df_team_stats[['GAME_ID', 'TEAM_ID']], df_rolling], axis=1)
    
    df_final = df_game_info.copy()
    
    # Merge Home Stats
    df_final = df_final.merge(df_stats_calculated, left_on=['HOME_TEAM_ID', 'GAME_ID'], right_on=['TEAM_ID', 'GAME_ID'], how='left').drop(columns=['TEAM_ID'])
    df_final = df_final.rename(columns={c: f"HOME_{c}" for c in df_rolling.columns})
    
    # Merge Away Stats
    df_final = df_final.merge(df_stats_calculated, left_on=['AWAY_TEAM_ID', 'GAME_ID'], right_on=['TEAM_ID', 'GAME_ID'], how='left').drop(columns=['TEAM_ID'])
    df_final = df_final.rename(columns={c: f"AWAY_{c}" for c in df_rolling.columns})

    return df_final

# ============================================================================
# 4. FINAL PREPARATION & TRAIN-TEST SPLIT
# ============================================================================
def prepare_datasets(df_final):
    """
    Creates contextual features (scarcity, timing), selects final feature subset,
    encodes variables, and performs a TEMPORAL Train/Test split.
    """
    
    print("Preparing training and test datasets...")
    
    # 1. Contextual Features
    # How many other games are on? (Scarcity drives value)
    df_final['NB_GAMES_THIS_DAY'] = df_final.groupby('GAME_DATE')['GAME_ID'].transform('count')
    # How closely matched are the teams?
    df_final['level_difference'] = df_final['HOME_TEAM_winpct_before'] - df_final['AWAY_TEAM_winpct_before']
    # Is it Inter-Conference?
    df_final['conference_difference'] = np.abs(df_final["HOME_TEAM_is_east"] - df_final["AWAY_TEAM_is_east"])
    
    # Clean & Sort
    df_clean = df_final.dropna().copy()
    df_ml = df_clean.sort_values(by='GAME_DATE').reset_index(drop=True)
    
    # 2. Date Features
    df_ml['month'] = df_ml['GAME_DATE'].dt.month
    df_ml['day_of_week'] = df_ml['GAME_DATE'].dt.dayofweek
    df_ml['is_weekend'] = df_ml['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 3. Select Final Feature Set
    target = 'media_value_per_game'
    cols_to_keep = [
        'HOME_view_per_game_last5', 'AWAY_view_per_game_last5', # Momentum/Popualrity
        'is_weekend', 'NB_GAMES_THIS_DAY',                      # Context
        'HOME_TEAM_winpct_before', 'AWAY_TEAM_winpct_before',   # Team Quality
        'HOME_PLUS_MINUS_last5', 'AWAY_PLUS_MINUS_last5',       # Performance
        'AWAY_TEAM_losses_before', 'HOME_TEAM_losses_before',
        'HOME_TEAM_streak_before', 'AWAY_TEAM_streak_before',
        'AWAY_PTS_last5', 'HOME_PTS_last5',                     # Scoring ability
        "HOME_TEAM_ID"                                          # ID for embeddings/specific team bias
    ]
    
    # Validate features exist
    features = [c for c in cols_to_keep if c in df_ml.columns]
    
    X = df_ml[features].copy()
    y = df_ml[target]
    
    # 4. One-Hot Encoding (for Categoricals like Team ID)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        features = X.columns.tolist()
        
    # 5. Temporal Split (No Random Shuffle!)
    # We split by time (first 80% dates vs last 20% dates) to simulate real-world forecasting.
    split_index = int(len(df_ml) * 0.80)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Keep metadata for analysis later
    meta_test = df_ml[['GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']].iloc[split_index:]
    
    # 6. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, meta_test, features

# ============================================================================
# 5. MODELING & EVALUATION
# ============================================================================
def run_models(X_train, y_train, X_test, y_test):
    """
    Trains baseline (Linear) and advanced (XGBoost, RF) models.
    Logs performance metrics (MAE, RMSE, R2) to a text file.
    """
    print("Training models...")
    
    results = {}
    metrics_log = []
    metrics_log.append(f"Mean Target Value: ${int(y_test.mean()):,}\n")

    def evaluate(model, name):
        """Helper to fit, predict, and calculate metrics."""
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        log_str = (f"--- Results for: {name} ---\n"
                   f"MAE  : ${mae:,.2f}\n"
                   f"RMSE : ${rmse:,.2f}\n"
                   f"R²   : {r2:.4f}\n"
                   f"{'-'*30}\n")
        print(log_str)
        metrics_log.append(log_str)
        return preds

    # 1. Linear Regression (Baseline)
    lr = LinearRegression()
    results['Linear'] = {'model': lr, 'preds': evaluate(lr, "Linear Regression"), 'color': 'blue'}

    # 2. XGBoost (Gradient Boosting)
    xgb = XGBRegressor(n_estimators=1000, random_state=24, n_jobs=-1)
    results['XGBoost'] = {'model': xgb, 'preds': evaluate(xgb, "XGBoost"), 'color': 'cyan'}

    # 3. Random Forest (Bagging)
    rf = RandomForestRegressor(n_estimators=1000, random_state=24, n_jobs=-1)
    results['RandomForest'] = {'model': rf, 'preds': evaluate(rf, "Random Forest"), 'color': 'orange'}
    
    # Save metrics report
    with open(f"{OUTPUT_DIR}/model_metrics.txt", "w") as f:
        f.writelines(metrics_log)
    print(f"Metrics saved to {OUTPUT_DIR}/model_metrics.txt")

    return results

# ============================================================================
# 6. VISUALIZATION
# ============================================================================
def save_performance_plots(results, meta_test, y_test):
    """
    Generates a 4-panel performance analysis plot for each model:
    1. Timeline (Forecast vs Actual over time)
    2. Scatter (Correlation)
    3. % Error vs Value (Does model fail on high-value games?)
    4. Residual Histogram (Error distribution)
    """

    print("Generating performance plots...")
    df_viz = meta_test.copy()
    df_viz['Actual'] = y_test
    
    model_names = ['Linear', 'XGBoost', 'RandomForest']
    display_names = ['Linear Regression', 'XGBoost', 'Random Forest']
    
    # Create a grid: 3 models x 4 plot types
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    
    for i, name in enumerate(model_names):
        preds = results[name]['preds']
        color = results[name]['color']
        disp_name = display_names[i]
        
        errors = df_viz['Actual'] - preds
        abs_errors = np.abs(errors)
        
        # Plot 1: Timeline
        sns.lineplot(data=df_viz, x='GAME_DATE', y='Actual', label='Actual', alpha=0.5, color='black', ax=axes[i, 0], errorbar=None)
        sns.lineplot(x=df_viz['GAME_DATE'], y=preds, label='Pred', alpha=0.8, color=color, ax=axes[i, 0],errorbar=None)
        axes[i, 0].set_title(f'{disp_name}: Timeline')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter (Predicted vs Actual)
        sns.scatterplot(x=df_viz['Actual'], y=preds, alpha=0.5, color=color, ax=axes[i, 1])
        # Add diagonal perfect-fit line
        axes[i, 1].plot([df_viz['Actual'].min(), df_viz['Actual'].max()], 
                        [df_viz['Actual'].min(), df_viz['Actual'].max()], 'r--')
        axes[i, 1].set_title(f'{disp_name}: Actual vs Pred')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Plot 3: Percentage Error vs Actual Value
        # Helps identify if the model struggles with outliers (Super Bowl equivalents)
        pct_errors = np.where(df_viz['Actual'] != 0, (abs_errors / df_viz['Actual']) * 100, 0)
        sorted_idx = np.argsort(df_viz['Actual'])
        sorted_act = df_viz['Actual'].iloc[sorted_idx]
        sorted_pct = pct_errors[sorted_idx]
        
        sns.scatterplot(x=df_viz['Actual'], y=pct_errors, alpha=0.4, color=color, ax=axes[i, 2])
        # Add Rolling Median Error Line
        win = max(10, len(sorted_act)//20)
        roll_med = pd.Series(sorted_pct).rolling(win, center=True).median()
        axes[i, 2].plot(sorted_act, roll_med, color='red')
        axes[i, 2].set_ylim(0, 200)
        axes[i, 2].set_title(f'{disp_name}: % Error')
        axes[i, 2].grid(True, alpha=0.3)
        
        # Plot 4: Error Distribution (Residuals)
        sns.histplot(errors, kde=True, color=color, ax=axes[i, 3], bins=30)
        axes[i, 3].axvline(0, color='red', linestyle='--')
        axes[i, 3].set_title(f'{disp_name}: Error Dist')
        axes[i, 3].grid(True, alpha=0.3)

    plt.suptitle("Model Performance Analysis", fontsize=20)
    plt.savefig(f"{OUTPUT_DIR}/performance_analysis.png")
    print(f"Performance plot saved to {OUTPUT_DIR}/performance_analysis.png")
    plt.close()

def save_feature_importance_plots(results, X_train, X_test, features):
    """
    Generates two importance plots per model:
    1. Native Feature Importance (Coefficients or Gini Importance)
    2. SHAP Summary Plot (Model explainability)
    """
    
    print("Generating feature importance plots...")
    shap.initjs()
    
    configs = [
        {'id': 'Linear', 'name': 'Linear Regression', 'type': 'linear', 'palette': 'vlag'},
        {'id': 'XGBoost', 'name': 'XGBoost', 'type': 'tree', 'palette': 'plasma'},
        {'id': 'RandomForest', 'name': 'Random Forest', 'type': 'tree', 'palette': 'viridis'}
    ]
    
    for config in configs:
        mid = config['id']
        model = results[mid]['model']
        
        fig, axes = plt.subplots(1, 2, figsize=(40, 12), constrained_layout=True)
        
        # Left Panel: Native Importance
        if config['type'] == 'linear':
            vals = model.coef_
        else:
            vals = model.feature_importances_
            
        df_imp = pd.DataFrame({'Feature': features, 'Val': vals}).sort_values(by='Val', ascending=False).head(15)
        sns.barplot(data=df_imp, x='Val', y='Feature', ax=axes[0], palette=config['palette'], edgecolor='black')
        axes[0].set_title(f"{config['name']}: Top 15 Features")
        axes[0].grid(True, axis='x', alpha=0.3)
        
        # Right Panel: SHAP Analysis
        try:
            plt.sca(axes[1])
            if config['type'] == 'linear':
                # Linear Explainer for Regression
                explainer = shap.LinearExplainer(model, X_train, feature_names=features)
                shap_vals = explainer.shap_values(X_test)
            else:
                # Tree Explainer for XGB/RF
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_test)
            
            shap.summary_plot(shap_vals, X_test, feature_names=features, show=False, plot_size=None)
            axes[1].set_title(f"SHAP Analysis: {config['name']}")
        except Exception as e:
            axes[1].text(0.5, 0.5, f"SHAP Error: {e}", ha='center', va='center')
            
        plt.savefig(f"{OUTPUT_DIR}/importance_{mid}.png")
        print(f"Importance plot saved to {OUTPUT_DIR}/importance_{mid}.png")
        plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # 1. Load Data
    df_raw = load_and_merge_data()
    
    # 2. Feature Engineering 1: Pre-match State
    df_pre = calculate_pre_match_stats(df_raw)
    
    # 3. Feature Engineering 2: Rolling Averages
    df_full = calculate_rolling_averages(df_pre)
    
    # 4. Final Prep & Split
    X_tr, y_tr, X_te, y_te, meta_te, feats = prepare_datasets(df_full)
    
    # 5. Training & Inference
    model_results = run_models(X_tr, y_tr, X_te, y_te)
    
    # 6. Save Visuals
    save_performance_plots(model_results, meta_te, y_te)
    save_feature_importance_plots(model_results, X_tr, X_te, feats)
    
    print("\nProcessing complete. All artifacts saved to 'model_outputs/' directory.")