import pandas as pd
import numpy as np

def process_cricket_stats():
    """
    Process cricket batting and bowling data to create comprehensive player statistics
    """
    
    # Read the CSV files
    print("Reading batting data...")
    batting_df = pd.read_csv('data/all_season_batting_card.csv')
    
    print("Reading bowling data...")
    bowling_df = pd.read_csv('data/all_season_bowling_card.csv')
    
    # Process batting statistics
    print("Processing batting statistics...")
    # Convert numeric columns to proper numeric types, handling any string issues
    batting_df['runs'] = pd.to_numeric(batting_df['runs'], errors='coerce').fillna(0)
    batting_df['ballsFaced'] = pd.to_numeric(batting_df['ballsFaced'], errors='coerce').fillna(0)
    batting_df['isNotOut'] = batting_df['isNotOut'].astype(bool)
    
    batting_stats = batting_df.groupby('fullName').agg({
        'runs': 'sum',
        'ballsFaced': 'sum',
        'isNotOut': 'sum'
    }).reset_index()
    
    # Calculate proper batting statistics
    batting_stats['total_runs'] = batting_stats['runs']
    batting_stats['total_balls_faced'] = batting_stats['ballsFaced']
    batting_stats['times_not_out'] = batting_stats['isNotOut']
    
    # Calculate innings played (count of non-null entries per player)
    innings_played = batting_df.groupby('fullName').size().reset_index(name='innings_played')
    batting_stats = batting_stats.merge(innings_played, on='fullName', how='left')
    
    # Calculate batting average (runs / (innings - not_outs))
    batting_stats['times_out'] = batting_stats['innings_played'] - batting_stats['times_not_out']
    batting_stats['batting_average'] = np.where(
        batting_stats['times_out'] > 0,
        batting_stats['total_runs'] / batting_stats['times_out'],
        np.inf  # If never out, batting average is infinite
    )
    
    # Calculate batting strike rate (runs per 100 balls)
    batting_stats['batting_strike_rate'] = np.where(
        batting_stats['total_balls_faced'] > 0,
        (batting_stats['total_runs'] / batting_stats['total_balls_faced']) * 100,
        0
    )
    
    # Process bowling statistics
    print("Processing bowling statistics...")
    # Convert numeric columns to proper numeric types, handling any string issues
    bowling_df['wickets'] = pd.to_numeric(bowling_df['wickets'], errors='coerce').fillna(0)
    bowling_df['conceded'] = pd.to_numeric(bowling_df['conceded'], errors='coerce').fillna(0)
    bowling_df['overs'] = pd.to_numeric(bowling_df['overs'], errors='coerce').fillna(0)
    
    bowling_stats = bowling_df.groupby('fullName').agg({
        'wickets': 'sum',
        'conceded': 'sum',
        'overs': 'sum'
    }).reset_index()
    
    # Calculate proper bowling statistics
    bowling_stats['total_wickets'] = bowling_stats['wickets']
    bowling_stats['total_runs_conceded'] = bowling_stats['conceded']
    bowling_stats['total_overs'] = bowling_stats['overs']
    
    # Calculate bowling average (runs conceded / wickets taken)
    bowling_stats['bowling_average'] = np.where(
        bowling_stats['total_wickets'] > 0,
        bowling_stats['total_runs_conceded'] / bowling_stats['total_wickets'],
        np.inf  # If no wickets taken, bowling average is infinite
    )
    
    # Calculate bowling strike rate (balls bowled / wickets taken)
    # Convert overs to balls (1 over = 6 balls)
    bowling_stats['total_balls_bowled'] = bowling_stats['total_overs'] * 6
    bowling_stats['bowling_strike_rate'] = np.where(
        bowling_stats['total_wickets'] > 0,
        bowling_stats['total_balls_bowled'] / bowling_stats['total_wickets'],
        np.inf  # If no wickets taken, bowling strike rate is infinite
    )
    
    # Get all unique players from both datasets
    all_players = set(batting_stats['fullName'].tolist() + bowling_stats['fullName'].tolist())
    
    # Create final comprehensive statistics
    print("Creating comprehensive player statistics...")
    final_stats = []
    
    for player in all_players:
        player_data = {'player_name': player}
        
        # Get batting stats
        batting_row = batting_stats[batting_stats['fullName'] == player]
        if not batting_row.empty:
            player_data['total_runs'] = int(batting_row['total_runs'].iloc[0])
            player_data['batting_average'] = round(batting_row['batting_average'].iloc[0], 2) if batting_row['batting_average'].iloc[0] != np.inf else 'N/A'
            player_data['batting_strike_rate'] = round(batting_row['batting_strike_rate'].iloc[0], 2)
        else:
            player_data['total_runs'] = 0
            player_data['batting_average'] = 'N/A'
            player_data['batting_strike_rate'] = 'N/A'
        
        # Get bowling stats
        bowling_row = bowling_stats[bowling_stats['fullName'] == player]
        if not bowling_row.empty:
            player_data['total_wickets'] = int(bowling_row['total_wickets'].iloc[0])
            player_data['bowling_average'] = round(bowling_row['bowling_average'].iloc[0], 2) if bowling_row['bowling_average'].iloc[0] != np.inf else 'N/A'
            player_data['bowling_strike_rate'] = round(bowling_row['bowling_strike_rate'].iloc[0], 2) if bowling_row['bowling_strike_rate'].iloc[0] != np.inf else 'N/A'
        else:
            player_data['total_wickets'] = 0
            player_data['bowling_average'] = 'N/A'
            player_data['bowling_strike_rate'] = 'N/A'
        
        final_stats.append(player_data)
    
    # Create DataFrame and sort by total runs (descending)
    final_df = pd.DataFrame(final_stats)
    final_df = final_df.sort_values('total_runs', ascending=False)
    
    # Reorder columns as requested
    column_order = [
        'player_name',
        'total_runs', 
        'total_wickets',
        'batting_average',
        'bowling_average',
        'batting_strike_rate',
        'bowling_strike_rate'
    ]
    final_df = final_df[column_order]
    
    # Save to CSV
    output_file = 'comprehensive_player_stats.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\nComprehensive player statistics saved to: {output_file}")
    print(f"Total players processed: {len(final_df)}")
    print("\nFirst 10 rows:")
    print(final_df.head(10).to_string(index=False))
    
    # Print some summary statistics
    print(f"\nSummary Statistics:")
    print(f"Players with batting data: {len(batting_stats)}")
    print(f"Players with bowling data: {len(bowling_stats)}")
    print(f"Total unique players: {len(final_df)}")
    
    return final_df

if __name__ == "__main__":
    try:
        df = process_cricket_stats()
        print("\nScript completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()