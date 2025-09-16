"""
Test script to debug and analyze specific bowlers in a simplified way
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cricket_transformer import create_cricket_transformer
from bowler_performance_analysis import BowlerProfile

def load_data(data_dir="data_generation/processed"):
    """Load processed data files"""
    print(f"Loading data from {data_dir}")
    
    # Load vocabulary
    with open(os.path.join(data_dir, "vocabulary.json"), 'r') as f:
        vocab = json.load(f)
    
    # Create reverse mapping
    idx_to_token = {v: k for k, v in vocab.items()}
    
    # Get special tokens
    pad_token_id = vocab.get("<PAD>", 21)
    start_token_id = vocab.get("<START>", 22)
    end_token_id = vocab.get("<END>", 23)
    
    # Load data samples
    with open(os.path.join(data_dir, "match_histories.pkl"), 'rb') as f:
        histories = pickle.load(f)
    
    with open(os.path.join(data_dir, "contexts.pkl"), 'rb') as f:
        contexts = pickle.load(f)
    
    with open(os.path.join(data_dir, "targets.pkl"), 'rb') as f:
        targets = pickle.load(f)
    
    print(f"Loaded {len(histories)} data samples")
    return histories, contexts, targets, vocab, idx_to_token, pad_token_id, start_token_id, end_token_id

def extract_bowler_data(histories, contexts, targets, bowler_stats, idx_to_token, special_tokens):
    """Extract bowlers and their data"""
    pad_token_id, start_token_id, end_token_id = special_tokens
    
    # Find bowlers based on bowling stats
    bowler_data = defaultdict(list)
    
    # Define bowler profiles to search for
    target_bowlers = {
        'pace_wicket_taker': {'avg': 26.9, 'sr': 22.8, 'wickets': 150},
        'economy_bowler': {'avg': 20.8, 'sr': 18.7, 'wickets': 139},
        'death_bowler': {'avg': 12.1, 'sr': 11.2, 'wickets': 22},
    }
    
    tolerance = 0.1  # Stat matching tolerance
    
    print("\nSearching for bowlers in data...")
    for bowler_type, stats in target_bowlers.items():
        matches_found = 0
        print(f"\nLooking for {bowler_type}: Avg={stats['avg']}, SR={stats['sr']}, Wickets={stats['wickets']}")
        
        for i, (history, context, target) in enumerate(zip(histories, contexts, targets)):
            if i % 5000 == 0:
                print(f"Processed {i} samples...")
            
            # Check each ball in the history for this bowler
            for ball in history:
                # Bowler stats are at indices 16, 17, 18
                bowling_avg = ball[16] 
                bowling_sr = ball[17]
                bowling_wickets = ball[18] * 10  # De-normalize
                
                # Check if this matches our target bowler
                if (abs(bowling_avg - stats['avg']) < tolerance and
                    abs(bowling_sr - stats['sr']) < tolerance and
                    abs(bowling_wickets - stats['wickets']) < 5):
                    
                    # Get over number from context
                    over_number = int(context[1])  # Context index 1 has over number
                    
                    # Convert target to readable tokens
                    target_sequence = []
                    for token_id in target:
                        if token_id not in [pad_token_id, start_token_id, end_token_id]:
                            token_str = idx_to_token.get(token_id, f"UNK_{token_id}")
                            target_sequence.append(token_str)
                    
                    # Calculate stats for this over
                    runs = calculate_runs(target_sequence)
                    wickets = calculate_wickets(target_sequence)
                    
                    # Store data
                    bowler_data[bowler_type].append({
                        'index': i,
                        'over_number': over_number,
                        'context': context.tolist() if isinstance(context, np.ndarray) else context,
                        'target_sequence': target_sequence,
                        'runs': runs,
                        'wickets': wickets
                    })
                    
                    matches_found += 1
                    # Only need one match per history
                    break
        
        print(f"Found {len(bowler_data[bowler_type])} matches for {bowler_type}")
        
        # Show sample data
        if bowler_data[bowler_type]:
            print("\nSample data:")
            for i, data in enumerate(bowler_data[bowler_type][:3]):
                print(f"\nSample {i+1}:")
                print(f"  Over: {data['over_number']}")
                print(f"  Target sequence: {data['target_sequence']}")
                print(f"  Runs: {data['runs']}")
                print(f"  Wickets: {data['wickets']}")
    
    return bowler_data

def calculate_runs(sequence):
    """Calculate total runs from a token sequence"""
    total_runs = 0
    
    for token in sequence:
        if not isinstance(token, str):
            continue
            
        if token in ['0', '1', '2', '3', '4', '5', '6']:
            total_runs += int(token)
        elif token.startswith('wd'):
            if len(token) > 2 and token[2:].isdigit():
                total_runs += 1 + int(token[2:])
            else:
                total_runs += 1
        elif token.startswith('nb'):
            if len(token) > 2 and token[2:].isdigit():
                total_runs += 1 + int(token[2:])
            else:
                total_runs += 1
        elif token in ['lb', 'b']:  # leg bye and bye
            total_runs += 1  # Usually 1 run for these
    
    return total_runs

def calculate_wickets(sequence):
    """Calculate total wickets from a token sequence"""
    return sum(1 for token in sequence if token == 'W')

def analyze_bowler_scenarios(bowler_data):
    """Analyze bowlers by match scenario"""
    scenarios = {
        "Powerplay": (0, 5),   # Overs 0-5
        "Middle": (6, 14),     # Overs 6-14
        "Death": (15, 19)      # Overs 15-19
    }
    
    scenario_stats = defaultdict(lambda: defaultdict(list))
    
    # Group data by scenario
    for bowler_type, data_points in bowler_data.items():
        print(f"\nAnalyzing {bowler_type}:")
        
        for data in data_points:
            over = data['over_number']
            runs = data['runs']
            wickets = data['wickets']
            
            # Determine scenario
            for scenario_name, (start, end) in scenarios.items():
                if start <= over <= end:
                    scenario_stats[bowler_type][scenario_name].append({
                        'over': over,
                        'runs': runs,
                        'wickets': wickets
                    })
        
        # Calculate stats per scenario
        for scenario, overs in scenario_stats[bowler_type].items():
            avg_runs = np.mean([o['runs'] for o in overs]) if overs else 0
            avg_wickets = np.mean([o['wickets'] for o in overs]) if overs else 0
            economy = avg_runs  # Runs per over = economy
            
            print(f"  {scenario} (Overs: {len(overs)}):")
            print(f"    Average Runs: {avg_runs:.2f}")
            print(f"    Average Wickets: {avg_wickets:.2f}")
            print(f"    Economy Rate: {economy:.2f}")
    
    return scenario_stats

if __name__ == "__main__":
    # Load data
    histories, contexts, targets, vocab, idx_to_token, pad_token_id, start_token_id, end_token_id = load_data()
    
    # Extract and analyze bowler data
    bowler_data = extract_bowler_data(
        histories, contexts, targets, vocab, idx_to_token, 
        (pad_token_id, start_token_id, end_token_id)
    )
    
    # Analyze by scenario
    scenario_stats = analyze_bowler_scenarios(bowler_data)