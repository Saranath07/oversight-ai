"""
Debug script for bowler data analysis
"""
import os
import sys
import json
import pickle
import numpy as np
import traceback

def main():
    """Main debug function with error handling"""
    try:
        data_dir = "data_generation/processed"
        print(f"Loading data from {data_dir}")
        
        # Check if files exist
        vocab_path = os.path.join(data_dir, "vocabulary.json")
        if not os.path.exists(vocab_path):
            print(f"ERROR: Vocabulary file not found at {vocab_path}")
            # Try alternate path
            alt_path = "../data_generation/processed/vocabulary.json"
            if os.path.exists(alt_path):
                print(f"Found vocab at alternate path: {alt_path}")
                vocab_path = alt_path
            else:
                print(f"Alternate path not found either: {alt_path}")
        
        # Load vocabulary with error handling
        try:
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            print(f"Successfully loaded vocabulary with {len(vocab)} tokens")
        except Exception as e:
            print(f"ERROR loading vocabulary: {e}")
            return
        
        # Try to load data
        try:
            # Try to load history data
            histories_path = os.path.join(data_dir, "match_histories.pkl")
            if not os.path.exists(histories_path):
                print(f"ERROR: Histories file not found at {histories_path}")
                # Try alternate path
                alt_path = "../data_generation/processed/match_histories.pkl"
                if os.path.exists(alt_path):
                    print(f"Found histories at alternate path: {alt_path}")
                    histories_path = alt_path
                else:
                    print(f"Alternate path not found either: {alt_path}")
                    # List contents of directory
                    dir_path = os.path.dirname(histories_path)
                    if os.path.exists(dir_path):
                        print(f"Contents of {dir_path}:")
                        for item in os.listdir(dir_path):
                            print(f"  {item}")
            
            # Try to load 
            with open(histories_path, 'rb') as f:
                histories = pickle.load(f)
            print(f"Successfully loaded {len(histories)} history sequences")
            
            # Print sample history shape
            if histories:
                if isinstance(histories[0], list):
                    print(f"First history has {len(histories[0])} balls")
                    if histories[0]:
                        print(f"First ball vector shape: {len(histories[0][0])}")
                        print(f"Sample ball vector: {histories[0][0]}")
                else:
                    print(f"Unexpected history format: {type(histories[0])}")
            
            # Try to load context data
            contexts_path = os.path.join(data_dir, "contexts.pkl")
            if not os.path.exists(contexts_path):
                print(f"ERROR: Contexts file not found at {contexts_path}")
                alt_path = "../data_generation/processed/contexts.pkl"
                if os.path.exists(alt_path):
                    print(f"Found contexts at alternate path: {alt_path}")
                    contexts_path = alt_path
            
            with open(contexts_path, 'rb') as f:
                contexts = pickle.load(f)
            print(f"Successfully loaded {len(contexts)} context vectors")
            
            # Print sample context shape
            if contexts:
                print(f"First context shape: {len(contexts[0])}")
                print(f"Sample context: {contexts[0]}")
            
            # Try to load target data
            targets_path = os.path.join(data_dir, "targets.pkl")
            if not os.path.exists(targets_path):
                print(f"ERROR: Targets file not found at {targets_path}")
                alt_path = "../data_generation/processed/targets.pkl"
                if os.path.exists(alt_path):
                    print(f"Found targets at alternate path: {alt_path}")
                    targets_path = alt_path
            
            with open(targets_path, 'rb') as f:
                targets = pickle.load(f)
            print(f"Successfully loaded {len(targets)} target sequences")
            
            # Print sample target shape
            if targets:
                print(f"First target shape: {len(targets[0])}")
                print(f"Sample target: {targets[0]}")
            
            # Check for bowlers
            if len(histories) > 0:
                print("\nSearching for bowlers in the dataset...")
                # Create reverse mapping for tokens
                idx_to_token = {v: k for k, v in vocab.items()}
                
                # Extract bowler information from ball vectors
                bowler_stats = {}
                count = 0
                
                # Look at first 1000 history sequences max
                for hist_idx, history in enumerate(histories[:1000]):
                    if hist_idx % 100 == 0:
                        print(f"Checking history {hist_idx}...")
                    
                    for ball_idx, ball_vector in enumerate(history):
                        # Check if there are enough elements in the ball vector
                        if len(ball_vector) < 19:  # We need at least 19 for bowler stats
                            print(f"WARNING: Ball vector too short ({len(ball_vector)}) at history {hist_idx}, ball {ball_idx}")
                            continue
                        
                        # Bowler stats are at indices 16, 17, 18
                        try:
                            bowling_avg = float(ball_vector[16])
                            bowling_sr = float(ball_vector[17])
                            total_wickets = float(ball_vector[18]) * 10  # De-normalize
                            
                            # Create a simple bowler identifier based on stats
                            bowler_key = f"bowler_{bowling_avg:.1f}_{bowling_sr:.1f}_{int(total_wickets)}"
                            
                            if bowler_key not in bowler_stats:
                                bowler_stats[bowler_key] = {
                                    'bowling_average': bowling_avg,
                                    'bowling_strike_rate': bowling_sr,
                                    'total_wickets': total_wickets,
                                    'count': 0
                                }
                            
                            bowler_stats[bowler_key]['count'] += 1
                            count += 1
                            
                            # Only need a few examples per history
                            if ball_idx > 5:
                                break
                        except Exception as e:
                            print(f"Error processing ball {ball_idx} in history {hist_idx}: {e}")
                
                print(f"\nFound {len(bowler_stats)} unique bowlers in {count} balls")
                
                # Print top bowlers by frequency
                top_bowlers = sorted(bowler_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
                print("\nTop 10 most frequent bowlers:")
                for i, (bowler_key, stats) in enumerate(top_bowlers):
                    print(f"{i+1}. {bowler_key}: Avg={stats['bowling_average']:.1f}, "
                          f"SR={stats['bowling_strike_rate']:.1f}, "
                          f"Wickets={int(stats['total_wickets'])}, "
                          f"Count={stats['count']}")
                
                # Now check if we can find a few specific matches for certain bowlers
                print("\nSearching for specific bowlers...")
                target_bowlers = [
                    ('bowler_26.9_22.8_150', 26.9, 22.8, 150),
                    ('bowler_20.8_18.7_139', 20.8, 18.7, 139),
                    ('bowler_12.1_11.2_22', 12.1, 11.2, 22),
                ]
                
                tolerance = 0.1  # Tolerance for matching stats
                
                for bowler_name, avg, sr, wickets in target_bowlers:
                    matches = 0
                    print(f"\nLooking for {bowler_name}...")
                    
                    for hist_idx, (history, context, target) in enumerate(zip(histories[:1000], contexts[:1000], targets[:1000])):
                        if hist_idx % 100 == 0 and matches == 0:
                            print(f"  Checking history {hist_idx}...")
                        
                        for ball_idx, ball_vector in enumerate(history):
                            if len(ball_vector) < 19:
                                continue
                            
                            # Check if this matches our target bowler
                            ball_avg = float(ball_vector[16])
                            ball_sr = float(ball_vector[17])
                            ball_wickets = float(ball_vector[18]) * 10
                            
                            if (abs(ball_avg - avg) < tolerance and
                                abs(ball_sr - sr) < tolerance and
                                abs(ball_wickets - wickets) < 5):
                                
                                matches += 1
                                
                                # Get over number from context
                                over_number = int(context[1]) if len(context) > 1 else 'unknown'
                                
                                # Get tokens from target
                                target_tokens = []
                                for token_id in target:
                                    if token_id not in [vocab.get('<PAD>', 21), vocab.get('<START>', 22), vocab.get('<END>', 23)]:
                                        token = idx_to_token.get(token_id, f"UNK_{token_id}")
                                        target_tokens.append(token)
                                
                                print(f"  Found match {matches} at history {hist_idx}, over {over_number}")
                                print(f"    Target tokens: {target_tokens}")
                                
                                if matches >= 3:
                                    break
                        
                        if matches >= 3:
                            break
                    
                    if matches == 0:
                        print(f"  No matches found for {bowler_name}")
                
        except Exception as e:
            print(f"ERROR loading data: {e}")
            print(traceback.format_exc())
            return
                
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()