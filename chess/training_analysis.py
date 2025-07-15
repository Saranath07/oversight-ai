#!/usr/bin/env python3
"""
Training Data Analysis for Chess Model
Provides detailed statistics about the training dataset
"""

import json
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_training_data():
    """Analyze the chess training data in detail."""
    
    training_data_path = "processed_data/chess_training_sample.json"
    
    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found at {training_data_path}")
        return {}
    
    print("📊 Loading and analyzing training data...")
    
    try:
        with open(training_data_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} training samples")
        
        # Initialize statistics
        stats = {
            'total_samples': len(data),
            'total_moves': 0,
            'move_distribution': defaultdict(int),
            'piece_activity': defaultdict(int),
            'special_moves': defaultdict(int),
            'game_length_distribution': [],
            'opening_moves': defaultdict(int),
            'sample_games': []
        }
        
        # Analyze each sample
        for i, sample in enumerate(data):
            if i < 10:  # Store first 10 games as samples
                stats['sample_games'].append(sample)
            
            if isinstance(sample, dict):
                # Handle different data formats
                moves = []
                if 'moves' in sample:
                    moves = sample['moves']
                elif 'game' in sample:
                    moves = sample['game'].split() if isinstance(sample['game'], str) else sample['game']
                elif isinstance(sample, list):
                    moves = sample
                
                if moves:
                    stats['total_moves'] += len(moves)
                    stats['game_length_distribution'].append(len(moves))
                    
                    # Analyze first few moves (opening)
                    for j, move in enumerate(moves[:10]):
                        if isinstance(move, str):
                            stats['opening_moves'][f"move_{j+1}_{move}"] += 1
                    
                    # Analyze all moves
                    for move in moves:
                        if isinstance(move, str) and move:
                            # Basic move analysis
                            if len(move) >= 2:
                                # Piece identification
                                if move[0].isupper():
                                    piece = move[0]
                                else:
                                    piece = 'P'  # Pawn
                                stats['piece_activity'][piece] += 1
                            
                            # Special move analysis
                            if 'x' in move:
                                stats['special_moves']['captures'] += 1
                            if '+' in move:
                                stats['special_moves']['checks'] += 1
                            if '#' in move:
                                stats['special_moves']['checkmates'] += 1
                            if 'O-O' in move:
                                stats['special_moves']['castling'] += 1
                            if '=' in move:
                                stats['special_moves']['promotions'] += 1
            
            elif isinstance(sample, list):
                # Handle list format
                stats['total_moves'] += len(sample)
                stats['game_length_distribution'].append(len(sample))
                
                for move in sample:
                    if isinstance(move, str) and move:
                        if 'x' in move:
                            stats['special_moves']['captures'] += 1
                        if '+' in move:
                            stats['special_moves']['checks'] += 1
        
        # Convert defaultdicts to regular dicts
        stats['move_distribution'] = dict(stats['move_distribution'])
        stats['piece_activity'] = dict(stats['piece_activity'])
        stats['special_moves'] = dict(stats['special_moves'])
        stats['opening_moves'] = dict(stats['opening_moves'])
        
        # Calculate additional statistics
        if stats['game_length_distribution']:
            stats['avg_game_length'] = sum(stats['game_length_distribution']) / len(stats['game_length_distribution'])
            stats['min_game_length'] = min(stats['game_length_distribution'])
            stats['max_game_length'] = max(stats['game_length_distribution'])
        
        return stats
        
    except Exception as e:
        print(f"❌ Error analyzing training data: {e}")
        return {}

def print_training_summary(stats):
    """Print a comprehensive summary of training data."""
    
    print("\n" + "="*60)
    print("📈 CHESS TRAINING DATA ANALYSIS")
    print("="*60)
    
    print(f"📊 Dataset Overview:")
    print(f"   Total Samples: {stats['total_samples']:,}")
    print(f"   Total Moves: {stats['total_moves']:,}")
    print(f"   Average Game Length: {stats.get('avg_game_length', 0):.1f} moves")
    print(f"   Min Game Length: {stats.get('min_game_length', 0)} moves")
    print(f"   Max Game Length: {stats.get('max_game_length', 0)} moves")
    
    if stats['piece_activity']:
        print(f"\n🎯 Piece Activity:")
        sorted_pieces = sorted(stats['piece_activity'].items(), key=lambda x: x[1], reverse=True)
        for piece, count in sorted_pieces:
            piece_name = {'P': 'Pawn', 'R': 'Rook', 'N': 'Knight', 'B': 'Bishop', 'Q': 'Queen', 'K': 'King'}.get(piece, piece)
            percentage = (count / stats['total_moves']) * 100 if stats['total_moves'] > 0 else 0
            print(f"   {piece_name}: {count:,} ({percentage:.1f}%)")
    
    if stats['special_moves']:
        print(f"\n⚡ Special Moves:")
        for move_type, count in stats['special_moves'].items():
            percentage = (count / stats['total_moves']) * 100 if stats['total_moves'] > 0 else 0
            print(f"   {move_type.title()}: {count:,} ({percentage:.1f}%)")
    
    if stats['sample_games']:
        print(f"\n🎮 Sample Games (first 3):")
        for i, game in enumerate(stats['sample_games'][:3]):
            print(f"   Game {i+1}: {str(game)[:100]}...")
    
    print("\n" + "="*60)

def main():
    """Main function for training data analysis."""
    print("🏁 Chess Training Data Analysis")
    print("=" * 50)
    
    stats = analyze_training_data()
    
    if stats:
        print_training_summary(stats)
        
        # Save detailed statistics
        with open("training_data_analysis.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"📄 Detailed analysis saved to: training_data_analysis.json")
    else:
        print("❌ Could not analyze training data")

if __name__ == "__main__":
    main()