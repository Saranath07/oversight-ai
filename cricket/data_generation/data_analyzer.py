"""
OverSight AI Data Analyzer
==========================

This module provides comprehensive analysis and visualization of the cricket data
to understand patterns and validate the data generation process.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import pickle

class CricketDataAnalyzer:
    """Comprehensive analyzer for cricket data"""
    
    def __init__(self, json_dir: str, player_stats_file: str, processed_dir: str = None):
        self.json_dir = Path(json_dir)
        self.player_stats_file = Path(player_stats_file)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        
        # Load player stats
        self.player_stats = pd.read_csv(self.player_stats_file)
        
        # Initialize analysis containers
        self.match_stats = []
        self.ball_outcomes = []
        self.over_patterns = []
        self.player_performance = defaultdict(list)
        
    def analyze_raw_data(self):
        """Analyze raw JSON match data"""
        
        print("=== ANALYZING RAW MATCH DATA ===")
        
        json_files = list(self.json_dir.glob('*.json'))
        print(f"Found {len(json_files)} match files")
        
        total_matches = 0
        total_balls = 0
        total_runs = 0
        total_wickets = 0
        
        venues = Counter()
        seasons = Counter()
        outcomes = Counter()
        
        for match_file in json_files[:50]:  # Analyze first 50 matches for speed
            try:
                with open(match_file, 'r') as f:
                    match_data = json.load(f)
                
                total_matches += 1
                
                # Match metadata
                info = match_data['info']
                venues[info['venue']] += 1
                seasons[info['season']] += 1
                
                # Outcome analysis
                if 'outcome' in info:
                    if 'winner' in info['outcome']:
                        outcomes['completed'] += 1
                    else:
                        outcomes['no_result'] += 1
                
                # Ball-by-ball analysis
                for innings in match_data['innings']:
                    for over in innings['overs']:
                        for delivery in over['deliveries']:
                            total_balls += 1
                            total_runs += delivery['runs']['total']
                            
                            if 'wickets' in delivery:
                                total_wickets += len(delivery['wickets'])
                            
                            # Collect ball outcome
                            runs = delivery['runs']['batter']
                            extras = delivery.get('extras', {})
                            wicket = 'wickets' in delivery
                            
                            outcome = self._classify_ball_outcome(runs, extras, wicket)
                            self.ball_outcomes.append(outcome)
                            
            except Exception as e:
                print(f"Error processing {match_file}: {e}")
                continue
        
        # Print summary statistics
        print(f"\n--- SUMMARY STATISTICS ---")
        print(f"Total matches analyzed: {total_matches}")
        print(f"Total balls: {total_balls}")
        print(f"Total runs: {total_runs}")
        print(f"Total wickets: {total_wickets}")
        print(f"Average runs per ball: {total_runs/total_balls:.2f}")
        print(f"Average balls per wicket: {total_balls/total_wickets:.2f}")
        
        print(f"\n--- TOP VENUES ---")
        for venue, count in venues.most_common(10):
            print(f"{venue}: {count} matches")
        
        print(f"\n--- SEASONS ---")
        for season, count in sorted(seasons.items()):
            print(f"{season}: {count} matches")
        
        return {
            'total_matches': total_matches,
            'total_balls': total_balls,
            'total_runs': total_runs,
            'total_wickets': total_wickets,
            'venues': dict(venues),
            'seasons': dict(seasons)
        }
    
    def _classify_ball_outcome(self, runs: int, extras: Dict, wicket: bool) -> str:
        """Classify a ball outcome into categories"""
        
        if wicket:
            return 'W'
        
        if extras:
            if 'wides' in extras:
                return f'wd{extras["wides"]}' if extras['wides'] > 1 else 'wd'
            elif 'noballs' in extras:
                return f'nb{runs}' if runs > 0 else 'nb'
            elif 'legbyes' in extras:
                return 'lb'
            elif 'byes' in extras:
                return 'b'
        
        return str(runs)
    
    def analyze_ball_outcomes(self):
        """Analyze distribution of ball outcomes"""
        
        print("\n=== BALL OUTCOME ANALYSIS ===")
        
        outcome_counts = Counter(self.ball_outcomes)
        total_balls = len(self.ball_outcomes)
        
        print(f"Total balls analyzed: {total_balls}")
        print(f"\n--- OUTCOME DISTRIBUTION ---")
        
        for outcome, count in outcome_counts.most_common():
            percentage = (count / total_balls) * 100
            print(f"{outcome}: {count} ({percentage:.2f}%)")
        
        return outcome_counts
    
    def analyze_over_patterns(self):
        """Analyze patterns within overs"""
        
        print("\n=== OVER PATTERN ANALYSIS ===")
        
        json_files = list(self.json_dir.glob('*.json'))
        over_scores = []
        over_wickets = []
        ball_position_outcomes = defaultdict(list)
        
        for match_file in json_files[:30]:  # Analyze subset for speed
            try:
                with open(match_file, 'r') as f:
                    match_data = json.load(f)
                
                for innings in match_data['innings']:
                    for over in innings['overs']:
                        over_runs = 0
                        over_wicket_count = 0
                        
                        for ball_idx, delivery in enumerate(over['deliveries']):
                            runs = delivery['runs']['total']
                            over_runs += runs
                            
                            if 'wickets' in delivery:
                                over_wicket_count += len(delivery['wickets'])
                            
                            # Track outcomes by ball position in over
                            outcome = self._classify_ball_outcome(
                                delivery['runs']['batter'],
                                delivery.get('extras', {}),
                                'wickets' in delivery
                            )
                            ball_position_outcomes[ball_idx].append(outcome)
                        
                        over_scores.append(over_runs)
                        over_wickets.append(over_wicket_count)
                        
            except Exception as e:
                continue
        
        # Analysis
        print(f"Analyzed {len(over_scores)} overs")
        print(f"Average runs per over: {np.mean(over_scores):.2f}")
        print(f"Median runs per over: {np.median(over_scores):.2f}")
        print(f"Max runs in an over: {max(over_scores)}")
        print(f"Average wickets per over: {np.mean(over_wickets):.3f}")
        
        # Ball position analysis
        print(f"\n--- BALL POSITION ANALYSIS ---")
        for ball_pos in range(6):
            if ball_pos in ball_position_outcomes:
                outcomes = ball_position_outcomes[ball_pos]
                avg_runs = np.mean([self._outcome_to_runs(o) for o in outcomes])
                print(f"Ball {ball_pos + 1}: {len(outcomes)} balls, avg {avg_runs:.2f} runs")
        
        return {
            'over_scores': over_scores,
            'over_wickets': over_wickets,
            'ball_position_outcomes': dict(ball_position_outcomes)
        }
    
    def _outcome_to_runs(self, outcome: str) -> int:
        """Convert outcome string to runs scored"""
        if outcome == 'W':
            return 0
        elif outcome.startswith('wd') or outcome.startswith('nb'):
            return 1  # Simplified
        elif outcome in ['lb', 'b']:
            return 1  # Simplified
        else:
            try:
                return int(outcome)
            except:
                return 0
    
    def analyze_player_stats(self):
        """Analyze player statistics distribution"""
        
        print("\n=== PLAYER STATISTICS ANALYSIS ===")
        
        # Basic statistics
        print(f"Total players in database: {len(self.player_stats)}")
        
        # Batting statistics
        batting_stats = self.player_stats[self.player_stats['total_runs'] > 0]
        print(f"Players with batting data: {len(batting_stats)}")
        print(f"Average batting average: {batting_stats['batting_average'].mean():.2f}")
        print(f"Average strike rate: {batting_stats['batting_strike_rate'].mean():.2f}")
        
        # Bowling statistics
        bowling_stats = self.player_stats[self.player_stats['total_wickets'] > 0]
        print(f"Players with bowling data: {len(bowling_stats)}")
        
        # Top performers
        print(f"\n--- TOP BATSMEN (by runs) ---")
        top_batsmen = batting_stats.nlargest(10, 'total_runs')
        for _, player in top_batsmen.iterrows():
            print(f"{player['player_name']}: {player['total_runs']} runs, "
                  f"avg {player['batting_average']:.1f}, SR {player['batting_strike_rate']:.1f}")
        
        print(f"\n--- TOP BOWLERS (by wickets) ---")
        top_bowlers = bowling_stats.nlargest(10, 'total_wickets')
        for _, player in top_bowlers.iterrows():
            print(f"{player['player_name']}: {player['total_wickets']} wickets, "
                  f"avg {player['bowling_average']:.1f}")
        
        return {
            'total_players': len(self.player_stats),
            'batsmen_count': len(batting_stats),
            'bowlers_count': len(bowling_stats)
        }
    
    def analyze_processed_data(self):
        """Analyze processed training data"""
        
        if not self.processed_dir or not self.processed_dir.exists():
            print("Processed data directory not found. Run data_processor.py first.")
            return
        
        print("\n=== PROCESSED DATA ANALYSIS ===")
        
        # Load metadata
        with open(self.processed_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"Number of training sequences: {metadata['num_sequences']}")
        print(f"Ball vector dimension: {metadata['ball_vector_dim']}")
        print(f"Context vector dimension: {metadata['context_vector_dim']}")
        print(f"Vocabulary size: {metadata['vocab_size']}")
        print(f"Max sequence length: {metadata['max_sequence_length']}")
        
        # Load vocabulary
        with open(self.processed_dir / 'vocabulary.json', 'r') as f:
            vocabulary = json.load(f)
        
        print(f"\nVocabulary: {list(vocabulary.keys())}")
        
        # Load and analyze targets
        with open(self.processed_dir / 'targets.pkl', 'rb') as f:
            targets = pickle.load(f)
        
        # Analyze target sequences
        target_lengths = [len(seq) for seq in targets]
        all_tokens = [token for seq in targets for token in seq]
        token_counts = Counter(all_tokens)
        
        print(f"\n--- TARGET SEQUENCE ANALYSIS ---")
        print(f"Average target length: {np.mean(target_lengths):.2f}")
        print(f"Min target length: {min(target_lengths)}")
        print(f"Max target length: {max(target_lengths)}")
        
        print(f"\n--- TOKEN DISTRIBUTION ---")
        for token, count in token_counts.most_common():
            percentage = (count / len(all_tokens)) * 100
            print(f"{token}: {count} ({percentage:.2f}%)")
        
        return {
            'metadata': metadata,
            'target_lengths': target_lengths,
            'token_counts': dict(token_counts)
        }
    
    def create_visualizations(self, save_dir: str = "data_generation/analysis"):
        """Create comprehensive visualizations"""
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Ball outcome distribution
        if self.ball_outcomes:
            outcome_counts = Counter(self.ball_outcomes)
            
            plt.figure(figsize=(12, 6))
            outcomes, counts = zip(*outcome_counts.most_common(15))
            plt.bar(outcomes, counts)
            plt.title('Distribution of Ball Outcomes')
            plt.xlabel('Outcome')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path / 'ball_outcomes.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Player statistics distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Batting average distribution
        batting_data = self.player_stats[self.player_stats['batting_average'] > 0]
        axes[0, 0].hist(batting_data['batting_average'], bins=30, alpha=0.7)
        axes[0, 0].set_title('Batting Average Distribution')
        axes[0, 0].set_xlabel('Batting Average')
        axes[0, 0].set_ylabel('Frequency')
        
        # Strike rate distribution
        sr_data = self.player_stats[self.player_stats['batting_strike_rate'] > 0]
        axes[0, 1].hist(sr_data['batting_strike_rate'], bins=30, alpha=0.7)
        axes[0, 1].set_title('Strike Rate Distribution')
        axes[0, 1].set_xlabel('Strike Rate')
        axes[0, 1].set_ylabel('Frequency')
        
        # Bowling average distribution
        bowling_avg_data = self.player_stats[self.player_stats['bowling_average'] > 0]
        axes[1, 0].hist(bowling_avg_data['bowling_average'], bins=30, alpha=0.7)
        axes[1, 0].set_title('Bowling Average Distribution')
        axes[1, 0].set_xlabel('Bowling Average')
        axes[1, 0].set_ylabel('Frequency')
        
        # Total runs vs wickets scatter
        axes[1, 1].scatter(self.player_stats['total_runs'], 
                          self.player_stats['total_wickets'], alpha=0.6)
        axes[1, 1].set_title('Total Runs vs Total Wickets')
        axes[1, 1].set_xlabel('Total Runs')
        axes[1, 1].set_ylabel('Total Wickets')
        
        plt.tight_layout()
        plt.savefig(save_path / 'player_stats.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {save_path}")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        
        print("=" * 60)
        print("OVERSIGHT AI - COMPREHENSIVE DATA ANALYSIS REPORT")
        print("=" * 60)
        
        # Analyze all components
        raw_stats = self.analyze_raw_data()
        outcome_analysis = self.analyze_ball_outcomes()
        over_analysis = self.analyze_over_patterns()
        player_analysis = self.analyze_player_stats()
        processed_analysis = self.analyze_processed_data()
        
        # Create visualizations
        self.create_visualizations()
        
        # Summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"✓ Raw data: {raw_stats['total_matches']} matches, {raw_stats['total_balls']} balls")
        print(f"✓ Ball outcomes: {len(outcome_analysis)} unique outcome types")
        print(f"✓ Player database: {player_analysis['total_players']} players")
        
        if processed_analysis:
            print(f"✓ Training sequences: {processed_analysis['metadata']['num_sequences']}")
            print(f"✓ Vocabulary size: {processed_analysis['metadata']['vocab_size']}")
        
        print("\nData is ready for transformer training!")
        
        return {
            'raw_stats': raw_stats,
            'outcome_analysis': dict(outcome_analysis),
            'over_analysis': over_analysis,
            'player_analysis': player_analysis,
            'processed_analysis': processed_analysis
        }

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CricketDataAnalyzer(
        json_dir="data/ipl_json",
        player_stats_file="data/comprehensive_player_stats.csv",
        processed_dir="data_generation/processed"
    )
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()