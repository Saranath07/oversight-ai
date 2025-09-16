"""
Cricket Bowler Performance Analysis
==================================

This script analyzes specific bowlers' performance across different match scenarios
and tests the model's prediction accuracy for these bowlers in various situations.

Features:
- Select random bowlers based on bowling statistics
- Analyze performance in different match phases (powerplay, middle, death overs)
- Test predictions in various match situations (defending/chasing targets)
- Generate comprehensive bowler-specific reports and visualizations
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cricket_transformer import CricketTransformer, create_cricket_transformer

@dataclass
class BowlerProfile:
    """Profile of a bowler with their statistics"""
    name: str
    bowling_average: float
    bowling_strike_rate: float
    total_wickets: int
    economy_rate: float
    match_count: int
    
    def __str__(self):
        return f"{self.name} (Avg: {self.bowling_average:.1f}, SR: {self.bowling_strike_rate:.1f}, Wickets: {self.total_wickets})"

@dataclass
class MatchScenario:
    """Different match scenarios for analysis"""
    name: str
    over_range: Tuple[int, int]  # (start_over, end_over)
    description: str

class BowlerPerformanceAnalyzer:
    """Analyzes cricket bowler performance across different scenarios"""
    
    def __init__(self, 
                 data_dir: str = "../data_generation/processed_ipl",
                 model_path: str = "checkpoints/best_model.pt",
                 k_predictions: int = 20,
                 num_bowlers: int = 8,
                 overs_per_bowler: int = 15):
        """
        Initialize the bowler analyzer
        
        Args:
            data_dir: Directory containing processed data files
            model_path: Path to trained model checkpoint
            k_predictions: Number of predictions per over (k)
            num_bowlers: Number of bowlers to analyze
            overs_per_bowler: Number of overs to analyze per bowler
        """
        self.data_dir = data_dir
        self.model_path = model_path
        self.k_predictions = k_predictions
        self.num_bowlers = num_bowlers
        self.overs_per_bowler = overs_per_bowler
        
        # Load vocabulary and metadata
        self.load_vocabulary()
        self.load_metadata()
        
        # Load model
        self.load_model()
        
        # Define match scenarios
        self.scenarios = [
            MatchScenario("Powerplay", (0, 5), "First 6 overs (0-5)"),
            MatchScenario("Middle Overs", (6, 14), "Middle phase (6-14)"),
            MatchScenario("Death Overs", (15, 19), "Final 5 overs (15-19)"),
        ]
        
        # Results storage
        self.results = {
            'bowler_profiles': [],
            'scenario_analysis': {},
            'bowler_performance': {},
            'prediction_accuracy': {},
            'comparative_stats': {}
        }
    
    def load_vocabulary(self):
        """Load vocabulary for token conversion"""
        vocab_path = os.path.join(self.data_dir, "vocabulary.json")
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        # Create reverse mapping
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        # Extract special tokens
        self.start_token_id = self.vocab.get("<START>", 22)
        self.end_token_id = self.vocab.get("<END>", 23)
        self.pad_token_id = self.vocab.get("<PAD>", 21)
        
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")
    
    def load_metadata(self):
        """Load dataset metadata"""
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Dataset metadata: {self.metadata}")
    
    def load_model(self):
        """Load trained cricket transformer model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model
        self.model = create_cricket_transformer(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        print(f"Loaded model with {self.model.count_parameters():,} parameters")
    
    def load_processed_data(self):
        """Load processed training data"""
        print("Loading processed data...")
        
        with open(os.path.join(self.data_dir, "match_histories.pkl"), 'rb') as f:
            histories = pickle.load(f)
        
        with open(os.path.join(self.data_dir, "contexts.pkl"), 'rb') as f:
            contexts = pickle.load(f)
        
        with open(os.path.join(self.data_dir, "targets.pkl"), 'rb') as f:
            targets = pickle.load(f)
        
        print(f"Loaded {len(histories)} training examples")
        return histories, contexts, targets
    
    def extract_bowler_profiles(self, histories, contexts):
        """Extract unique bowler profiles from the data"""
        print("Extracting bowler profiles...")
        
        bowler_stats = defaultdict(lambda: {
            'bowling_averages': [],
            'bowling_strike_rates': [],
            'total_wickets': [],
            'occurrences': 0
        })
        
        # Extract bowler information from ball vectors
        for history in histories:
            for ball_vector in history:
                # Bowler stats are at indices 16, 17, 18 in the ball vector
                bowling_avg = ball_vector[16]
                bowling_sr = ball_vector[17] 
                total_wickets = ball_vector[18] * 10  # De-normalize
                
                # Create a simple bowler identifier based on stats
                bowler_key = f"bowler_{bowling_avg:.1f}_{bowling_sr:.1f}_{int(total_wickets)}"
                
                bowler_stats[bowler_key]['bowling_averages'].append(bowling_avg)
                bowler_stats[bowler_key]['bowling_strike_rates'].append(bowling_sr)
                bowler_stats[bowler_key]['total_wickets'].append(total_wickets)
                bowler_stats[bowler_key]['occurrences'] += 1
        
        # Create bowler profiles
        bowler_profiles = []
        for bowler_key, stats in bowler_stats.items():
            if stats['occurrences'] >= 10:  # Only bowlers with sufficient data
                profile = BowlerProfile(
                    name=bowler_key,
                    bowling_average=np.mean(stats['bowling_averages']),
                    bowling_strike_rate=np.mean(stats['bowling_strike_rates']),
                    total_wickets=int(np.mean(stats['total_wickets'])),
                    economy_rate=np.mean(stats['bowling_averages']) * 6 / np.mean(stats['bowling_strike_rates']),
                    match_count=stats['occurrences']
                )
                bowler_profiles.append(profile)
        
        # Sort by total wickets and select diverse bowlers
        bowler_profiles.sort(key=lambda x: x.total_wickets, reverse=True)
        
        print(f"Found {len(bowler_profiles)} bowlers with sufficient data")
        return bowler_profiles
    
    def select_diverse_bowlers(self, bowler_profiles):
        """Select diverse bowlers for analysis"""
        print("Selecting diverse bowlers for analysis...")
        
        selected_bowlers = []
        
        # Categories for selection
        categories = {
            'pace_wicket_takers': [],    # High wickets, good average
            'economical_bowlers': [],    # Low economy rate
            'strike_bowlers': [],        # Low strike rate
            'experienced_bowlers': [],   # High match count
        }
        
        # Categorize bowlers
        for bowler in bowler_profiles:
            if bowler.total_wickets > 50 and bowler.bowling_average < 30:
                categories['pace_wicket_takers'].append(bowler)
            if bowler.economy_rate < 7.5:
                categories['economical_bowlers'].append(bowler)
            if bowler.bowling_strike_rate < 20:
                categories['strike_bowlers'].append(bowler)
            if bowler.match_count > 50:
                categories['experienced_bowlers'].append(bowler)
        
        # Select bowlers from each category
        bowlers_per_category = max(1, self.num_bowlers // len(categories))
        
        for category_name, bowlers in categories.items():
            if bowlers:
                # Sort by relevant metric for each category
                if category_name == 'pace_wicket_takers':
                    bowlers.sort(key=lambda x: x.total_wickets, reverse=True)
                elif category_name == 'economical_bowlers':
                    bowlers.sort(key=lambda x: x.economy_rate)
                elif category_name == 'strike_bowlers':
                    bowlers.sort(key=lambda x: x.bowling_strike_rate)
                else:  # experienced_bowlers
                    bowlers.sort(key=lambda x: x.match_count, reverse=True)
                
                # Select top bowlers from this category
                selected_from_category = bowlers[:bowlers_per_category]
                selected_bowlers.extend(selected_from_category)
        
        # Fill remaining slots randomly
        remaining_slots = self.num_bowlers - len(selected_bowlers)
        if remaining_slots > 0:
            remaining_bowlers = [b for b in bowler_profiles if b not in selected_bowlers]
            selected_bowlers.extend(random.sample(remaining_bowlers, 
                                                min(remaining_slots, len(remaining_bowlers))))
        
        self.results['bowler_profiles'] = selected_bowlers[:self.num_bowlers]
        
        print(f"Selected {len(self.results['bowler_profiles'])} bowlers:")
        for i, bowler in enumerate(self.results['bowler_profiles'], 1):
            print(f"  {i}. {bowler}")
        
        return self.results['bowler_profiles']
    
    def find_bowler_specific_data(self, histories, contexts, targets, bowler_profile):
        """Find data points specific to a bowler"""
        bowler_data = []
        
        tolerance = 0.1  # Tolerance for matching bowler stats
        
        for i, (history, context, target) in enumerate(zip(histories, contexts, targets)):
            # Check if any ball in the history matches this bowler
            for ball_vector in history:
                bowling_avg = ball_vector[16]
                bowling_sr = ball_vector[17]
                total_wickets = ball_vector[18] * 10
                
                # Check if this matches our bowler (with tolerance)
                if (abs(bowling_avg - bowler_profile.bowling_average) < tolerance and
                    abs(bowling_sr - bowler_profile.bowling_strike_rate) < tolerance and
                    abs(total_wickets - bowler_profile.total_wickets) < 5):
                    
                    # Extract over number from context
                    over_number = int(context[1])  # Over number is at index 1
                    
                    bowler_data.append({
                        'index': i,
                        'history': history,
                        'context': context,
                        'target': target,
                        'over_number': over_number,
                        'scenario': self.get_scenario_for_over(over_number)
                    })
                    break
        
        return bowler_data
    
    def get_scenario_for_over(self, over_number):
        """Get the scenario for a given over number"""
        for scenario in self.scenarios:
            if scenario.over_range[0] <= over_number <= scenario.over_range[1]:
                return scenario.name
        return "Unknown"
    
    def predict_over_multiple_times(self, history, context, k=20):
        """Predict an over k times using the model"""
        predictions = []
        
        # Convert to batch format and move to device
        history_batch = history.unsqueeze(0).to(self.device)
        context_batch = context.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(k):
                generated = self.model.generate(
                    histories=history_batch,
                    contexts=context_batch,
                    start_token_id=self.start_token_id,
                    end_token_id=self.end_token_id,
                    max_length=15,
                    temperature=1.0,
                    top_k=None,
                    top_p=None
                )
                
                # Extract prediction
                pred_tokens = generated[0, 1:].tolist()
                
                # Stop at end token
                if self.end_token_id in pred_tokens:
                    end_idx = pred_tokens.index(self.end_token_id)
                    pred_tokens = pred_tokens[:end_idx]
                
                # Convert to token strings
                pred_sequence = [self.idx_to_token.get(token_id, f"UNK_{token_id}") 
                               for token_id in pred_tokens]
                
                predictions.append(pred_sequence)
        
        return predictions
    
    def calculate_runs(self, sequence: List[str]) -> int:
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
    
    def calculate_wickets(self, sequence: List[str]) -> int:
        """Calculate total wickets from a token sequence"""
        return sum(1 for token in sequence if token == 'W')
    
    def analyze_bowler_performance(self, bowler_profile, bowler_data):
        """Analyze performance of a specific bowler"""
        print(f"Analyzing performance for {bowler_profile.name}...")
        
        scenario_stats = defaultdict(lambda: {
            'overs': [],
            'runs_accuracy': [],
            'wickets_accuracy': [],
            'runs_conceded': [],
            'wickets_taken': [],
            'economy_rates': []
        })
        
        # Sample data points for analysis
        sampled_data = random.sample(bowler_data, 
                                   min(self.overs_per_bowler, len(bowler_data)))
        
        for data_point in sampled_data:
            history_tensor = torch.tensor(data_point['history'], dtype=torch.float32)
            context_tensor = torch.tensor(data_point['context'], dtype=torch.float32)
            
            # Convert target to token sequence - handle both lists and arrays
            if isinstance(data_point['target'], (list, np.ndarray)):
                actual_sequence = []
                for token_id in data_point['target']:
                    if isinstance(token_id, (int, np.integer)) and token_id not in [self.pad_token_id, self.start_token_id, self.end_token_id]:
                        token_str = self.idx_to_token.get(int(token_id), f"UNK_{int(token_id)}")
                        actual_sequence.append(token_str)
            else:
                actual_sequence = []
            
            # Skip if sequence is empty or too short
            if len(actual_sequence) == 0:
                continue
                
            # Make predictions
            predictions = self.predict_over_multiple_times(
                history_tensor, context_tensor, self.k_predictions
            )
            
            # Calculate actual stats
            actual_runs = self.calculate_runs(actual_sequence)
            actual_wickets = self.calculate_wickets(actual_sequence)
            
            # Extract scenario (need to define before using it in debug)
            scenario = data_point['scenario']
            
            # Debug print for first few sequences
            if scenario in scenario_stats and len(scenario_stats[scenario]['overs']) < 3:
                print(f"    Debug - Scenario: {scenario}, Over: {data_point['over_number']}")
                print(f"    Debug - Actual sequence: {actual_sequence}")
                print(f"    Debug - Actual runs: {actual_runs}, Actual wickets: {actual_wickets}")
            
            # Calculate predicted stats
            pred_runs = [self.calculate_runs(pred) for pred in predictions]
            pred_wickets = [self.calculate_wickets(pred) for pred in predictions]
            
            # Calculate accuracies
            runs_accuracy = (sum(1 for r in pred_runs if r == actual_runs) / len(pred_runs)) * 100
            wickets_accuracy = (sum(1 for w in pred_wickets if w == actual_wickets) / len(pred_wickets)) * 100
            
            # Store scenario-specific stats
            scenario_stats[scenario]['overs'].append(data_point['over_number'])
            scenario_stats[scenario]['runs_accuracy'].append(runs_accuracy)
            scenario_stats[scenario]['wickets_accuracy'].append(wickets_accuracy)
            scenario_stats[scenario]['runs_conceded'].append(actual_runs)
            scenario_stats[scenario]['wickets_taken'].append(actual_wickets)
            # Store economy rate as runs per over for this specific over
            scenario_stats[scenario]['economy_rates'].append(float(actual_runs))  # Ensure it's a float
        
        # Calculate summary statistics
        bowler_summary = {}
        for scenario, stats in scenario_stats.items():
            if stats['overs']:
                bowler_summary[scenario] = {
                    'overs_analyzed': len(stats['overs']),
                    'avg_runs_accuracy': np.mean(stats['runs_accuracy']),
                    'avg_wickets_accuracy': np.mean(stats['wickets_accuracy']),
                    'avg_runs_conceded': np.mean(stats['runs_conceded']),
                    'avg_wickets_taken': np.mean(stats['wickets_taken']),
                    'economy_rate': np.mean(stats['economy_rates']),
                    'std_runs_accuracy': np.std(stats['runs_accuracy']),
                    'std_wickets_accuracy': np.std(stats['wickets_accuracy'])
                }
        
        self.results['bowler_performance'][bowler_profile.name] = {
            'profile': bowler_profile,
            'scenario_stats': dict(scenario_stats),
            'summary': bowler_summary
        }
        
        return bowler_summary
    
    def analyze_all_bowlers(self):
        """Analyze all selected bowlers"""
        print(f"Starting comprehensive bowler analysis...")
        
        # Load data
        histories, contexts, targets = self.load_processed_data()
        
        # Extract and select bowlers
        bowler_profiles = self.extract_bowler_profiles(histories, contexts)
        selected_bowlers = self.select_diverse_bowlers(bowler_profiles)
        
        # Analyze each bowler
        for bowler in selected_bowlers:
            bowler_data = self.find_bowler_specific_data(histories, contexts, targets, bowler)
            
            if len(bowler_data) >= 5:  # Minimum data points required
                self.analyze_bowler_performance(bowler, bowler_data)
                print(f"Completed analysis for {bowler.name} ({len(bowler_data)} data points)")
            else:
                print(f"Insufficient data for {bowler.name} ({len(bowler_data)} data points)")
        
        # Calculate comparative statistics
        self.calculate_comparative_stats()
    
    def calculate_comparative_stats(self):
        """Calculate comparative statistics across all bowlers"""
        print("Calculating comparative statistics...")
        
        scenario_comparisons = defaultdict(lambda: {
            'runs_accuracy': [],
            'wickets_accuracy': [],
            'economy_rates': [],
            'bowler_names': []
        })
        
        overall_stats = {
            'best_runs_predictor': ('', 0),
            'best_wickets_predictor': ('', 0),
            'most_economical': ('', float('inf')),
            'best_powerplay': ('', 0),
            'best_death_over': ('', 0)
        }
        
        for bowler_name, performance in self.results['bowler_performance'].items():
            for scenario, stats in performance['summary'].items():
                scenario_comparisons[scenario]['runs_accuracy'].append(stats['avg_runs_accuracy'])
                scenario_comparisons[scenario]['wickets_accuracy'].append(stats['avg_wickets_accuracy'])
                scenario_comparisons[scenario]['economy_rates'].append(stats['economy_rate'])
                scenario_comparisons[scenario]['bowler_names'].append(bowler_name)
                
                # Track best performers
                if stats['avg_runs_accuracy'] > overall_stats['best_runs_predictor'][1]:
                    overall_stats['best_runs_predictor'] = (bowler_name, stats['avg_runs_accuracy'])
                
                if stats['avg_wickets_accuracy'] > overall_stats['best_wickets_predictor'][1]:
                    overall_stats['best_wickets_predictor'] = (bowler_name, stats['avg_wickets_accuracy'])
                
                if stats['economy_rate'] < overall_stats['most_economical'][1]:
                    overall_stats['most_economical'] = (bowler_name, stats['economy_rate'])
                
                if scenario == 'Powerplay' and stats['avg_runs_accuracy'] > overall_stats['best_powerplay'][1]:
                    overall_stats['best_powerplay'] = (bowler_name, stats['avg_runs_accuracy'])
                
                if scenario == 'Death Overs' and stats['avg_runs_accuracy'] > overall_stats['best_death_over'][1]:
                    overall_stats['best_death_over'] = (bowler_name, stats['avg_runs_accuracy'])
        
        self.results['comparative_stats'] = {
            'scenario_comparisons': dict(scenario_comparisons),
            'overall_stats': overall_stats
        }
    
    def generate_bowler_report(self, output_file: str = "bowler_performance_report.txt"):
        """Generate comprehensive bowler analysis report"""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CRICKET BOWLER PERFORMANCE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary
            f.write("ANALYSIS SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of Bowlers Analyzed: {len(self.results['bowler_performance'])}\n")
            f.write(f"Predictions per Over: {self.k_predictions}\n")
            f.write(f"Scenarios Analyzed: {len(self.scenarios)}\n\n")
            
            # Best performers
            f.write("BEST PERFORMERS\n")
            f.write("-" * 20 + "\n")
            stats = self.results['comparative_stats']['overall_stats']
            
            # Check for valid statistics before writing them
            best_runs = stats['best_runs_predictor']
            best_wickets = stats['best_wickets_predictor']
            most_economical = stats['most_economical']
            best_powerplay = stats['best_powerplay']
            best_death = stats['best_death_over']
            
            f.write(f"Best Runs Predictor: {best_runs[0] if best_runs[1] > 0 else 'N/A'} "
                   f"({best_runs[1]:.1f}% accuracy)\n")
            
            f.write(f"Best Wickets Predictor: {best_wickets[0] if best_wickets[1] > 0 else 'N/A'} "
                   f"({best_wickets[1]:.1f}% accuracy)\n")
            
            if most_economical[1] != float('inf') and not np.isnan(most_economical[1]):
                f.write(f"Most Economical: {most_economical[0]} ({most_economical[1]:.2f} RPO)\n")
            else:
                f.write("Most Economical: N/A\n")
            
            f.write(f"Best Powerplay Bowler: {best_powerplay[0] if best_powerplay[1] > 0 else 'N/A'} "
                   f"({best_powerplay[1]:.1f}% accuracy)\n")
            
            f.write(f"Best Death Over Bowler: {best_death[0] if best_death[1] > 0 else 'N/A'} "
                   f"({best_death[1]:.1f}% accuracy)\n\n")
            
            # Individual bowler analysis
            f.write("="*80 + "\n")
            f.write("INDIVIDUAL BOWLER ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            for bowler_name, performance in self.results['bowler_performance'].items():
                profile = performance['profile']
                f.write(f"BOWLER: {bowler_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Profile: Avg {profile.bowling_average:.1f}, SR {profile.bowling_strike_rate:.1f}, "
                       f"Wickets {profile.total_wickets}, Economy {profile.economy_rate:.2f}\n\n")
                
                if performance['summary']:  # Check if summary exists
                    f.write("Scenario Performance:\n")
                    for scenario, stats in performance['summary'].items():
                        # Skip scenarios with no meaningful data
                        if stats['overs_analyzed'] == 0:
                            continue
                            
                        f.write(f"  {scenario}:\n")
                        f.write(f"    Overs Analyzed: {stats['overs_analyzed']}\n")
                        
                        # Format statistics safely
                        runs_acc = stats.get('avg_runs_accuracy', 0)
                        runs_std = stats.get('std_runs_accuracy', 0)
                        if not np.isnan(runs_acc) and not np.isnan(runs_std):
                            f.write(f"    Runs Accuracy: {runs_acc:.1f}% ± {runs_std:.1f}%\n")
                        else:
                            f.write(f"    Runs Accuracy: N/A\n")
                            
                        wickets_acc = stats.get('avg_wickets_accuracy', 0)
                        wickets_std = stats.get('std_wickets_accuracy', 0)
                        if not np.isnan(wickets_acc) and not np.isnan(wickets_std):
                            f.write(f"    Wickets Accuracy: {wickets_acc:.1f}% ± {wickets_std:.1f}%\n")
                        else:
                            f.write(f"    Wickets Accuracy: N/A\n")
                            
                        avg_runs = stats.get('avg_runs_conceded', 0)
                        if not np.isnan(avg_runs):
                            f.write(f"    Avg Runs Conceded: {avg_runs:.1f}\n")
                        else:
                            f.write(f"    Avg Runs Conceded: N/A\n")
                            
                        econ = stats.get('economy_rate', 0)
                        if not np.isnan(econ) and econ != float('inf'):
                            f.write(f"    Economy Rate: {econ:.2f}\n\n")
                        else:
                            f.write(f"    Economy Rate: N/A\n\n")
                else:
                    f.write("No scenario data available for this bowler.\n\n")
                
                f.write("\n")
        
        print(f"Bowler analysis report saved to: {output_file}")
    
    def create_visualizations(self, output_dir: str = "bowler_analysis_plots"):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Bowler comparison across scenarios
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for plotting with error handling
        bowler_names = []
        powerplay_acc = []
        middle_acc = []
        death_acc = []
        economy_rates = []
        
        for bowler_name, performance in self.results['bowler_performance'].items():
            # Use a shortened version of the bowler name for readability
            short_name = bowler_name.split('_')[1] if '_' in bowler_name else bowler_name[:6]
            bowler_names.append(short_name)
            
            # Get scenario accuracies with safe defaults
            pp_stats = performance['summary'].get('Powerplay', {})
            mid_stats = performance['summary'].get('Middle Overs', {})
            death_stats = performance['summary'].get('Death Overs', {})
            
            # Extract values with error handling
            pp_acc_value = pp_stats.get('avg_runs_accuracy', 0)
            mid_acc_value = mid_stats.get('avg_runs_accuracy', 0)
            death_acc_value = death_stats.get('avg_runs_accuracy', 0)
            
            # Replace NaN values with zeros
            pp_acc_value = 0 if np.isnan(pp_acc_value) else pp_acc_value
            mid_acc_value = 0 if np.isnan(mid_acc_value) else mid_acc_value
            death_acc_value = 0 if np.isnan(death_acc_value) else death_acc_value
            
            powerplay_acc.append(pp_acc_value)
            middle_acc.append(mid_acc_value)
            death_acc.append(death_acc_value)
            
            # Compute economy safely
            valid_econ_values = []
            for s in performance['summary'].values():
                econ = s.get('economy_rate', None)
                if econ is not None and not np.isnan(econ) and econ != float('inf'):
                    valid_econ_values.append(econ)
                    
            avg_econ = np.mean(valid_econ_values) if valid_econ_values else 0
            economy_rates.append(avg_econ)
        
        # Runs accuracy by scenario
        x = np.arange(len(bowler_names))
        width = 0.25
        
        ax1.bar(x - width, powerplay_acc, width, label='Powerplay', alpha=0.8)
        ax1.bar(x, middle_acc, width, label='Middle Overs', alpha=0.8)
        ax1.bar(x + width, death_acc, width, label='Death Overs', alpha=0.8)
        ax1.set_xlabel('Bowlers')
        ax1.set_ylabel('Runs Prediction Accuracy (%)')
        ax1.set_title('Runs Prediction Accuracy by Scenario')
        ax1.set_xticks(x)
        ax1.set_xticklabels(bowler_names, rotation=45)
        ax1.legend()
        
        # Economy rates
        ax2.bar(bowler_names, economy_rates, alpha=0.7, color='orange')
        ax2.set_xlabel('Bowlers')
        ax2.set_ylabel('Economy Rate (RPO)')
        ax2.set_title('Economy Rates by Bowler')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Wickets accuracy comparison
        wickets_acc = []
        for bowler_name, performance in self.results['bowler_performance'].items():
            avg_wickets_acc = np.mean([s.get('avg_wickets_accuracy', 0) 
                                     for s in performance['summary'].values()])
            wickets_acc.append(avg_wickets_acc)
        
        ax3.bar(bowler_names, wickets_acc, alpha=0.7, color='red')
        ax3.set_xlabel('Bowlers')
        ax3.set_ylabel('Wickets Prediction Accuracy (%)')
        ax3.set_title('Average Wickets Prediction Accuracy')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Overall performance scatter
        overall_runs_acc = []
        overall_economy = []
        for bowler_name, performance in self.results['bowler_performance'].items():
            avg_runs_acc = np.mean([s.get('avg_runs_accuracy', 0) 
                                  for s in performance['summary'].values()])
            avg_econ = np.mean([s.get('economy_rate', 0) 
                              for s in performance['summary'].values()])
            overall_runs_acc.append(avg_runs_acc)
            overall_economy.append(avg_econ)
        
        ax4.scatter(overall_economy, overall_runs_acc, alpha=0.7, s=100)
        for i, name in enumerate(bowler_names):
            ax4.annotate(name, (overall_economy[i], overall_runs_acc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Economy Rate (RPO)')
        ax4.set_ylabel('Runs Prediction Accuracy (%)')
        ax4.set_title('Prediction Accuracy vs Economy Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bowler_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scenario-wise analysis - handle edge cases better
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, scenario in enumerate(['Powerplay', 'Middle Overs', 'Death Overs']):
            scenario_data = self.results['comparative_stats']['scenario_comparisons'].get(scenario, {})
            runs_accuracy = scenario_data.get('runs_accuracy', [])
            
            # Filter out NaN or invalid values
            valid_runs_accuracy = [acc for acc in runs_accuracy if not np.isnan(acc)]
            
            if valid_runs_accuracy:
                # Calculate a reasonable number of bins based on data size
                num_bins = min(10, max(3, len(valid_runs_accuracy) // 2))
                
                # Create the histogram
                axes[i].hist(valid_runs_accuracy, bins=num_bins, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel('Runs Prediction Accuracy (%)')
                axes[i].set_ylabel('Number of Bowlers')
                axes[i].set_title(f'{scenario} - Accuracy Distribution')
                
                # Plot the mean line if we have valid data
                mean_acc = np.mean(valid_runs_accuracy)
                axes[i].axvline(mean_acc, color='red', linestyle='--',
                              label=f'Mean: {mean_acc:.1f}%')
                axes[i].legend()
            else:
                # Display a message when no valid data
                axes[i].text(0.5, 0.5, f'No valid data for {scenario}',
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[i].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scenario_accuracy_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}/")
    
    def run_complete_analysis(self):
        """Run the complete bowler analysis pipeline"""
        print("Starting Complete Cricket Bowler Performance Analysis...")
        print(f"Configuration:")
        print(f"  - Data Directory: {self.data_dir}")
        print(f"  - Model Path: {self.model_path}")
        print(f"  - Predictions per Over: {self.k_predictions}")
        print(f"  - Number of Bowlers: {self.num_bowlers}")
        print(f"  - Overs per Bowler: {self.overs_per_bowler}")
        print()
        
        # Run analysis
        self.analyze_all_bowlers()
        
        # Generate outputs
        self.generate_bowler_report()
        self.create_visualizations()
        
        print("\nBowler Analysis completed successfully!")
        return self.results

def main():
    """Main function to run the bowler analysis"""
    # Configuration
    analyzer = BowlerPerformanceAnalyzer(
        data_dir="data_generation/processed",
        model_path="checkpoints/best_model_1.pt",
        k_predictions=20,
        num_bowlers=8,
        overs_per_bowler=15
    )
    
    # Run analysis
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*80)
    print("BOWLER ANALYSIS COMPLETED")
    print("="*80)
    print("Generated files:")
    print("  - bowler_performance_report.txt")
    print("  - bowler_analysis_plots/bowler_performance_comparison.png")
    print("  - bowler_analysis_plots/scenario_accuracy_distributions.png")

if __name__ == "__main__":
    main()