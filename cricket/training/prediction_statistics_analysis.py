"""
Cricket Prediction Statistics Analysis
=====================================

This script analyzes prediction accuracy by comparing predicted vs actual outcomes
for cricket overs. It performs multiple predictions (k=20) per over and calculates
comprehensive statistics.

Features:
- Load random matches from processed data
- Predict each over k=20 times using trained model
- Compare runs, wickets, and individual ball outcomes
- Calculate accuracy percentages and statistical significance
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
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cricket_transformer import CricketTransformer, create_cricket_transformer

class PredictionStatisticsAnalyzer:
    """Analyzes cricket prediction statistics by comparing predicted vs actual outcomes"""
    
    def __init__(self, 
                 data_dir: str = "../data_generation/processed_ipl",
                 model_path: str = "checkpoints/best_model.pt",
                 k_predictions: int = 20,
                 num_matches: int = 5):
        """
        Initialize the analyzer
        
        Args:
            data_dir: Directory containing processed data files
            model_path: Path to trained model checkpoint
            k_predictions: Number of predictions per over (k)
            num_matches: Number of random matches to analyze
        """
        self.data_dir = data_dir
        self.model_path = model_path
        self.k_predictions = k_predictions
        self.num_matches = num_matches
        
        # Load vocabulary and metadata
        self.load_vocabulary()
        self.load_metadata()
        
        # Load model
        self.load_model()
        
        # Results storage
        self.results = {
            'match_analysis': [],
            'over_statistics': [],
            'ball_statistics': [],
            'accuracy_metrics': {}
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
        print(f"Special tokens - START: {self.start_token_id}, END: {self.end_token_id}, PAD: {self.pad_token_id}")
    
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
        print(f"Loaded model with {self.model.count_parameters():,} parameters on {self.device}")
    
    def load_processed_data(self):
        """Load processed training data"""
        print("Loading processed data...")
        
        # Load data files
        with open(os.path.join(self.data_dir, "match_histories.pkl"), 'rb') as f:
            histories = pickle.load(f)
        
        with open(os.path.join(self.data_dir, "contexts.pkl"), 'rb') as f:
            contexts = pickle.load(f)
        
        with open(os.path.join(self.data_dir, "targets.pkl"), 'rb') as f:
            targets = pickle.load(f)
        
        print(f"Loaded {len(histories)} training examples")
        return histories, contexts, targets
    
    def select_random_matches(self, histories, contexts, targets):
        """Select random matches for analysis"""
        # Group sequences by match (assuming they are in chronological order)
        # We'll take every 20 sequences as one match (approximate)
        total_sequences = len(histories)
        sequences_per_match = 20  # Approximate overs per match
        
        # Calculate available matches
        available_matches = total_sequences // sequences_per_match
        print(f"Estimated {available_matches} matches available")
        
        # Select random match indices
        selected_match_indices = random.sample(range(available_matches), 
                                             min(self.num_matches, available_matches))
        
        selected_data = []
        for match_idx in selected_match_indices:
            start_idx = match_idx * sequences_per_match
            end_idx = min(start_idx + sequences_per_match, total_sequences)
            
            match_histories = histories[start_idx:end_idx]
            match_contexts = contexts[start_idx:end_idx]
            match_targets = targets[start_idx:end_idx]
            
            selected_data.append({
                'match_id': match_idx,
                'histories': match_histories,
                'contexts': match_contexts,
                'targets': match_targets
            })
        
        print(f"Selected {len(selected_data)} matches for analysis")
        return selected_data
    
    def predict_over_multiple_times(self, history, context, k=20):
        """
        Predict an over k times using the model
        
        Args:
            history: Match history tensor [seq_len, ball_vector_dim]
            context: Context tensor [context_dim]
            k: Number of predictions to make
            
        Returns:
            List of k predictions, each containing token sequences
        """
        predictions = []
        
        # Convert to batch format and move to device
        history_batch = history.unsqueeze(0).to(self.device)  # [1, seq_len, ball_vector_dim]
        context_batch = context.unsqueeze(0).to(self.device)  # [1, context_dim]
        
        with torch.no_grad():
            for _ in range(k):
                # Generate prediction
                generated = self.model.generate(
                    histories=history_batch,
                    contexts=context_batch,
                    start_token_id=self.start_token_id,
                    end_token_id=self.end_token_id,
                    max_length=15,
                    temperature=1.0,  # Add some randomness
                    top_k=None,
                    top_p=None
                )
                
                # Extract prediction (remove start token, stop at end token)
                pred_tokens = generated[0, 1:].tolist()  # Remove start token
                
                # Stop at end token
                if self.end_token_id in pred_tokens:
                    end_idx = pred_tokens.index(self.end_token_id)
                    pred_tokens = pred_tokens[:end_idx]
                
                # Convert to token strings
                pred_sequence = [self.idx_to_token.get(token_id, f"UNK_{token_id}") 
                               for token_id in pred_tokens]
                
                predictions.append(pred_sequence)
        
        return predictions
    
    def analyze_over_outcomes(self, predictions: List[List[str]], actual: List[str]):
        """
        Analyze predicted vs actual outcomes for an over
        
        Args:
            predictions: List of k predicted sequences
            actual: Actual sequence
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'actual': actual,
            'predictions': predictions,
            'runs_analysis': {},
            'wickets_analysis': {},
            'ball_analysis': {}
        }
        
        # Calculate runs and wickets for actual
        actual_runs = self.calculate_runs(actual)
        actual_wickets = self.calculate_wickets(actual)
        
        # Calculate runs and wickets for each prediction
        pred_runs = [self.calculate_runs(pred) for pred in predictions]
        pred_wickets = [self.calculate_wickets(pred) for pred in predictions]
        
        # Runs analysis
        analysis['runs_analysis'] = {
            'actual_runs': actual_runs,
            'predicted_runs': pred_runs,
            'mean_predicted': np.mean(pred_runs),
            'std_predicted': np.std(pred_runs),
            'correct_predictions': sum(1 for r in pred_runs if r == actual_runs),
            'accuracy_percentage': (sum(1 for r in pred_runs if r == actual_runs) / len(pred_runs)) * 100
        }
        
        # Wickets analysis
        analysis['wickets_analysis'] = {
            'actual_wickets': actual_wickets,
            'predicted_wickets': pred_wickets,
            'mean_predicted': np.mean(pred_wickets),
            'std_predicted': np.std(pred_wickets),
            'correct_predictions': sum(1 for w in pred_wickets if w == actual_wickets),
            'accuracy_percentage': (sum(1 for w in pred_wickets if w == actual_wickets) / len(pred_wickets)) * 100
        }
        
        # Ball-by-ball analysis
        ball_accuracy = []
        max_len = max(len(actual), max(len(pred) for pred in predictions) if predictions else 0)
        
        for ball_idx in range(max_len):
            actual_ball = actual[ball_idx] if ball_idx < len(actual) else None
            pred_balls = [pred[ball_idx] if ball_idx < len(pred) else None for pred in predictions]
            
            if actual_ball is not None:
                correct_preds = sum(1 for pb in pred_balls if pb == actual_ball)
                accuracy = (correct_preds / len(pred_balls)) * 100 if pred_balls else 0
                ball_accuracy.append({
                    'ball_number': ball_idx + 1,
                    'actual': actual_ball,
                    'predicted_distribution': Counter(pb for pb in pred_balls if pb is not None),
                    'accuracy_percentage': accuracy
                })
        
        analysis['ball_analysis'] = ball_accuracy
        
        return analysis
    
    def calculate_runs(self, sequence: List[str]) -> int:
        """Calculate total runs from a token sequence"""
        total_runs = 0
        
        for token in sequence:
            if token in ['0', '1', '2', '3', '4', '5', '6']:
                total_runs += int(token)
            elif token.startswith('wd'):
                # Wide: 1 run + any additional runs
                if len(token) > 2:
                    total_runs += 1 + int(token[2:])
                else:
                    total_runs += 1
            elif token.startswith('nb'):
                # No ball: 1 run + any additional runs
                if len(token) > 2:
                    total_runs += 1 + int(token[2:])
                else:
                    total_runs += 1
            # Note: lb (leg bye) and b (bye) runs are already included in the token value
        
        return total_runs
    
    def calculate_wickets(self, sequence: List[str]) -> int:
        """Calculate total wickets from a token sequence"""
        return sum(1 for token in sequence if token == 'W')
    
    def analyze_all_matches(self):
        """Analyze all selected matches"""
        print(f"Starting analysis of {self.num_matches} matches with {self.k_predictions} predictions per over...")
        
        # Load data
        histories, contexts, targets = self.load_processed_data()
        selected_matches = self.select_random_matches(histories, contexts, targets)
        
        overall_stats = {
            'total_overs': 0,
            'runs_accuracy': [],
            'wickets_accuracy': [],
            'ball_accuracy': [],
            'runs_distribution': defaultdict(int),
            'wickets_distribution': defaultdict(int)
        }
        
        for match_data in selected_matches:
            match_id = match_data['match_id']
            print(f"\nAnalyzing Match {match_id}...")
            
            match_stats = {
                'match_id': match_id,
                'overs': [],
                'summary': {}
            }
            
            for over_idx, (history, context, target) in enumerate(zip(
                match_data['histories'], 
                match_data['contexts'], 
                match_data['targets']
            )):
                print(f"  Analyzing Over {over_idx + 1}...")
                
                # Convert data to tensors and move to device
                history_tensor = torch.tensor(history, dtype=torch.float32).to(self.device)
                context_tensor = torch.tensor(context, dtype=torch.float32).to(self.device)
                
                # Convert target to token sequence
                actual_sequence = [self.idx_to_token.get(token_id, f"UNK_{token_id}") 
                                 for token_id in target if token_id not in [self.pad_token_id, self.start_token_id, self.end_token_id]]
                
                # Make k predictions
                predictions = self.predict_over_multiple_times(
                    history_tensor, context_tensor, self.k_predictions
                )
                
                # Analyze this over
                over_analysis = self.analyze_over_outcomes(predictions, actual_sequence)
                
                # Store results
                match_stats['overs'].append(over_analysis)
                self.results['over_statistics'].append({
                    'match_id': match_id,
                    'over_number': over_idx + 1,
                    **over_analysis
                })
                
                # Update overall statistics
                overall_stats['total_overs'] += 1
                overall_stats['runs_accuracy'].append(over_analysis['runs_analysis']['accuracy_percentage'])
                overall_stats['wickets_accuracy'].append(over_analysis['wickets_analysis']['accuracy_percentage'])
                
                # Collect runs and wickets distribution
                actual_runs = over_analysis['runs_analysis']['actual_runs']
                actual_wickets = over_analysis['wickets_analysis']['actual_wickets']
                overall_stats['runs_distribution'][actual_runs] += 1
                overall_stats['wickets_distribution'][actual_wickets] += 1
                
                # Ball-by-ball accuracy
                for ball_stats in over_analysis['ball_analysis']:
                    overall_stats['ball_accuracy'].append(ball_stats['accuracy_percentage'])
            
            # Calculate match summary
            match_runs_acc = np.mean([over['runs_analysis']['accuracy_percentage'] 
                                    for over in match_stats['overs']])
            match_wickets_acc = np.mean([over['wickets_analysis']['accuracy_percentage'] 
                                       for over in match_stats['overs']])
            
            match_stats['summary'] = {
                'average_runs_accuracy': match_runs_acc,
                'average_wickets_accuracy': match_wickets_acc,
                'total_overs': len(match_stats['overs'])
            }
            
            self.results['match_analysis'].append(match_stats)
            print(f"  Match {match_id} Summary - Runs Accuracy: {match_runs_acc:.1f}%, Wickets Accuracy: {match_wickets_acc:.1f}%")
        
        # Calculate final statistics
        self.results['accuracy_metrics'] = {
            'total_overs_analyzed': overall_stats['total_overs'],
            'average_runs_accuracy': np.mean(overall_stats['runs_accuracy']),
            'std_runs_accuracy': np.std(overall_stats['runs_accuracy']),
            'average_wickets_accuracy': np.mean(overall_stats['wickets_accuracy']),
            'std_wickets_accuracy': np.std(overall_stats['wickets_accuracy']),
            'average_ball_accuracy': np.mean(overall_stats['ball_accuracy']),
            'std_ball_accuracy': np.std(overall_stats['ball_accuracy']),
            'runs_distribution': dict(overall_stats['runs_distribution']),
            'wickets_distribution': dict(overall_stats['wickets_distribution'])
        }
        
        print(f"\n" + "="*60)
        print("FINAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Overs Analyzed: {self.results['accuracy_metrics']['total_overs_analyzed']}")
        print(f"Predictions per Over: {self.k_predictions}")
        print(f"Total Predictions Made: {self.results['accuracy_metrics']['total_overs_analyzed'] * self.k_predictions}")
        print(f"\nACCURACY METRICS:")
        print(f"Average Runs Accuracy: {self.results['accuracy_metrics']['average_runs_accuracy']:.2f}% ± {self.results['accuracy_metrics']['std_runs_accuracy']:.2f}%")
        print(f"Average Wickets Accuracy: {self.results['accuracy_metrics']['average_wickets_accuracy']:.2f}% ± {self.results['accuracy_metrics']['std_wickets_accuracy']:.2f}%")
        print(f"Average Ball-by-Ball Accuracy: {self.results['accuracy_metrics']['average_ball_accuracy']:.2f}% ± {self.results['accuracy_metrics']['std_ball_accuracy']:.2f}%")
    
    def generate_detailed_report(self, output_file: str = "prediction_analysis_report.txt"):
        """Generate a detailed text report"""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CRICKET PREDICTION STATISTICS ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 20 + "\n")
            metrics = self.results['accuracy_metrics']
            f.write(f"Total Overs Analyzed: {metrics['total_overs_analyzed']}\n")
            f.write(f"Predictions per Over: {self.k_predictions}\n")
            f.write(f"Total Predictions: {metrics['total_overs_analyzed'] * self.k_predictions}\n\n")
            
            f.write("ACCURACY METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Runs Prediction Accuracy: {metrics['average_runs_accuracy']:.2f}% ± {metrics['std_runs_accuracy']:.2f}%\n")
            f.write(f"Wickets Prediction Accuracy: {metrics['average_wickets_accuracy']:.2f}% ± {metrics['std_wickets_accuracy']:.2f}%\n")
            f.write(f"Ball-by-Ball Accuracy: {metrics['average_ball_accuracy']:.2f}% ± {metrics['std_ball_accuracy']:.2f}%\n\n")
            
            # Distribution analysis
            f.write("ACTUAL OUTCOMES DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            f.write("Runs per Over:\n")
            for runs, count in sorted(metrics['runs_distribution'].items()):
                percentage = (count / metrics['total_overs_analyzed']) * 100
                f.write(f"  {runs} runs: {count} overs ({percentage:.1f}%)\n")
            
            f.write("\nWickets per Over:\n")
            for wickets, count in sorted(metrics['wickets_distribution'].items()):
                percentage = (count / metrics['total_overs_analyzed']) * 100
                f.write(f"  {wickets} wickets: {count} overs ({percentage:.1f}%)\n")
            
            # Match-by-match analysis
            f.write("\n" + "="*80 + "\n")
            f.write("MATCH-BY-MATCH ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            for match in self.results['match_analysis']:
                f.write(f"MATCH {match['match_id']}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Overs: {match['summary']['total_overs']}\n")
                f.write(f"Average Runs Accuracy: {match['summary']['average_runs_accuracy']:.2f}%\n")
                f.write(f"Average Wickets Accuracy: {match['summary']['average_wickets_accuracy']:.2f}%\n\n")
                
                # Top 5 best and worst predicted overs
                overs_by_runs_acc = sorted(match['overs'], 
                                         key=lambda x: x['runs_analysis']['accuracy_percentage'], 
                                         reverse=True)
                
                f.write("Top 3 Best Runs Predictions:\n")
                for i, over in enumerate(overs_by_runs_acc[:3]):
                    over_idx = match['overs'].index(over) + 1
                    f.write(f"  Over {over_idx}: {over['runs_analysis']['accuracy_percentage']:.1f}% "
                           f"({over['runs_analysis']['actual_runs']} runs predicted correctly "
                           f"{over['runs_analysis']['correct_predictions']}/{self.k_predictions} times)\n")
                
                f.write("Bottom 3 Runs Predictions:\n")
                for i, over in enumerate(overs_by_runs_acc[-3:]):
                    over_idx = match['overs'].index(over) + 1
                    f.write(f"  Over {over_idx}: {over['runs_analysis']['accuracy_percentage']:.1f}% "
                           f"({over['runs_analysis']['actual_runs']} runs predicted correctly "
                           f"{over['runs_analysis']['correct_predictions']}/{self.k_predictions} times)\n")
                
                f.write("\n")
        
        print(f"Detailed report saved to: {output_file}")
    
    def create_visualizations(self, output_dir: str = "prediction_analysis_plots"):
        """Create visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Accuracy Distribution
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Runs accuracy histogram
        runs_accuracies = [over['runs_analysis']['accuracy_percentage'] 
                          for match in self.results['match_analysis'] 
                          for over in match['overs']]
        ax1.hist(runs_accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Runs Prediction Accuracy (%)')
        ax1.set_ylabel('Number of Overs')
        ax1.set_title('Distribution of Runs Prediction Accuracy')
        ax1.axvline(np.mean(runs_accuracies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(runs_accuracies):.1f}%')
        ax1.legend()
        
        # Wickets accuracy histogram
        wickets_accuracies = [over['wickets_analysis']['accuracy_percentage'] 
                             for match in self.results['match_analysis'] 
                             for over in match['overs']]
        ax2.hist(wickets_accuracies, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Wickets Prediction Accuracy (%)')
        ax2.set_ylabel('Number of Overs')
        ax2.set_title('Distribution of Wickets Prediction Accuracy')
        ax2.axvline(np.mean(wickets_accuracies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(wickets_accuracies):.1f}%')
        ax2.legend()
        
        # Runs distribution
        runs_dist = self.results['accuracy_metrics']['runs_distribution']
        ax3.bar(runs_dist.keys(), runs_dist.values(), alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Runs per Over')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Actual Runs Distribution')
        
        # Wickets distribution
        wickets_dist = self.results['accuracy_metrics']['wickets_distribution']
        ax4.bar(wickets_dist.keys(), wickets_dist.values(), alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Wickets per Over')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Actual Wickets Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Match-wise comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        match_ids = [match['match_id'] for match in self.results['match_analysis']]
        match_runs_acc = [match['summary']['average_runs_accuracy'] for match in self.results['match_analysis']]
        match_wickets_acc = [match['summary']['average_wickets_accuracy'] for match in self.results['match_analysis']]
        
        ax1.bar(match_ids, match_runs_acc, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Match ID')
        ax1.set_ylabel('Average Runs Accuracy (%)')
        ax1.set_title('Runs Prediction Accuracy by Match')
        ax1.set_ylim(0, 100)
        
        ax2.bar(match_ids, match_wickets_acc, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Match ID')
        ax2.set_ylabel('Average Wickets Accuracy (%)')
        ax2.set_title('Wickets Prediction Accuracy by Match')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'match_wise_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}/")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Complete Cricket Prediction Analysis...")
        print(f"Configuration:")
        print(f"  - Data Directory: {self.data_dir}")
        print(f"  - Model Path: {self.model_path}")
        print(f"  - Predictions per Over (k): {self.k_predictions}")
        print(f"  - Number of Matches: {self.num_matches}")
        print()
        
        # Run analysis
        self.analyze_all_matches()
        
        # Generate outputs
        self.generate_detailed_report()
        self.create_visualizations()
        
        print("\nAnalysis completed successfully!")
        return self.results

def main():
    """Main function to run the analysis"""
    # Configuration
    analyzer = PredictionStatisticsAnalyzer(
        data_dir="data_generation/processed",
        model_path="checkpoints/best_model_1.pt",
        k_predictions=20,
        num_matches=5
    )
    
    # Run analysis
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    print("Generated files:")
    print("  - prediction_analysis_report.txt")
    print("  - prediction_analysis_plots/accuracy_distributions.png")
    print("  - prediction_analysis_plots/match_wise_accuracy.png")

if __name__ == "__main__":
    main()