#!/usr/bin/env python3
"""
Cricket Transformer Model Accuracy Evaluation
=============================================

Comprehensive evaluation script to measure model accuracy against ground truth data.
Provides detailed metrics including:
- Token-level accuracy
- Sequence-level accuracy  
- Ball outcome accuracy
- Run prediction accuracy
- Statistical analysis across different match situations
"""

import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cricket_transformer import create_cricket_transformer
from data_generation.data_loader import CricketDataLoader, decode_predictions

class ModelAccuracyEvaluator:
    """Comprehensive model accuracy evaluator"""
    
    def __init__(self, model_path: str, data_dir: str):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Path to processed data directory
        """
        self.model_path = model_path
        self.data_dir = data_dir
        
        # Load model and data
        self.model, self.vocabulary, self.idx_to_token = self._load_model()
        self.data_loader = self._load_data()
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Data loaded from: {data_dir}")
        print(f"✓ Using device: {self.device}")
        print(f"✓ Vocabulary size: {len(self.vocabulary)}")
    
    def _load_model(self):
        """Load trained model"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        config = checkpoint['config']
        vocabulary = checkpoint['vocabulary']
        idx_to_token = {v: k for k, v in vocabulary.items()}
        
        model = create_cricket_transformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, vocabulary, idx_to_token
    
    def _load_data(self):
        """Load data loader"""
        return CricketDataLoader(
            data_dir=self.data_dir,
            batch_size=32,
            max_history_length=128,
            train_split=0.8
        )
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device with compatibility"""
        moved_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if self.device.type == 'mps' and v.dtype == torch.float64:
                    v = v.float()
                moved_batch[k] = v.to(self.device)
            else:
                moved_batch[k] = v
        return moved_batch
    
    def evaluate_full_dataset(self, dataset_type: str = 'validation') -> Dict:
        """
        Evaluate model on full dataset
        
        Args:
            dataset_type: 'train' or 'validation'
        
        Returns:
            Dictionary with comprehensive accuracy metrics
        """
        print(f"\n🎯 Evaluating model on {dataset_type} dataset...")
        print("=" * 60)
        
        data_loader = self.data_loader.val_loader if dataset_type == 'validation' else self.data_loader.train_loader
        
        all_predictions = []
        all_targets = []
        all_logits = []
        batch_losses = []
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                try:
                    # Move to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    logits = self.model(
                        histories=batch['histories'],
                        contexts=batch['contexts'],
                        target_inputs=batch['target_inputs'],
                        history_mask=batch['history_mask'],
                        target_mask=batch['target_mask']
                    )
                    
                    # Calculate loss
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        batch['target_outputs'].reshape(-1)
                    )
                    batch_losses.extend(loss.cpu().numpy())
                    
                    # Get predictions
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch['target_outputs'].cpu().numpy())
                    all_logits.extend(logits.cpu().numpy())
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
                
                except Exception as e:
                    print(f"  Warning: Skipped batch {batch_idx} due to error: {e}")
                    continue
        
        print(f"✓ Processed {len(all_predictions)} sequences")
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            all_predictions, all_targets, all_logits, batch_losses
        )
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, predictions: List, targets: List, 
                                       logits: List, losses: List) -> Dict:
        """Calculate comprehensive accuracy metrics"""
        
        print("\n📊 Calculating comprehensive metrics...")
        
        # Decode predictions and targets
        decoded_predictions = []
        decoded_targets = []
        
        for pred_seq, target_seq in zip(predictions, targets):
            # Decode prediction
            pred_tokens = []
            for token_id in pred_seq:
                if token_id == self.vocabulary.get('<END>', 0):
                    break
                if token_id != self.vocabulary.get('<PAD>', 0) and token_id != self.vocabulary.get('<START>', 0):
                    token = self.idx_to_token.get(token_id, '<UNK>')
                    pred_tokens.append(token)
            
            # Decode target
            target_tokens = []
            for token_id in target_seq:
                if token_id == self.vocabulary.get('<END>', 0):
                    break
                if token_id != self.vocabulary.get('<PAD>', 0):
                    token = self.idx_to_token.get(token_id, '<UNK>')
                    target_tokens.append(token)
            
            decoded_predictions.append(pred_tokens)
            decoded_targets.append(target_tokens)
        
        # Calculate metrics
        metrics = {}
        
        # 1. Basic Accuracy Metrics
        metrics['basic'] = self._calculate_basic_accuracy(decoded_predictions, decoded_targets)
        
        # 2. Ball Outcome Accuracy
        metrics['ball_outcomes'] = self._calculate_ball_outcome_accuracy(decoded_predictions, decoded_targets)
        
        # 3. Run Prediction Accuracy
        metrics['runs'] = self._calculate_run_accuracy(decoded_predictions, decoded_targets)
        
        # 4. Sequence Length Accuracy
        metrics['sequence_length'] = self._calculate_sequence_length_accuracy(decoded_predictions, decoded_targets)
        
        # 5. Loss Statistics
        metrics['loss'] = {
            'mean_loss': np.mean(losses),
            'median_loss': np.median(losses),
            'std_loss': np.std(losses),
            'perplexity': np.exp(np.mean(losses))
        }
        
        # 6. Token Distribution Analysis
        metrics['token_distribution'] = self._analyze_token_distribution(decoded_predictions, decoded_targets)
        
        return metrics
    
    def _calculate_basic_accuracy(self, predictions: List[List[str]], 
                                targets: List[List[str]]) -> Dict:
        """Calculate basic accuracy metrics"""
        
        exact_matches = 0
        token_matches = 0
        total_tokens = 0
        position_matches = defaultdict(int)
        position_totals = defaultdict(int)
        
        for pred, target in zip(predictions, targets):
            # Exact sequence match
            if pred == target:
                exact_matches += 1
            
            # Token-level accuracy
            max_len = max(len(pred), len(target))
            for i in range(max_len):
                position_totals[i] += 1
                
                if i < len(pred) and i < len(target):
                    if pred[i] == target[i]:
                        token_matches += 1
                        position_matches[i] += 1
                
                total_tokens += 1
        
        # Position-wise accuracy
        position_accuracy = {}
        for pos in position_totals:
            position_accuracy[pos] = position_matches[pos] / position_totals[pos]
        
        return {
            'exact_match_accuracy': exact_matches / len(predictions),
            'token_accuracy': token_matches / total_tokens,
            'total_sequences': len(predictions),
            'exact_matches': exact_matches,
            'total_tokens': total_tokens,
            'token_matches': token_matches,
            'position_accuracy': position_accuracy
        }
    
    def _calculate_ball_outcome_accuracy(self, predictions: List[List[str]], 
                                       targets: List[List[str]]) -> Dict:
        """Calculate accuracy for different ball outcomes"""
        
        outcome_stats = {
            'runs': {'correct': 0, 'total': 0},
            'wickets': {'correct': 0, 'total': 0},
            'extras': {'correct': 0, 'total': 0},
            'dots': {'correct': 0, 'total': 0},
            'boundaries': {'correct': 0, 'total': 0}
        }
        
        for pred, target in zip(predictions, targets):
            for i in range(min(len(pred), len(target))):
                pred_token = pred[i]
                target_token = target[i]
                
                # Categorize tokens
                pred_category = self._categorize_token(pred_token)
                target_category = self._categorize_token(target_token)
                
                if target_category in outcome_stats:
                    outcome_stats[target_category]['total'] += 1
                    if pred_category == target_category:
                        outcome_stats[target_category]['correct'] += 1
        
        # Calculate accuracies
        outcome_accuracies = {}
        for outcome, stats in outcome_stats.items():
            if stats['total'] > 0:
                outcome_accuracies[outcome] = stats['correct'] / stats['total']
            else:
                outcome_accuracies[outcome] = 0.0
        
        return {
            'outcome_accuracies': outcome_accuracies,
            'outcome_stats': outcome_stats
        }
    
    def _categorize_token(self, token: str) -> str:
        """Categorize a token into outcome type"""
        if token == 'W':
            return 'wickets'
        elif token in ['wd', 'nb1', 'nb2', 'nb3', 'nb4', 'lb', 'bye']:
            return 'extras'
        elif token == '0':
            return 'dots'
        elif token in ['4', '6']:
            return 'boundaries'
        elif token in ['1', '2', '3', '5']:
            return 'runs'
        else:
            return 'other'
    
    def _calculate_run_accuracy(self, predictions: List[List[str]], 
                              targets: List[List[str]]) -> Dict:
        """Calculate run prediction accuracy"""
        
        pred_runs = []
        target_runs = []
        
        for pred, target in zip(predictions, targets):
            pred_total = self._calculate_over_runs(pred)
            target_total = self._calculate_over_runs(target)
            
            pred_runs.append(pred_total)
            target_runs.append(target_total)
        
        # Calculate metrics
        exact_run_matches = sum(1 for p, t in zip(pred_runs, target_runs) if p == t)
        run_differences = [abs(p - t) for p, t in zip(pred_runs, target_runs)]
        
        return {
            'exact_run_accuracy': exact_run_matches / len(pred_runs),
            'mean_run_error': np.mean(run_differences),
            'median_run_error': np.median(run_differences),
            'std_run_error': np.std(run_differences),
            'max_run_error': max(run_differences),
            'predicted_runs': pred_runs[:10],  # Sample
            'target_runs': target_runs[:10]    # Sample
        }
    
    def _calculate_over_runs(self, tokens: List[str]) -> int:
        """Calculate total runs from token sequence"""
        total_runs = 0
        
        for token in tokens:
            if token.isdigit():
                total_runs += int(token)
            elif token == 'wd':
                total_runs += 1
            elif token.startswith('nb'):
                total_runs += 1
                if len(token) > 2 and token[2:].isdigit():
                    total_runs += int(token[2:])
        
        return total_runs
    
    def _calculate_sequence_length_accuracy(self, predictions: List[List[str]], 
                                          targets: List[List[str]]) -> Dict:
        """Calculate sequence length accuracy"""
        
        pred_lengths = [len(pred) for pred in predictions]
        target_lengths = [len(target) for target in targets]
        
        exact_length_matches = sum(1 for p, t in zip(pred_lengths, target_lengths) if p == t)
        length_differences = [abs(p - t) for p, t in zip(pred_lengths, target_lengths)]
        
        return {
            'exact_length_accuracy': exact_length_matches / len(predictions),
            'mean_length_error': np.mean(length_differences),
            'median_length_error': np.median(length_differences),
            'std_length_error': np.std(length_differences),
            'predicted_lengths': Counter(pred_lengths),
            'target_lengths': Counter(target_lengths)
        }
    
    def _analyze_token_distribution(self, predictions: List[List[str]], 
                                  targets: List[List[str]]) -> Dict:
        """Analyze token distribution differences"""
        
        pred_tokens = [token for seq in predictions for token in seq]
        target_tokens = [token for seq in targets for token in seq]
        
        pred_counter = Counter(pred_tokens)
        target_counter = Counter(target_tokens)
        
        # Calculate distribution similarity
        all_tokens = set(pred_tokens + target_tokens)
        distribution_diff = {}
        
        for token in all_tokens:
            pred_freq = pred_counter.get(token, 0) / len(pred_tokens) if pred_tokens else 0
            target_freq = target_counter.get(token, 0) / len(target_tokens) if target_tokens else 0
            distribution_diff[token] = abs(pred_freq - target_freq)
        
        return {
            'predicted_distribution': dict(pred_counter),
            'target_distribution': dict(target_counter),
            'distribution_differences': distribution_diff,
            'total_predicted_tokens': len(pred_tokens),
            'total_target_tokens': len(target_tokens)
        }
    
    def print_accuracy_report(self, metrics: Dict, dataset_type: str = 'validation'):
        """Print comprehensive accuracy report"""
        
        print(f"\n🎯 MODEL ACCURACY REPORT - {dataset_type.upper()} DATASET")
        print("=" * 80)
        
        # Basic Accuracy
        basic = metrics['basic']
        print(f"\n📈 BASIC ACCURACY METRICS:")
        print(f"   Exact Match Accuracy: {basic['exact_match_accuracy']:.4f} ({basic['exact_matches']}/{basic['total_sequences']})")
        print(f"   Token-Level Accuracy: {basic['token_accuracy']:.4f} ({basic['token_matches']}/{basic['total_tokens']})")
        
        # Position-wise accuracy
        print(f"\n📍 POSITION-WISE ACCURACY:")
        for pos in sorted(basic['position_accuracy'].keys())[:6]:  # First 6 positions
            acc = basic['position_accuracy'][pos]
            print(f"   Ball {pos+1}: {acc:.4f}")
        
        # Ball Outcome Accuracy
        outcomes = metrics['ball_outcomes']
        print(f"\n⚾ BALL OUTCOME ACCURACY:")
        for outcome, accuracy in outcomes['outcome_accuracies'].items():
            stats = outcomes['outcome_stats'][outcome]
            print(f"   {outcome.capitalize()}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")
        
        # Run Accuracy
        runs = metrics['runs']
        print(f"\n🏃 RUN PREDICTION ACCURACY:")
        print(f"   Exact Run Matches: {runs['exact_run_accuracy']:.4f}")
        print(f"   Mean Run Error: {runs['mean_run_error']:.2f}")
        print(f"   Median Run Error: {runs['median_run_error']:.2f}")
        print(f"   Max Run Error: {runs['max_run_error']}")
        
        # Sequence Length
        seq_len = metrics['sequence_length']
        print(f"\n📏 SEQUENCE LENGTH ACCURACY:")
        print(f"   Exact Length Matches: {seq_len['exact_length_accuracy']:.4f}")
        print(f"   Mean Length Error: {seq_len['mean_length_error']:.2f}")
        
        # Loss Statistics
        loss_stats = metrics['loss']
        print(f"\n📉 LOSS STATISTICS:")
        print(f"   Mean Loss: {loss_stats['mean_loss']:.4f}")
        print(f"   Perplexity: {loss_stats['perplexity']:.2f}")
        print(f"   Loss Std Dev: {loss_stats['std_loss']:.4f}")
        
        # Token Distribution
        token_dist = metrics['token_distribution']
        print(f"\n🎲 TOKEN DISTRIBUTION:")
        print(f"   Total Predicted Tokens: {token_dist['total_predicted_tokens']:,}")
        print(f"   Total Target Tokens: {token_dist['total_target_tokens']:,}")
        
        # Most common tokens
        pred_dist = token_dist['predicted_distribution']
        target_dist = token_dist['target_distribution']
        
        print(f"\n   Top Predicted Tokens:")
        for token, count in sorted(pred_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {token}: {count} ({count/token_dist['total_predicted_tokens']:.3f})")
        
        print(f"\n   Top Target Tokens:")
        for token, count in sorted(target_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {token}: {count} ({count/token_dist['total_target_tokens']:.3f})")
        
        # Overall Assessment
        overall_score = (
            basic['exact_match_accuracy'] * 0.4 +
            basic['token_accuracy'] * 0.3 +
            runs['exact_run_accuracy'] * 0.2 +
            seq_len['exact_length_accuracy'] * 0.1
        )
        
        print(f"\n🎯 OVERALL ACCURACY SCORE: {overall_score:.4f}")
        print("   (Weighted: 40% exact match + 30% token + 20% runs + 10% length)")
        
        # Performance Assessment
        if overall_score >= 0.8:
            assessment = "EXCELLENT 🌟"
        elif overall_score >= 0.6:
            assessment = "GOOD ✅"
        elif overall_score >= 0.4:
            assessment = "FAIR ⚠️"
        else:
            assessment = "NEEDS IMPROVEMENT ❌"
        
        print(f"\n🏆 MODEL PERFORMANCE: {assessment}")
        
        return overall_score

def main():
    """Main evaluation function"""
    
    print("🏏 CRICKET TRANSFORMER MODEL ACCURACY EVALUATION")
    print("=" * 80)
    
    # Configuration
    model_path = "training/checkpoints/best_model.pt"
    data_dir = "data_generation/processed"
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model checkpoint not found: {model_path}")
        print("Please train the model first or provide correct path.")
        return
    
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run data generation pipeline first.")
        return
    
    try:
        # Create evaluator
        evaluator = ModelAccuracyEvaluator(model_path, data_dir)
        
        # Evaluate on validation set
        val_metrics = evaluator.evaluate_full_dataset('validation')
        val_score = evaluator.print_accuracy_report(val_metrics, 'validation')
        
        # Evaluate on training set (sample)
        print(f"\n" + "="*80)
        train_metrics = evaluator.evaluate_full_dataset('train')
        train_score = evaluator.print_accuracy_report(train_metrics, 'training')
        
        # Summary
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 80)
        print(f"Validation Accuracy: {val_score:.4f}")
        print(f"Training Accuracy: {train_score:.4f}")
        
        if train_score - val_score > 0.1:
            print("⚠️  Model may be overfitting (training >> validation accuracy)")
        elif val_score > train_score:
            print("✅ Model generalizes well (validation >= training accuracy)")
        else:
            print("✅ Model performance is consistent across datasets")
        
        # Save results
        results = {
            'validation_metrics': val_metrics,
            'training_metrics': train_metrics,
            'validation_score': val_score,
            'training_score': train_score,
            'model_path': model_path,
            'data_dir': data_dir
        }
        
        results_path = "training/accuracy_evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()