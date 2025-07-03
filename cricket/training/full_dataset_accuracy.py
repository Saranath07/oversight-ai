#!/usr/bin/env python3
"""
Full Dataset Accuracy Evaluation
================================

Evaluate model accuracy on the ENTIRE dataset (training + validation combined)
to get overall performance against all ground truth data.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cricket_transformer import create_cricket_transformer
from data_generation.data_loader import CricketDataset, collate_fn, decode_predictions

class FullDatasetEvaluator:
    """Evaluate model on entire dataset"""
    
    def __init__(self, model_path: str, data_dir: str):
        """Initialize evaluator"""
        self.model_path = model_path
        self.data_dir = data_dir
        
        # Load model
        self.model, self.vocabulary, self.idx_to_token = self._load_model()
        
        # Load entire dataset (no train/val split)
        self.dataset = CricketDataset(data_dir, max_history_length=128)
        
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
        print(f"✓ Full dataset loaded: {len(self.dataset)} sequences")
        print(f"✓ Using device: {self.device}")
    
    def _load_model(self):
        """Load trained model"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        config = checkpoint['config']
        vocabulary = checkpoint['vocabulary']
        idx_to_token = {v: k for k, v in vocabulary.items()}
        
        model = create_cricket_transformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, vocabulary, idx_to_token
    
    def evaluate_entire_dataset(self, batch_size: int = 32) -> Dict:
        """Evaluate model on entire dataset"""
        
        print(f"\n🎯 EVALUATING MODEL ON ENTIRE DATASET")
        print("=" * 60)
        print(f"Total sequences: {len(self.dataset):,}")
        print(f"Batch size: {batch_size}")
        
        # Create data loader for entire dataset
        from torch.utils.data import DataLoader
        data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        all_predictions = []
        all_targets = []
        all_target_tokens = []
        batch_losses = []
        processed_sequences = 0
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        print(f"\nProcessing {len(data_loader)} batches...")
        
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
                    all_target_tokens.extend(batch['target_tokens'])
                    
                    processed_sequences += batch['histories'].shape[0]
                    
                    # Progress update
                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
                        progress = (batch_idx + 1) / len(data_loader) * 100
                        print(f"  Progress: {batch_idx + 1}/{len(data_loader)} batches ({progress:.1f}%) - {processed_sequences:,} sequences")
                
                except Exception as e:
                    print(f"  Warning: Skipped batch {batch_idx} due to error: {e}")
                    continue
        
        print(f"\n✓ Processed {processed_sequences:,} sequences successfully")
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            all_predictions, all_targets, all_target_tokens, batch_losses
        )
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        moved_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if self.device.type == 'mps' and v.dtype == torch.float64:
                    v = v.float()
                moved_batch[k] = v.to(self.device)
            else:
                moved_batch[k] = v
        return moved_batch
    
    def _calculate_comprehensive_metrics(self, predictions: List, targets: List, 
                                       target_tokens: List, losses: List) -> Dict:
        """Calculate comprehensive accuracy metrics"""
        
        print(f"\n📊 CALCULATING COMPREHENSIVE ACCURACY METRICS")
        print("-" * 60)
        
        # Decode predictions and targets
        decoded_predictions = []
        decoded_targets = []
        
        for pred_seq, target_seq, orig_tokens in zip(predictions, targets, target_tokens):
            # Decode prediction
            pred_tokens = []
            for token_id in pred_seq:
                if token_id == self.vocabulary.get('<END>', 0):
                    break
                if token_id != self.vocabulary.get('<PAD>', 0) and token_id != self.vocabulary.get('<START>', 0):
                    token = self.idx_to_token.get(token_id, '<UNK>')
                    pred_tokens.append(token)
            
            # Use original target tokens (ground truth)
            decoded_predictions.append(pred_tokens)
            decoded_targets.append(orig_tokens)
        
        print(f"✓ Decoded {len(decoded_predictions):,} prediction sequences")
        
        # Calculate all metrics
        metrics = {}
        
        # 1. Basic Accuracy
        print("  Calculating basic accuracy...")
        metrics['basic'] = self._calculate_basic_accuracy(decoded_predictions, decoded_targets)
        
        # 2. Ball Outcome Accuracy
        print("  Calculating ball outcome accuracy...")
        metrics['ball_outcomes'] = self._calculate_ball_outcome_accuracy(decoded_predictions, decoded_targets)
        
        # 3. Run Prediction Accuracy
        print("  Calculating run prediction accuracy...")
        metrics['runs'] = self._calculate_run_accuracy(decoded_predictions, decoded_targets)
        
        # 4. Sequence Length Accuracy
        print("  Calculating sequence length accuracy...")
        metrics['sequence_length'] = self._calculate_sequence_length_accuracy(decoded_predictions, decoded_targets)
        
        # 5. Loss Statistics
        print("  Calculating loss statistics...")
        metrics['loss'] = {
            'mean_loss': np.mean(losses),
            'median_loss': np.median(losses),
            'std_loss': np.std(losses),
            'perplexity': np.exp(np.mean(losses))
        }
        
        # 6. Token Distribution Analysis
        print("  Analyzing token distributions...")
        metrics['token_distribution'] = self._analyze_token_distribution(decoded_predictions, decoded_targets)
        
        return metrics
    
    def _calculate_basic_accuracy(self, predictions: List[List[str]], targets: List[List[str]]) -> Dict:
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
        
        return {
            'exact_match_accuracy': exact_matches / len(predictions),
            'token_accuracy': token_matches / total_tokens,
            'total_sequences': len(predictions),
            'exact_matches': exact_matches,
            'total_tokens': total_tokens,
            'token_matches': token_matches,
            'position_accuracy': {pos: position_matches[pos] / position_totals[pos] for pos in position_totals}
        }
    
    def _calculate_ball_outcome_accuracy(self, predictions: List[List[str]], targets: List[List[str]]) -> Dict:
        """Calculate accuracy for different ball outcomes"""
        
        outcome_stats = {
            'runs': {'correct': 0, 'total': 0, 'predicted': 0},
            'wickets': {'correct': 0, 'total': 0, 'predicted': 0},
            'extras': {'correct': 0, 'total': 0, 'predicted': 0},
            'dots': {'correct': 0, 'total': 0, 'predicted': 0},
            'boundaries': {'correct': 0, 'total': 0, 'predicted': 0}
        }
        
        for pred, target in zip(predictions, targets):
            for i in range(max(len(pred), len(target))):
                # Get tokens (pad with None if sequence is shorter)
                pred_token = pred[i] if i < len(pred) else None
                target_token = target[i] if i < len(target) else None
                
                # Categorize tokens
                if target_token:
                    target_category = self._categorize_token(target_token)
                    if target_category in outcome_stats:
                        outcome_stats[target_category]['total'] += 1
                
                if pred_token:
                    pred_category = self._categorize_token(pred_token)
                    if pred_category in outcome_stats:
                        outcome_stats[pred_category]['predicted'] += 1
                
                # Check if both exist and match
                if pred_token and target_token:
                    pred_category = self._categorize_token(pred_token)
                    target_category = self._categorize_token(target_token)
                    if pred_category == target_category and target_category in outcome_stats:
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
    
    def _calculate_run_accuracy(self, predictions: List[List[str]], targets: List[List[str]]) -> Dict:
        """Calculate run prediction accuracy with over-by-over comparison"""
        
        over_comparisons = []  # List of {over_id, predicted_runs, actual_runs, difference, match}
        pred_runs = []
        target_runs = []
        
        for over_id, (pred, target) in enumerate(zip(predictions, targets)):
            pred_total = self._calculate_over_runs(pred)
            target_total = self._calculate_over_runs(target)
            
            pred_runs.append(pred_total)
            target_runs.append(target_total)
            
            # Store over-by-over comparison
            difference = pred_total - target_total
            is_match = (pred_total == target_total)
            
            over_comparisons.append({
                'over_id': over_id + 1,
                'predicted_runs': pred_total,
                'actual_runs': target_total,
                'difference': difference,
                'absolute_difference': abs(difference),
                'is_exact_match': is_match,
                'predicted_sequence': pred,
                'actual_sequence': target
            })
        
        # Calculate summary statistics
        exact_run_matches = sum(1 for comp in over_comparisons if comp['is_exact_match'])
        run_differences = [comp['absolute_difference'] for comp in over_comparisons]
        signed_differences = [comp['difference'] for comp in over_comparisons]
        
        # Tolerance-based accuracy
        within_1_run = sum(1 for diff in run_differences if diff <= 1)
        within_2_runs = sum(1 for diff in run_differences if diff <= 2)
        
        # Run range distribution analysis
        run_distribution_comparison = {}
        for runs in range(0, 21):  # 0 to 20+ runs
            pred_count = sum(1 for p in pred_runs if p == runs)
            target_count = sum(1 for t in target_runs if t == runs)
            run_distribution_comparison[f'{runs}_runs'] = {
                'predicted': pred_count,
                'actual': target_count,
                'difference': pred_count - target_count
            }
        
        return {
            # Over-by-over detailed comparison
            'over_by_over_comparison': over_comparisons,
            
            # Summary accuracy metrics
            'exact_run_accuracy': exact_run_matches / len(pred_runs),
            'within_1_run_accuracy': within_1_run / len(pred_runs),
            'within_2_runs_accuracy': within_2_runs / len(pred_runs),
            
            # Error statistics
            'mean_run_error': np.mean(run_differences),
            'median_run_error': np.median(run_differences),
            'std_run_error': np.std(run_differences),
            'max_run_error': max(run_differences) if run_differences else 0,
            
            # Bias analysis (positive = overestimate, negative = underestimate)
            'mean_bias': np.mean(signed_differences),
            'median_bias': np.median(signed_differences),
            
            # Total runs comparison
            'total_predicted_runs': sum(pred_runs),
            'total_actual_runs': sum(target_runs),
            'total_run_difference': sum(pred_runs) - sum(target_runs),
            
            # Distribution comparison
            'run_distribution_comparison': run_distribution_comparison,
            
            # Sample comparisons for quick view
            'sample_comparisons': over_comparisons[:10]  # First 10 overs
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
    
    def _calculate_sequence_length_accuracy(self, predictions: List[List[str]], targets: List[List[str]]) -> Dict:
        """Calculate sequence length accuracy"""
        
        pred_lengths = [len(pred) for pred in predictions]
        target_lengths = [len(target) for target in targets]
        
        exact_length_matches = sum(1 for p, t in zip(pred_lengths, target_lengths) if p == t)
        length_differences = [abs(p - t) for p, t in zip(pred_lengths, target_lengths)]
        
        return {
            'exact_length_accuracy': exact_length_matches / len(predictions),
            'mean_length_error': np.mean(length_differences),
            'median_length_error': np.median(length_differences),
            'predicted_avg_length': np.mean(pred_lengths),
            'target_avg_length': np.mean(target_lengths)
        }
    
    def _analyze_token_distribution(self, predictions: List[List[str]], targets: List[List[str]]) -> Dict:
        """Analyze token distribution differences"""
        
        pred_tokens = [token for seq in predictions for token in seq]
        target_tokens = [token for seq in targets for token in seq]
        
        pred_counter = Counter(pred_tokens)
        target_counter = Counter(target_tokens)
        
        return {
            'predicted_distribution': dict(pred_counter),
            'target_distribution': dict(target_counter),
            'total_predicted_tokens': len(pred_tokens),
            'total_target_tokens': len(target_tokens)
        }
    
    def print_comprehensive_report(self, metrics: Dict):
        """Print comprehensive accuracy report"""
        
        print(f"\n🎯 COMPREHENSIVE MODEL ACCURACY REPORT")
        print("=" * 80)
        print(f"📊 EVALUATED ON ENTIRE DATASET")
        print("=" * 80)
        
        # Basic Accuracy
        basic = metrics['basic']
        print(f"\n📈 BASIC ACCURACY METRICS:")
        print(f"   Total Sequences Evaluated: {basic['total_sequences']:,}")
        print(f"   Exact Match Accuracy: {basic['exact_match_accuracy']:.4f} ({basic['exact_matches']:,}/{basic['total_sequences']:,})")
        print(f"   Token-Level Accuracy: {basic['token_accuracy']:.4f} ({basic['token_matches']:,}/{basic['total_tokens']:,})")
        
        # Position-wise accuracy (first 6 balls)
        print(f"\n📍 BALL-BY-BALL ACCURACY:")
        for pos in sorted(basic['position_accuracy'].keys())[:6]:
            acc = basic['position_accuracy'][pos]
            print(f"   Ball {pos+1}: {acc:.4f}")
        
        # Ball Outcome Accuracy
        outcomes = metrics['ball_outcomes']
        print(f"\n⚾ BALL OUTCOME ACCURACY:")
        for outcome, accuracy in outcomes['outcome_accuracies'].items():
            stats = outcomes['outcome_stats'][outcome]
            print(f"   {outcome.capitalize()}: {accuracy:.4f} ({stats['correct']:,}/{stats['total']:,}) [Predicted: {stats['predicted']:,}]")
        
        # Run Accuracy - Enhanced with over-by-over comparison
        runs = metrics['runs']
        print(f"\n🏃 TOTAL RUNS ACCURACY (Over-by-Over Comparison):")
        print("=" * 60)
        print(f"   Exact Run Matches: {runs['exact_run_accuracy']:.4f} ({int(runs['exact_run_accuracy'] * len(runs['over_by_over_comparison']))}/{len(runs['over_by_over_comparison'])})")
        print(f"   Within 1 Run Accuracy: {runs['within_1_run_accuracy']:.4f}")
        print(f"   Within 2 Runs Accuracy: {runs['within_2_runs_accuracy']:.4f}")
        print(f"   Mean Run Error: {runs['mean_run_error']:.2f}")
        print(f"   Mean Bias: {runs['mean_bias']:+.2f} (+ = overestimate, - = underestimate)")
        
        print(f"\n📊 TOTAL RUNS SUMMARY:")
        print(f"   Total Predicted Runs: {runs['total_predicted_runs']:,}")
        print(f"   Total Actual Runs: {runs['total_actual_runs']:,}")
        print(f"   Total Difference: {runs['total_run_difference']:+,}")
        
        # Show sample over-by-over comparisons
        print(f"\n📋 SAMPLE OVER-BY-OVER COMPARISONS:")
        print("   Over | Predicted | Actual | Diff | Match | Predicted Sequence → Actual Sequence")
        print("   " + "-" * 75)
        
        for comp in runs['sample_comparisons']:
            match_symbol = "✓" if comp['is_exact_match'] else "✗"
            pred_seq_str = " ".join(comp['predicted_sequence'][:6])  # Show first 6 balls
            actual_seq_str = " ".join(comp['actual_sequence'][:6])
            
            print(f"   {comp['over_id']:4d} | {comp['predicted_runs']:9d} | {comp['actual_runs']:6d} | {comp['difference']:+4d} | {match_symbol:5s} | {pred_seq_str} → {actual_seq_str}")
        
        # Show run distribution comparison
        print(f"\n📈 RUN DISTRIBUTION COMPARISON (Most Common):")
        print("   Runs | Predicted | Actual | Difference")
        print("   " + "-" * 35)
        
        # Show top 10 most common run totals
        run_dist = runs['run_distribution_comparison']
        sorted_runs = sorted(run_dist.items(), key=lambda x: x[1]['actual'], reverse=True)[:10]
        
        for run_key, stats in sorted_runs:
            runs_value = run_key.replace('_runs', '')
            print(f"   {runs_value:4s} | {stats['predicted']:9d} | {stats['actual']:6d} | {stats['difference']:+10d}")
        
        # Sequence Length
        seq_len = metrics['sequence_length']
        print(f"\n📏 SEQUENCE LENGTH ACCURACY:")
        print(f"   Exact Length Matches: {seq_len['exact_length_accuracy']:.4f}")
        print(f"   Average Predicted Length: {seq_len['predicted_avg_length']:.2f}")
        print(f"   Average Target Length: {seq_len['target_avg_length']:.2f}")
        
        # Loss Statistics
        loss_stats = metrics['loss']
        print(f"\n📉 LOSS STATISTICS:")
        print(f"   Mean Loss: {loss_stats['mean_loss']:.4f}")
        print(f"   Perplexity: {loss_stats['perplexity']:.2f}")
        
        # Token Distribution
        token_dist = metrics['token_distribution']
        print(f"\n🎲 TOKEN DISTRIBUTION:")
        print(f"   Total Predicted Tokens: {token_dist['total_predicted_tokens']:,}")
        print(f"   Total Target Tokens: {token_dist['total_target_tokens']:,}")
        
        # Overall Assessment
        overall_score = (
            basic['exact_match_accuracy'] * 0.4 +
            basic['token_accuracy'] * 0.3 +
            runs['exact_run_accuracy'] * 0.2 +
            seq_len['exact_length_accuracy'] * 0.1
        )
        
        print(f"\n🎯 OVERALL ACCURACY SCORE: {overall_score:.4f}")
        print("   (Weighted: 40% exact match + 30% token + 20% runs + 10% length)")
        
       
        
        return 

def main():
    """Main evaluation function"""
    
    print("🏏 CRICKET TRANSFORMER - FULL DATASET ACCURACY EVALUATION")
    print("=" * 80)
    
    # Configuration
    model_path = "checkpoints/best_model.pt"
    data_dir = "data_generation/processed"
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model checkpoint not found: {model_path}")
        return
    
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    try:
        # Create evaluator
        evaluator = FullDatasetEvaluator(model_path, data_dir)
        
        # Evaluate on entire dataset
        metrics = evaluator.evaluate_entire_dataset(batch_size=32)
        
        # Print comprehensive report
        evaluator.print_comprehensive_report(metrics)
        
        
        
        return 
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
    
 