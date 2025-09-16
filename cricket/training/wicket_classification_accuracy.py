#!/usr/bin/env python3
"""
Cricket Transformer Wicket Classification Accuracy Test
======================================================

Script to evaluate model accuracy for classifying overs into two categories:
1. 0 wickets in an over
2. More than 1 wicket in an over

This classification is important for strategic decision-making in cricket.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import sys
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import cricket transformer modules
from cricket_transformer import create_cricket_transformer
from data_generation.data_loader import CricketDataLoader


class WicketClassificationEvaluator:
    """Evaluator for wicket classification accuracy"""
    
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
    
    def count_wickets_in_over(self, sequence: List[str]) -> int:
        """Count the number of wickets in an over"""
        return sum(1 for token in sequence if token == 'W')
    
    def classify_over(self, sequence: List[str]) -> int:
        """
        Classify an over as:
        - 0: No wickets
        - 1: More than 1 wicket
        """
        wickets = self.count_wickets_in_over(sequence)
        return 0 if wickets == 0 else 1
    
    def evaluate_wicket_classification(self, dataset_type: str = 'validation') -> Dict:
        """
        Evaluate model's wicket classification accuracy
        
        Args:
            dataset_type: 'train' or 'validation'
        
        Returns:
            Dictionary with classification accuracy metrics
        """
        print(f"\n🎯 Evaluating wicket classification on {dataset_type} dataset...")
        print("=" * 60)
        
        data_loader = self.data_loader.val_loader if dataset_type == 'validation' else self.data_loader.train_loader
        
        all_predictions = []
        all_targets = []
        true_wickets = []
        pred_wickets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                try:
                    # Move to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Get target sequences
                    target_sequences = self._decode_sequences(batch['target_outputs'])
                    
                    # Forward pass for prediction
                    generated = self.model.generate(
                        histories=batch['histories'],
                        contexts=batch['contexts'],
                        start_token_id=self.vocabulary.get('<START>', 0),
                        end_token_id=self.vocabulary.get('<END>', 0),
                        history_mask=batch['history_mask']
                    )
                    
                    # Decode predictions
                    pred_sequences = self._decode_sequences(generated)
                    
                    # Classify overs
                    for pred_seq, target_seq in zip(pred_sequences, target_sequences):
                        pred_class = self.classify_over(pred_seq)
                        target_class = self.classify_over(target_seq)
                        
                        # Store results
                        all_predictions.append(pred_class)
                        all_targets.append(target_class)
                        
                        # Store actual wicket counts
                        pred_wickets.append(self.count_wickets_in_over(pred_seq))
                        true_wickets.append(self.count_wickets_in_over(target_seq))
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
                
                except Exception as e:
                    print(f"  Warning: Skipped batch {batch_idx} due to error: {e}")
                    continue
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(all_predictions, all_targets, pred_wickets, true_wickets)
        
        return metrics
    
    def _decode_sequences(self, sequences_tensor):
        """Decode token sequences from tensor"""
        decoded_sequences = []
        
        for seq in sequences_tensor.cpu().numpy():
            tokens = []
            for token_id in seq:
                if token_id == self.vocabulary.get('<END>', 0):
                    break
                if token_id != self.vocabulary.get('<PAD>', 0) and token_id != self.vocabulary.get('<START>', 0):
                    token = self.idx_to_token.get(token_id, '<UNK>')
                    tokens.append(token)
            decoded_sequences.append(tokens)
        
        return decoded_sequences
    
    def _calculate_classification_metrics(self, predictions, targets, pred_wickets, true_wickets):
        """Calculate classification metrics"""
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        class_names = ['0 wickets', '>1 wicket']
        report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
        metrics['classification_report'] = report
        
        # Calculate class distribution
        class_dist = Counter(targets)
        metrics['class_distribution'] = {
            '0_wickets': class_dist.get(0, 0),
            'more_than_1_wicket': class_dist.get(1, 0)
        }
        
        # Wicket count analysis
        metrics['wicket_analysis'] = {
            'true_mean': np.mean(true_wickets),
            'pred_mean': np.mean(pred_wickets),
            'true_max': np.max(true_wickets),
            'pred_max': np.max(pred_wickets),
            'true_counts': Counter(true_wickets),
            'pred_counts': Counter(pred_wickets)
        }
        
        return metrics
    
    def print_wicket_classification_report(self, metrics):
        """Print wicket classification report"""
        
        print("\n📊 WICKET CLASSIFICATION REPORT")
        print("=" * 60)
        
        # Class distribution
        dist = metrics['class_distribution']
        total = dist['0_wickets'] + dist['more_than_1_wicket']
        print(f"\n📈 CLASS DISTRIBUTION:")
        print(f"   0 wickets: {dist['0_wickets']} ({dist['0_wickets']/total:.2%})")
        print(f"   >1 wicket: {dist['more_than_1_wicket']} ({dist['more_than_1_wicket']/total:.2%})")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        print(f"\n🔢 CONFUSION MATRIX:")
        print(f"              │ Predicted   │")
        print(f"              │ 0    >1     │")
        print(f"   ──────────┼─────────────┤")
        print(f"   Actual  0 │ {cm[0][0]:<5} {cm[0][1]:<7}│")
        print(f"          >1 │ {cm[1][0]:<5} {cm[1][1]:<7}│")
        
        # Classification metrics
        report = metrics['classification_report']
        print(f"\n🎯 CLASSIFICATION METRICS:")
        print(f"   Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Class           │ Precision │ Recall   │ F1-Score │ Support  │")
        print(f"   ───────────────┼───────────┼──────────┼──────────┼──────────┤")
        print(f"   0 wickets      │ {report['0 wickets']['precision']:.4f}    │ {report['0 wickets']['recall']:.4f}   │ {report['0 wickets']['f1-score']:.4f}   │ {report['0 wickets']['support']:<8}│")
        print(f"   >1 wicket      │ {report['>1 wicket']['precision']:.4f}    │ {report['>1 wicket']['recall']:.4f}   │ {report['>1 wicket']['f1-score']:.4f}   │ {report['>1 wicket']['support']:<8}│")
        print(f"   ───────────────┼───────────┼──────────┼──────────┼──────────┤")
        print(f"   Weighted Avg   │ {report['weighted avg']['precision']:.4f}    │ {report['weighted avg']['recall']:.4f}   │ {report['weighted avg']['f1-score']:.4f}   │ {report['weighted avg']['support']:<8}│")
        
        # Wicket analysis
        wicket = metrics['wicket_analysis']
        print(f"\n🏏 WICKET COUNT ANALYSIS:")
        print(f"   Actual mean: {wicket['true_mean']:.2f} wickets per over")
        print(f"   Predicted mean: {wicket['pred_mean']:.2f} wickets per over")
        
        # Create formatted distribution table
        print(f"\n📊 WICKET DISTRIBUTION COMPARISON:")
        max_wickets = max(max(wicket['true_counts'].keys()), max(wicket['pred_counts'].keys()))
        
        print(f"   Wickets │ Actual    │ Predicted │")
        print(f"   ────────┼───────────┼───────────┤")
        
        for i in range(max_wickets + 1):
            true_count = wicket['true_counts'].get(i, 0)
            pred_count = wicket['pred_counts'].get(i, 0)
            true_pct = true_count / sum(wicket['true_counts'].values()) if true_count else 0
            pred_pct = pred_count / sum(wicket['pred_counts'].values()) if pred_count else 0
            
            print(f"   {i:<7} │ {true_count:<5} ({true_pct:.1%}) │ {pred_count:<5} ({pred_pct:.1%}) │")
        
        print("\n" + "=" * 60)
        return metrics['accuracy']


def main():
    """Main function to evaluate wicket classification accuracy"""
    
    print("🏏 Cricket Transformer - Wicket Classification Analysis")
    print("=" * 60)
    
    # Configure the model path to use best_model_1.pt
    model_path = "checkpoints/best_model_1.pt"
    
    # Find data directory
    possible_data_dirs = [
        "data_generation/processed",
        "../data_generation/processed"
    ]
    
    data_dir = None
    for path in possible_data_dirs:
        if Path(path).exists():
            data_dir = path
            break
    
    if not data_dir:
        print("❌ No processed data directory found!")
        return
    
    if not Path(model_path).exists():
        print(f"❌ Model checkpoint not found: {model_path}")
        return
    
    try:
        # Create evaluator
        evaluator = WicketClassificationEvaluator(model_path, data_dir)
        
        # Evaluate on validation set
        val_metrics = evaluator.evaluate_wicket_classification('validation')
        val_accuracy = evaluator.print_wicket_classification_report(val_metrics)
        
        # Evaluate on training set
        print("\n" + "=" * 60)
        print("Evaluating on training set...")
        train_metrics = evaluator.evaluate_wicket_classification('train')
        train_accuracy = evaluator.print_wicket_classification_report(train_metrics)
        
        # Summary
        print(f"\n🏆 SUMMARY")
        print("=" * 60)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        print("\nClassification Results:")
        print("-" * 60)
        print(f"{'Dataset':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} |")
        print("-" * 60)
        
        # Validation metrics
        val_report = val_metrics['classification_report']
        val_weighted = val_report['weighted avg']
        print(f"{'Validation':<12} | {val_accuracy:.4f}     | {val_weighted['precision']:.4f}     | {val_weighted['recall']:.4f}     | {val_weighted['f1-score']:.4f}     |")
        
        # Training metrics
        train_report = train_metrics['classification_report']
        train_weighted = train_report['weighted avg']
        print(f"{'Training':<12} | {train_accuracy:.4f}     | {train_weighted['precision']:.4f}     | {train_weighted['recall']:.4f}     | {train_weighted['f1-score']:.4f}     |")
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()