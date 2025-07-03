#!/usr/bin/env python3
"""
Quick Accuracy Test Runner
==========================

Simple script to run model accuracy evaluation with proper error handling
and path detection.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_model_checkpoint():
    """Find the best available model checkpoint"""
    
    possible_paths = [
        "training/checkpoints/best_model.pt",
        "checkpoints/best_model.pt",
        "training/checkpoints/checkpoint_epoch_10.pt",
        "checkpoints/checkpoint_epoch_10.pt"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    # Look for any checkpoint file
    checkpoint_dirs = ["training/checkpoints", "checkpoints"]
    for checkpoint_dir in checkpoint_dirs:
        if Path(checkpoint_dir).exists():
            checkpoint_files = list(Path(checkpoint_dir).glob("*.pt"))
            if checkpoint_files:
                return str(checkpoint_files[0])
    
    return None

def find_data_directory():
    """Find the processed data directory"""
    
    possible_paths = [
        "data_generation/processed",
        "../data_generation/processed"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return None

def main():
    """Run accuracy evaluation with path detection"""
    
    print("🏏 Cricket Transformer - Quick Accuracy Test")
    print("=" * 50)
    
    # Find model checkpoint
    model_path = find_model_checkpoint()
    if not model_path:
        print("❌ No model checkpoint found!")
        print("Available locations checked:")
        print("  - training/checkpoints/")
        print("  - checkpoints/")
        print("\nPlease train the model first using:")
        print("  python training/trainer.py")
        return False
    
    print(f"✓ Found model: {model_path}")
    
    # Find data directory
    data_dir = find_data_directory()
    if not data_dir:
        print("❌ No processed data found!")
        print("Available locations checked:")
        print("  - data_generation/processed/")
        print("  - ../data_generation/processed/")
        print("\nPlease run data generation first using:")
        print("  python data_generation/generate_data.py")
        return False
    
    print(f"✓ Found data: {data_dir}")
    
    # Import and run evaluation
    try:
        from model_accuracy_evaluation import ModelAccuracyEvaluator
        
        print(f"\n🎯 Starting accuracy evaluation...")
        print("-" * 50)
        
        # Create evaluator
        evaluator = ModelAccuracyEvaluator(model_path, data_dir)
        
        # Run evaluation on validation set
        print(f"\n📊 Evaluating on validation set...")
        val_metrics = evaluator.evaluate_full_dataset('validation')
        val_score = evaluator.print_accuracy_report(val_metrics, 'validation')
        
        # Quick summary
        print(f"\n🎯 QUICK SUMMARY")
        print("=" * 50)
        print(f"Overall Accuracy Score: {val_score:.4f}")
        
        basic = val_metrics['basic']
        print(f"Exact Match Accuracy: {basic['exact_match_accuracy']:.4f}")
        print(f"Token-Level Accuracy: {basic['token_accuracy']:.4f}")
        
        runs = val_metrics['runs']
        print(f"Run Prediction Accuracy: {runs['exact_run_accuracy']:.4f}")
        
        # Performance assessment
        if val_score >= 0.8:
            print(f"🌟 EXCELLENT performance!")
        elif val_score >= 0.6:
            print(f"✅ GOOD performance!")
        elif val_score >= 0.4:
            print(f"⚠️  FAIR performance - room for improvement")
        else:
            print(f"❌ POOR performance - needs significant improvement")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ Accuracy evaluation completed successfully!")
    else:
        print(f"\n❌ Accuracy evaluation failed!")
        sys.exit(1)