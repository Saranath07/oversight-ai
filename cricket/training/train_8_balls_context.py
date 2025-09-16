"""
Cricket Transformer Training - 8 Balls Context
==============================================

Training script optimized for 8 balls context window. This approach focuses on 
immediate patterns and recent ball-by-ball dynamics (approximately 1.3 overs).

Key benefits:
- Very recent context (8 balls ≈ 1.3 overs)
- Maximum training samples (1.6M+ samples)
- Focuses on immediate tactical patterns
- Prevents data leakage from same match
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cricket_transformer import CricketTransformer, create_cricket_transformer
from data_generation.data_loader import CricketDataLoader, decode_predictions
from trainer import CricketTrainer

def main():
    """Main training function for 8 balls context"""
    
    print("="*80)
    print("CRICKET TRANSFORMER - 8 BALLS CONTEXT TRAINING")
    print("="*80)
    print("Context: 8 balls (≈1.3 overs)")
    print("Focus: Immediate patterns and recent dynamics")
    print("Expected samples: ~1.6M training samples")
    print("="*80)
    # SMALL Configuration optimized for 8 balls context - Fast experimentation
    config = {
        # Model architecture - SMALL/LIGHTWEIGHT
        'ball_vector_dim': 22,  # Match actual data dimensions
        'context_dim': 20,
        'vocab_size': 24,
        'd_model': 128,        # Reduced from 512 to 128
        'nhead': 4,            # Reduced from 8 to 4
        'num_encoder_layers': 2,  # Reduced from 6 to 2
        'num_decoder_layers': 2,  # Reduced from 6 to 2
        'dim_feedforward': 256,   # Reduced from 2048 to 256
        'dropout': 0.1,
        'max_target_length': 15,
        
        # Training parameters
        'learning_rate': 2e-4,  # Slightly higher for faster convergence
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'batch_size': 32,      # Increased batch size for small model
        
        # 8 BALLS CONTEXT CONFIGURATION
        'use_sliding_window': True,
        'context_window_size': 8,  # 8 balls context
        'max_history_length': 128,  # Legacy fallback
        
        # Training schedule - SHORTER for experimentation
        'num_epochs': 20,        # Reduced from 100 to 20
        'teacher_forcing_epochs': 10,  # Reduced from 50 to 10
        'save_every': 5,
        'validate_every': 1
    }
    
    
    # Create save directory specific to 8 balls context
    save_dir = "checkpoints_8_balls_context"
    Path(save_dir).mkdir(exist_ok=True)
    
    # Data directory
    if Path("training").exists():
        data_dir = "data_generation/processed"
    else:
        data_dir = "../data_generation/processed"
    
    if not Path(data_dir).exists():
        print(f"Error: Data directory {data_dir} not found!")
        print("Please run the data generation pipeline first.")
        return
    
    # Create data loader with 8 balls sliding window
    print("Loading data with 8 balls context...")
    data_loader = CricketDataLoader(
        data_dir=data_dir,
        batch_size=config['batch_size'],
        max_history_length=config['max_history_length'],
        train_split=0.8,
        use_sliding_window=config['use_sliding_window'],
        context_window_size=config['context_window_size']
    )
    
    print(f"\n=== 8 BALLS CONTEXT CONFIGURATION ===")
    print(f"Context window: {config['context_window_size']} balls")
    print(f"Training samples: {len(data_loader.train_dataset):,}")
    print(f"Validation samples: {len(data_loader.val_dataset):,}")
    print(f"Training batches: {len(data_loader.train_loader):,}")
    print(f"Benefits: Immediate patterns, maximum training data")
    print("="*50)
    
    # Update config with actual vocabulary size
    config['vocab_size'] = len(data_loader.vocabulary)
    
    # Create model
    print("Creating model for 8 balls context...")
    model = create_cricket_transformer(config)
    
    # Create trainer with specific save directory
    trainer = CricketTrainer(model, data_loader, config, save_dir=save_dir)
    
    # Start training
    print(f"\nStarting training with 8 balls context...")
    print(f"Model checkpoints will be saved in: {save_dir}/")
    
    start_time = time.time()
    
    trainer.train(
        num_epochs=config['num_epochs'],
        teacher_forcing_epochs=config['teacher_forcing_epochs'],
        save_every=config['save_every'],
        validate_every=config['validate_every']
    )
    
    training_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED - 8 BALLS CONTEXT")
    print("="*80)
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Total training samples used: {len(data_loader.train_dataset):,}")
    print(f"Model saved in: {save_dir}/")
    print(f"Best model: {save_dir}/best_model.pt")
    print("\nModel specializes in:")
    print("✓ Immediate ball-by-ball patterns")
    print("✓ Recent bowling/batting dynamics")
    print("✓ Short-term tactical decisions")
    print("✓ Over-by-over momentum shifts")
    print("="*80)

if __name__ == "__main__":
    main()