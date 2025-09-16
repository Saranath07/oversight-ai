"""
Cricket Transformer Training - 16 Balls Context
===============================================

Training script optimized for 16 balls context window. This approach balances
recent patterns with broader tactical context (approximately 2.6 overs).

Key benefits:
- Balanced context (16 balls ≈ 2.6 overs)  
- Good training samples (1.5M+ samples)
- Captures recent trends and tactical shifts
- Includes bowling changes and batting adjustments
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
    """Main training function for 16 balls context"""
    
    print("="*80)
    print("CRICKET TRANSFORMER - 16 BALLS CONTEXT TRAINING")
    print("="*80)
    print("Context: 16 balls (≈2.6 overs)")
    print("Focus: Balanced recent patterns with tactical context")
    print("Expected samples: ~1.5M training samples")
    print("="*80)
    # SMALL Configuration optimized for 16 balls context - Fast experimentation
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
        
        # 16 BALLS CONTEXT CONFIGURATION
        'use_sliding_window': True,
        'context_window_size': 16,  # 16 balls context
        'max_history_length': 128,  # Legacy fallback
        
        # Training schedule - SHORTER for experimentation
        'num_epochs': 20,        # Reduced from 100 to 20
        'teacher_forcing_epochs': 10,  # Reduced from 50 to 10
        'save_every': 5,
        'validate_every': 1
    }
    
    
    # Create save directory specific to 16 balls context
    save_dir = "checkpoints_16_balls_context"
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
    
    # Create data loader with 16 balls sliding window
    print("Loading data with 16 balls context...")
    data_loader = CricketDataLoader(
        data_dir=data_dir,
        batch_size=config['batch_size'],
        max_history_length=config['max_history_length'],
        train_split=0.8,
        use_sliding_window=config['use_sliding_window'],
        context_window_size=config['context_window_size']
    )
    
    print(f"\n=== 16 BALLS CONTEXT CONFIGURATION ===")
    print(f"Context window: {config['context_window_size']} balls")
    print(f"Training samples: {len(data_loader.train_dataset):,}")
    print(f"Validation samples: {len(data_loader.val_dataset):,}")
    print(f"Training batches: {len(data_loader.train_loader):,}")
    print(f"Benefits: Balanced context, tactical patterns")
    print("="*50)
    
    # Update config with actual vocabulary size
    config['vocab_size'] = len(data_loader.vocabulary)
    
    # Create model
    print("Creating model for 16 balls context...")
    model = create_cricket_transformer(config)
    
    # Create trainer with specific save directory
    trainer = CricketTrainer(model, data_loader, config, save_dir=save_dir)
    
    # Start training
    print(f"\nStarting training with 16 balls context...")
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
    print("TRAINING COMPLETED - 16 BALLS CONTEXT")
    print("="*80)
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Total training samples used: {len(data_loader.train_dataset):,}")
    print(f"Model saved in: {save_dir}/")
    print(f"Best model: {save_dir}/best_model.pt")
    print("\nModel specializes in:")
    print("✓ Recent ball-by-ball patterns")
    print("✓ Bowling/batting tactical shifts")
    print("✓ Medium-term momentum analysis")
    print("✓ Over-to-over strategy changes")
    print("✓ Balanced immediate + tactical context")
    print("="*80)

if __name__ == "__main__":
    main()