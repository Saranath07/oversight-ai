#!/usr/bin/env python3
"""
Cricket Transformer Continue Training
====================================

Script to continue training from the best model checkpoint with 3 epochs of autonomous training.
"""

import argparse
import sys
from pathlib import Path

def continue_training(config_name='small', epochs=3):
    """Continue training from the best model checkpoint with autonomous training"""
    print(f"🚀 Continuing Cricket Transformer Training...")
    print(f"Configuration: {config_name}")
    print(f"Autonomous training for {epochs} epochs")
    print("=" * 50)
    
    try:
        from trainer import CricketTrainer
        from cricket_transformer import create_cricket_transformer
        from data_generation.data_loader import CricketDataLoader
        from config import get_config
        
        # Load configuration
        config = get_config(config_name)
        
        # Print configuration
        from config import print_config
        print_config(config)
        
        # Check data directory (adjust path based on current working directory)
        if Path("training").exists():
            # Running from root directory
            data_dir = "data_generation/processed"
            checkpoint_dir = "checkpoints"  # Corrected path
        else:
            # Running from training directory
            data_dir = "../data_generation/processed"
            checkpoint_dir = "checkpoints"
            
        if not Path(data_dir).exists():
            print(f"\n❌ Error: Data directory {data_dir} not found!")
            return False
            
        if not Path(checkpoint_dir).exists() or not Path(f"{checkpoint_dir}/best_model.pt").exists():
            print(f"\n❌ Error: Best model checkpoint not found!")
            print(f"Expected at: {checkpoint_dir}/best_model.pt")
            return False
        
        # Create data loader
        print(f"\n📊 Loading data from {data_dir}...")
        data_loader = CricketDataLoader(
            data_dir=data_dir,
            batch_size=config['batch_size'],
            max_history_length=config['max_history_length'],
            train_split=0.8
        )
        
        # Update config with actual vocabulary size
        config['vocab_size'] = len(data_loader.vocabulary)
        
        # Create model
        print(f"\n🏗️ Creating model...")
        model = create_cricket_transformer(config)
        
        # Create trainer
        trainer = CricketTrainer(model, data_loader, config, save_dir=checkpoint_dir)
        
        # Load the best model checkpoint
        print(f"\n📥 Loading best model checkpoint...")
        trainer.load_checkpoint('best_model.pt')
        
        # Continue training for the specified epochs
        print(f"\n🧠 Continuing autonomous training for {epochs} epochs...")
        trainer.train(
            num_epochs=trainer.epoch + epochs,  # Current epoch + additional epochs
            teacher_forcing_epochs=0,  # Force autonomous training
            save_every=1,
            validate_every=1
        )
        
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Cricket Transformer Continue Training')
    parser.add_argument('--config', default='small', 
                      choices=['small', 'base', 'large', 'fast'],
                      help='Configuration to use for training')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of epochs to train')
    
    args = parser.parse_args()
    
    print("🏏 Cricket Transformer Continue Training")
    print("=" * 50)
    
    success = continue_training(args.config, args.epochs)
    
    if success:
        print(f"\n🎉 Training completed successfully!")
    else:
        print(f"\n💥 Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()