#!/usr/bin/env python3
"""
Cricket Transformer Runner
=========================

Simple script to run different training and testing scenarios.
"""

import argparse
import sys
from pathlib import Path

def run_test():
    """Run model tests"""
    print("🧪 Running Cricket Transformer Tests...")
    print("=" * 50)
    
    try:
        from test_model import main as test_main
        test_main()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

def run_training(config_name='base', epochs=None):
    """Run training with specified configuration"""
    print(f"🚀 Starting Cricket Transformer Training...")
    print(f"Configuration: {config_name}")
    print("=" * 50)
    
    try:
        from trainer import CricketTrainer
        from cricket_transformer import create_cricket_transformer
        from data_generation.data_loader import CricketDataLoader
        from config import get_config
        
        # Load configuration
        config = get_config(config_name)
        if epochs:
            config['num_epochs'] = epochs
            config['teacher_forcing_epochs'] = epochs // 2
        
        # Print configuration
        from config import print_config, estimate_memory
        print_config(config)
        
        memory_info = estimate_memory(config)
        print(f"\nEstimated Requirements:")
        print(f"  Parameters: {memory_info['parameters']:,}")
        print(f"  GPU Memory: {memory_info['estimated_gpu_memory_mb']:.1f} MB")
        
        # Check data directory (adjust path based on current working directory)
        if Path("training").exists():
            # Running from root directory
            data_dir = "data_generation/processed"
        else:
            # Running from training directory
            data_dir = "../data_generation/processed"
            
        if not Path(data_dir).exists():
            print(f"\n❌ Error: Data directory {data_dir} not found!")
            print("Please run the data generation pipeline first:")
            if Path("training").exists():
                print("  python data_generation/generate_data.py")
            else:
                print("  cd ../data_generation && python generate_data.py")
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
        trainer = CricketTrainer(model, data_loader, config, save_dir="checkpoints-8-balls-context")
        
        # Start training
        trainer.train(
            num_epochs=config['num_epochs'],
            teacher_forcing_epochs=config['teacher_forcing_epochs'],
            save_every=config['save_every'],
            validate_every=config['validate_every']
        )
        
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_configs():
    """Show available configurations"""
    print("📋 Available Configurations:")
    print("=" * 50)
    
    from config import CONFIGS, estimate_memory
    
    for name, config in CONFIGS.items():
        memory_info = estimate_memory(config)
        print(f"\n{name.upper()}:")
        print(f"  Model: {config['d_model']}d, {config['num_encoder_layers']}+{config['num_decoder_layers']} layers")
        print(f"  Parameters: ~{memory_info['parameters']:,}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Epochs: {config['num_epochs']} (TF: {config['teacher_forcing_epochs']})")
        print(f"  Memory: ~{memory_info['estimated_gpu_memory_mb']:.0f} MB")

def main():
    parser = argparse.ArgumentParser(description='Cricket Transformer Runner')
    parser.add_argument('command', choices=['test', 'train', 'configs'], 
                       help='Command to run')
    parser.add_argument('--config', default='base', 
                       choices=['small', 'base', 'large', 'fast'],
                       help='Configuration to use for training')
    parser.add_argument('--epochs', type=int, 
                       help='Number of epochs to train (overrides config)')
    
    args = parser.parse_args()
    
    print("🏏 Cricket Transformer Runner")
    print("=" * 50)
    
    if args.command == 'test':
        success = run_test()
    elif args.command == 'train':
        success = run_training(args.config, args.epochs)
    elif args.command == 'configs':
        show_configs()
        success = True
    else:
        print(f"❌ Unknown command: {args.command}")
        success = False
    
    if success:
        print(f"\n🎉 Command '{args.command}' completed successfully!")
    else:
        print(f"\n💥 Command '{args.command}' failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()