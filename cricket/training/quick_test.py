#!/usr/bin/env python3
"""
Quick Test Script for Cricket Transformer
=========================================

Simple test to verify the model works with your actual data.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_with_actual_data():
    """Test model with the actual cricket data"""
    
    print("🏏 Cricket Transformer - Quick Test")
    print("=" * 50)
    
    # Check if data exists
    data_dir = "data_generation/processed"
    if not Path(data_dir).exists():
        print(f"❌ Data directory {data_dir} not found!")
        print("Please run the data generation pipeline first:")
        print("  python data_generation/generate_data.py")
        return False
    
    try:
        from training.cricket_transformer import create_cricket_transformer
        from data_generation.data_loader import create_sample_batch, decode_predictions
        
        # Load sample data
        print("📊 Loading sample data...")
        batch, vocabulary, idx_to_token = create_sample_batch(data_dir, num_samples=2)
        
        print(f"✓ Data loaded successfully!")
        print(f"  - Vocabulary size: {len(vocabulary)}")
        print(f"  - Sample batch size: {batch['histories'].shape[0]}")
        print(f"  - History shape: {batch['histories'].shape}")
        
        # Create model
        print("\n🏗️ Creating model...")
        config = {
            'ball_vector_dim': 18,
            'context_dim': 20,
            'vocab_size': len(vocabulary),
            'd_model': 256,  # Smaller for quick test
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 1024,
            'dropout': 0.1
        }
        
        model = create_cricket_transformer(config)
        
        # Detect device
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✓ Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("✓ Using CUDA GPU")
        else:
            device = torch.device('cpu')
            print("✓ Using CPU")
        
        model.to(device)
        print(f"✓ Model created with {model.count_parameters():,} parameters")
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                if device.type == 'mps' and batch[key].dtype == torch.float64:
                    batch[key] = batch[key].float()
                batch[key] = batch[key].to(device)
        
        # Test forward pass
        print("\n🧪 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            logits = model(
                histories=batch['histories'],
                contexts=batch['contexts'],
                target_inputs=batch['target_inputs'],
                history_mask=batch['history_mask'],
                target_mask=batch['target_mask']
            )
            
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {logits.shape}")
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            decoded_predictions = decode_predictions(predictions, idx_to_token)
            
            print(f"\n🎯 Sample Predictions:")
            for i in range(min(2, len(decoded_predictions))):
                target = batch['target_tokens'][i]
                pred = decoded_predictions[i]
                print(f"  Sample {i+1}:")
                print(f"    Target: {target}")
                print(f"    Predicted: {pred}")
        
        # Test generation
        print(f"\n🎲 Testing generation...")
        with torch.no_grad():
            generated = model.generate(
                histories=batch['histories'],
                contexts=batch['contexts'],
                start_token_id=vocabulary['<START>'],
                end_token_id=vocabulary['<END>'],
                max_length=10,
                history_mask=batch['history_mask'],
                temperature=0.8
            )
            
            decoded_generated = decode_predictions(generated, idx_to_token)
            
            print(f"✓ Generation successful!")
            print(f"Generated sequences:")
            for i, seq in enumerate(decoded_generated):
                print(f"  {i+1}. {seq}")
        
        print(f"\n🎉 All tests passed! Model is ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_actual_data()
    if success:
        print(f"\n🚀 Ready to train! Run:")
        print(f"  python training/trainer.py")
        print(f"  # or")
        print(f"  python training/run.py train --config small")
    else:
        print(f"\n💥 Fix the issues above before training.")