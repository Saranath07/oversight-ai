"""
Cricket Transformer Test Script
==============================

Test the cricket transformer model with sample data to validate architecture
and demonstrate prediction capabilities.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cricket_transformer import create_cricket_transformer
from data_generation.data_loader import CricketDataLoader, create_sample_batch, decode_predictions

def test_model_architecture():
    """Test model creation and basic forward pass"""
    
    print("=" * 60)
    print("CRICKET TRANSFORMER ARCHITECTURE TEST")
    print("=" * 60)
    
    # Test configuration
    config = {
        'ball_vector_dim': 18,
        'context_dim': 20,
        'vocab_size': 24,
        'd_model': 256,  # Smaller for testing
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_target_length': 15
    }
    
    # Create model
    print("Creating model...")
    model = create_cricket_transformer(config)
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    model.to(device)
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Test forward pass with random data
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 20
    tgt_len = 8
    
    # Create random test data
    histories = torch.randn(batch_size, seq_len, 18).to(device)
    contexts = torch.randn(batch_size, 20).to(device)
    target_inputs = torch.randint(1, 24, (batch_size, tgt_len)).to(device)
    history_mask = torch.ones(batch_size, seq_len).to(device)
    target_mask = torch.ones(batch_size, tgt_len).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(histories, contexts, target_inputs, history_mask, target_mask)
    
    print(f"✓ Forward pass successful!")
    print(f"  Input shapes:")
    print(f"    - Histories: {histories.shape}")
    print(f"    - Contexts: {contexts.shape}")
    print(f"    - Target inputs: {target_inputs.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Test generation
    print("\nTesting generation...")
    with torch.no_grad():
        generated = model.generate(
            histories=histories,
            contexts=contexts,
            start_token_id=1,
            end_token_id=2,
            max_length=10,
            history_mask=history_mask,
            temperature=1.0
        )
    
    print(f"✓ Generation successful!")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample generated sequence: {generated[0].cpu().numpy()}")
    
    return model, device

def test_with_real_data():
    """Test model with real cricket data"""
    
    print("\n" + "=" * 60)
    print("REAL DATA TEST")
    print("=" * 60)
    
    data_dir = "data_generation/processed"
    
    if not Path(data_dir).exists():
        print(f"Error: Data directory {data_dir} not found!")
        print("Please run the data generation pipeline first.")
        return None
    
    # Load sample data
    print("Loading sample data...")
    batch, vocabulary, idx_to_token = create_sample_batch(data_dir, num_samples=3)
    
    print(f"✓ Loaded sample batch:")
    print(f"  - Batch size: {batch['histories'].shape[0]}")
    print(f"  - History shape: {batch['histories'].shape}")
    print(f"  - Context shape: {batch['contexts'].shape}")
    print(f"  - Target input shape: {batch['target_inputs'].shape}")
    print(f"  - Vocabulary size: {len(vocabulary)}")
    
    # Create model with correct vocabulary size
    config = {
        'ball_vector_dim': 18,
        'context_dim': 20,
        'vocab_size': len(vocabulary),
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1
    }
    
    model = create_cricket_transformer(config)
    
    # Move to device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            if device.type == 'mps' and batch[key].dtype == torch.float64:
                batch[key] = batch[key].float()
            batch[key] = batch[key].to(device)
    
    print(f"\nTesting with real data on {device}...")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            logits = model(
                histories=batch['histories'],
                contexts=batch['contexts'],
                target_inputs=batch['target_inputs'],
                history_mask=batch['history_mask'],
                target_mask=batch['target_mask']
            )
            
            print(f"✓ Forward pass successful!")
            print(f"  Output logits shape: {logits.shape}")
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            decoded_predictions = decode_predictions(predictions, idx_to_token)
            
            print(f"\nPrediction Results:")
            for i in range(min(3, len(decoded_predictions))):
                print(f"  Sample {i+1}:")
                print(f"    Target tokens: {batch['target_tokens'][i]}")
                print(f"    Predicted: {decoded_predictions[i]}")
                
                # Calculate simple accuracy
                target = batch['target_tokens'][i]
                pred = decoded_predictions[i]
                min_len = min(len(target), len(pred))
                matches = sum(1 for j in range(min_len) if target[j] == pred[j])
                accuracy = matches / max(len(target), len(pred)) if max(len(target), len(pred)) > 0 else 0
                print(f"    Token accuracy: {accuracy:.2f}")
            
        except Exception as e:
            print(f"✗ Error during forward pass: {e}")
            return None
    
    # Test generation
    print(f"\nTesting generation...")
    try:
        with torch.no_grad():
            generated = model.generate(
                histories=batch['histories'],
                contexts=batch['contexts'],
                start_token_id=vocabulary['<START>'],
                end_token_id=vocabulary['<END>'],
                max_length=12,
                history_mask=batch['history_mask'],
                temperature=0.8
            )
            
            decoded_generated = decode_predictions(generated, idx_to_token)
            
            print(f"✓ Generation successful!")
            print(f"Generated sequences:")
            for i, seq in enumerate(decoded_generated[:3]):
                print(f"  {i+1}. {seq}")
                
    except Exception as e:
        print(f"✗ Error during generation: {e}")
    
    return model, batch, vocabulary, idx_to_token

def test_training_step():
    """Test a single training step"""
    
    print("\n" + "=" * 60)
    print("TRAINING STEP TEST")
    print("=" * 60)
    
    data_dir = "../data_generation/processed"
    
    if not Path(data_dir).exists():
        print(f"Error: Data directory {data_dir} not found!")
        return
    
    # Create data loader
    print("Creating data loader...")
    data_loader = CricketDataLoader(
        data_dir=data_dir,
        batch_size=4,
        max_history_length=100,
        train_split=0.8
    )
    
    # Create model
    config = {
        'ball_vector_dim': 18,
        'context_dim': 20,
        'vocab_size': len(data_loader.vocabulary),
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 1024,
        'dropout': 0.1
    }
    
    model = create_cricket_transformer(config)
    
    # Setup training components
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    print(f"Testing training step on {device}...")
    
    # Get a batch
    for batch in data_loader.train_loader:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                if device.type == 'mps' and batch[key].dtype == torch.float64:
                    batch[key] = batch[key].float()
                batch[key] = batch[key].to(device)
        
        try:
            # Training step
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(
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
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"✓ Training step successful!")
            print(f"  Batch size: {batch['histories'].shape[0]}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Perplexity: {np.exp(loss.item()):.2f}")
            
            break
            
        except Exception as e:
            print(f"✗ Error during training step: {e}")
            break

def main():
    """Run all tests"""
    
    print("CRICKET TRANSFORMER MODEL TESTING")
    print("=" * 60)
    
    # Test 1: Model architecture
    try:
        model, device = test_model_architecture()
        print("✓ Architecture test passed!")
    except Exception as e:
        print(f"✗ Architecture test failed: {e}")
        return
    
    # Test 2: Real data
    try:
        result = test_with_real_data()
        if result is not None:
            print("✓ Real data test passed!")
        else:
            print("✗ Real data test failed!")
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
    
    # Test 3: Training step
    try:
        test_training_step()
        print("✓ Training step test passed!")
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\nThe model is ready for training!")
    print("Run 'python trainer.py' to start training.")

if __name__ == "__main__":
    main()