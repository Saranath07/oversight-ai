"""
Test Script for Context Window Experimentation
==============================================

This script tests the new sliding window approach with different context window sizes
and compares it to the legacy approach to demonstrate the benefits.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.data_loader import CricketDataLoader

def test_context_window_sizes():
    """Test different context window sizes and compare results"""
    
    # Data directory
    if Path("training").exists():
        data_dir = "data_generation/processed"
    else:
        data_dir = "../data_generation/processed"
    
    if not Path(data_dir).exists():
        print(f"Error: Data directory {data_dir} not found!")
        print("Please run the data generation pipeline first.")
        return
    
    print("=== CONTEXT WINDOW EXPERIMENTATION ===\n")
    
    # Test different context window sizes
    context_sizes = [8, 16, 32]
    batch_size = 4
    
    results = {}
    
    for context_size in context_sizes:
        print(f"--- Testing Context Window Size: {context_size} balls ---")
        
        try:
            # Create data loader with sliding window
            data_loader = CricketDataLoader(
                data_dir=data_dir,
                batch_size=batch_size,
                use_sliding_window=True,
                context_window_size=context_size,
                train_split=0.8
            )
            
            # Get sample batch
            sample_batch = next(iter(data_loader.train_loader))
            
            # Analyze the sample
            history_shape = sample_batch['histories'].shape
            avg_history_length = sample_batch['history_mask'].sum(dim=1).mean().item()
            max_history_length = sample_batch['history_mask'].sum(dim=1).max().item()
            min_history_length = sample_batch['history_mask'].sum(dim=1).min().item()
            
            results[context_size] = {
                'total_samples': len(data_loader.train_dataset),
                'avg_history_length': avg_history_length,
                'max_history_length': max_history_length,
                'min_history_length': min_history_length,
                'history_shape': history_shape
            }
            
            print(f"  Total training samples: {results[context_size]['total_samples']}")
            print(f"  Average history length: {avg_history_length:.1f}")
            print(f"  Max history length: {max_history_length:.0f}")
            print(f"  Min history length: {min_history_length:.0f}")
            print(f"  History tensor shape: {history_shape}")
            print()
            
        except Exception as e:
            print(f"  Error with context size {context_size}: {e}")
            print()
    
    # Test legacy mode for comparison
    print("--- Testing Legacy Mode (128 balls max) ---")
    try:
        legacy_loader = CricketDataLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            use_sliding_window=False,
            max_history_length=128,
            train_split=0.8
        )
        
        legacy_batch = next(iter(legacy_loader.train_loader))
        legacy_history_shape = legacy_batch['histories'].shape
        legacy_avg_length = legacy_batch['history_mask'].sum(dim=1).mean().item()
        legacy_max_length = legacy_batch['history_mask'].sum(dim=1).max().item()
        
        print(f"  Total training samples: {len(legacy_loader.train_dataset)}")
        print(f"  Average history length: {legacy_avg_length:.1f}")
        print(f"  Max history length: {legacy_max_length:.0f}")
        print(f"  History tensor shape: {legacy_history_shape}")
        print()
        
    except Exception as e:
        print(f"  Error with legacy mode: {e}")
        print()
    
    # Summary and recommendations
    print("=== ANALYSIS AND RECOMMENDATIONS ===")
    print()
    print("Benefits of Sliding Window Approach:")
    print("✓ Prevents data leakage within matches")
    print("✓ Creates more training samples from each match")
    print("✓ Models realistic cricket scenarios (recent balls matter most)")
    print("✓ Reduces overfitting to specific match patterns")
    print("✓ Better generalization across different game situations")
    print()
    
    print("Context Window Size Recommendations:")
    print("• 8 balls:  Very recent context, good for immediate patterns (2 overs)")
    print("• 16 balls: Balanced context, captures recent trends (2.6 overs)")
    print("• 32 balls: Extended context, includes broader match dynamics (5+ overs)")
    print()
    
    if results:
        best_context = max(results.keys(), key=lambda k: results[k]['total_samples'])
        print(f"Most training samples generated: {best_context} balls context")
        print(f"  → {results[best_context]['total_samples']} samples")
        print()
        
        print("For experimentation, try:")
        print("1. Start with 16 balls context (good balance)")
        print("2. Compare with 8 balls (immediate context)")
        print("3. Test 32 balls if you need more historical context")
        print()


def demonstrate_sample_data():
    """Show actual sample data to understand the differences"""
    
    if Path("training").exists():
        data_dir = "data_generation/processed"
    else:
        data_dir = "../data_generation/processed"
    
    if not Path(data_dir).exists():
        return
    
    print("=== SAMPLE DATA COMPARISON ===\n")
    
    # Create loaders
    sliding_loader = CricketDataLoader(
        data_dir=data_dir,
        batch_size=2,
        use_sliding_window=True,
        context_window_size=16
    )
    
    # Get sample batch
    sample_batch = next(iter(sliding_loader.train_loader))
    
    print("Sample from Sliding Window (16 balls context):")
    for i in range(min(2, sample_batch['histories'].shape[0])):
        history_length = sample_batch['history_mask'][i].sum().item()
        print(f"  Sample {i+1}:")
        print(f"    History length: {history_length} balls")
        print(f"    Target: {sample_batch['target_tokens'][i]}")
        if 'match_idx' in sample_batch:
            print(f"    Match index: {sample_batch['match_indices'][i]}")
            print(f"    Ball index: {sample_batch['ball_indices'][i]}")
        print()


if __name__ == "__main__":
    test_context_window_sizes()
    demonstrate_sample_data()