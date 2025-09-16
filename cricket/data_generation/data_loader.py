"""
OverSight AI Data Loader
========================

This module provides utilities for loading and batching the processed cricket data
for training the transformer model.
"""

import pickle
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
from pathlib import Path

class CricketDataset(Dataset):
    """PyTorch Dataset for cricket ball-by-ball prediction"""
    
    def __init__(self, data_dir: str, max_history_length: int = 500,
                 use_sliding_window: bool = True, context_window_size: int = 16):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing processed data files
            max_history_length: Maximum length of match history sequence (for legacy mode)
            use_sliding_window: If True, create multiple samples per match using sliding windows
            context_window_size: Number of recent balls to use as context (8, 16, 32, etc.)
        """
        self.data_dir = Path(data_dir)
        self.max_history_length = max_history_length
        self.use_sliding_window = use_sliding_window
        self.context_window_size = context_window_size
        
        # Load processed data
        self.raw_match_histories = self._load_pickle('match_histories.pkl')
        self.raw_contexts = self._load_pickle('contexts.pkl')
        self.raw_targets = self._load_pickle('targets.pkl')
        
        # Load vocabulary and metadata
        with open(self.data_dir / 'vocabulary.json', 'r') as f:
            self.vocabulary = json.load(f)
        
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Create reverse vocabulary for decoding
        self.idx_to_token = {v: k for k, v in self.vocabulary.items()}
        
        # Process data based on mode
        if use_sliding_window:
            self.training_samples = self._create_sliding_window_samples()
            print(f"Created {len(self.training_samples)} training samples using sliding windows")
            print(f"Context window size: {context_window_size} balls")
        else:
            # Legacy mode: filter long sequences
            self.match_histories = self.raw_match_histories
            self.contexts = self.raw_contexts
            self.targets = self.raw_targets
            self._filter_sequences()
            print(f"Using legacy mode with {len(self.match_histories)} sequences")
            print(f"Max history length: {max_history_length}")
        
        print(f"Ball vector dimension: {self.metadata['ball_vector_dim']}")
        print(f"Context vector dimension: {self.metadata['context_vector_dim']}")
    
    def _load_pickle(self, filename: str):
        """Load pickle file"""
        with open(self.data_dir / filename, 'rb') as f:
            return pickle.load(f)
    
    def _filter_sequences(self):
        """Filter sequences that are too long"""
        filtered_histories = []
        filtered_contexts = []
        filtered_targets = []
        
        for i, history in enumerate(self.match_histories):
            if len(history) <= self.max_history_length:
                filtered_histories.append(history)
                filtered_contexts.append(self.contexts[i])
                filtered_targets.append(self.targets[i])
        
        self.match_histories = filtered_histories
        self.contexts = filtered_contexts
        self.targets = filtered_targets
        
        print(f"Filtered to {len(self.match_histories)} sequences (max length: {self.max_history_length})")
    
    def _create_sliding_window_samples(self):
        """Create training samples using sliding context windows"""
        samples = []
        min_context_length = max(4, self.context_window_size // 2)  # At least half the window size
        
        for match_idx, match_history in enumerate(self.raw_match_histories):
            match_context = self.raw_contexts[match_idx]
            match_target = self.raw_targets[match_idx]
            
            # Create multiple samples from this match using sliding windows
            for ball_idx in range(len(match_history)):
                
                # Determine context window bounds
                context_start = max(0, ball_idx - self.context_window_size + 1)
                context_end = ball_idx + 1
                
                # Get context window (recent balls only)
                context_history = match_history[context_start:context_end]
                
                # Skip if context is too short
                if len(context_history) < min_context_length:
                    continue
                
                # Skip samples that would predict beyond the match
                # (to avoid temporal leakage within the same match)
                if ball_idx >= len(match_history) - 1:
                    continue
                
                # Create sample
                sample = {
                    'history': context_history,
                    'context': match_context,  # Match-level context (venue, teams, etc.)
                    'target': match_target,
                    'match_idx': match_idx,
                    'ball_idx': ball_idx
                }
                
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        if self.use_sliding_window:
            return len(self.training_samples)
        else:
            return len(self.match_histories)
    
    def __getitem__(self, idx):
        """Get a single training example"""
        
        if self.use_sliding_window:
            # Sliding window mode
            sample = self.training_samples[idx]
            
            # Convert match history to tensor
            history = torch.tensor(np.array(sample['history']), dtype=torch.float32)
            
            # Convert context to tensor
            context = torch.tensor(sample['context'], dtype=torch.float32)
            
            # Convert target tokens to indices
            target_tokens = sample['target']
            target_indices = [self.vocabulary.get(token, self.vocabulary['<PAD>']) for token in target_tokens]
            
            # Add start and end tokens
            target_input = [self.vocabulary['<START>']] + target_indices
            target_output = target_indices + [self.vocabulary['<END>']]
            
            return {
                'history': history,
                'context': context,
                'target_input': torch.tensor(target_input, dtype=torch.long),
                'target_output': torch.tensor(target_output, dtype=torch.long),
                'target_tokens': target_tokens,  # Keep original tokens for debugging
                'match_idx': sample['match_idx'],
                'ball_idx': sample['ball_idx']
            }
        else:
            # Legacy mode
            # Convert match history to tensor
            history = torch.tensor(np.array(self.match_histories[idx]), dtype=torch.float32)
            
            # Convert context to tensor
            context = torch.tensor(self.contexts[idx], dtype=torch.float32)
            
            # Convert target tokens to indices
            target_tokens = self.targets[idx]
            target_indices = [self.vocabulary.get(token, self.vocabulary['<PAD>']) for token in target_tokens]
            
            # Add start and end tokens
            target_input = [self.vocabulary['<START>']] + target_indices
            target_output = target_indices + [self.vocabulary['<END>']]
            
            return {
                'history': history,
                'context': context,
                'target_input': torch.tensor(target_input, dtype=torch.long),
                'target_output': torch.tensor(target_output, dtype=torch.long),
                'target_tokens': target_tokens  # Keep original tokens for debugging
            }

def collate_fn(batch):
    """Custom collate function for batching variable-length sequences"""
    
    histories = [item['history'] for item in batch]
    contexts = torch.stack([item['context'] for item in batch])
    target_inputs = [item['target_input'] for item in batch]
    target_outputs = [item['target_output'] for item in batch]
    target_tokens = [item['target_tokens'] for item in batch]
    
    # Pad histories to same length
    histories_padded = pad_sequence(histories, batch_first=True, padding_value=0.0)
    
    # Pad targets to same length
    target_inputs_padded = pad_sequence(target_inputs, batch_first=True, 
                                       padding_value=0)  # PAD token index
    target_outputs_padded = pad_sequence(target_outputs, batch_first=True, 
                                        padding_value=0)  # PAD token index
    
    # Create attention masks
    history_mask = (histories_padded.sum(dim=-1) != 0).float()  # Non-zero vectors
    target_mask = (target_inputs_padded != 0).float()  # Non-PAD tokens
    
    return {
        'histories': histories_padded,
        'contexts': contexts,
        'target_inputs': target_inputs_padded,
        'target_outputs': target_outputs_padded,
        'history_mask': history_mask,
        'target_mask': target_mask,
        'target_tokens': target_tokens
    }

class CricketDataLoader:
    """Wrapper class for creating data loaders"""
    
    def __init__(self, data_dir: str, batch_size: int = 32,
                 max_history_length: int = 500, train_split: float = 0.8,
                 use_sliding_window: bool = True, context_window_size: int = 16):
        """
        Initialize data loaders
        
        Args:
            data_dir: Directory containing processed data
            batch_size: Batch size for training
            max_history_length: Maximum sequence length (legacy mode)
            train_split: Fraction of data for training
            use_sliding_window: Use sliding window approach for better training
            context_window_size: Number of recent balls for context (8, 16, 32, etc.)
        """
        
        # Load full dataset
        full_dataset = CricketDataset(
            data_dir,
            max_history_length=max_history_length,
            use_sliding_window=use_sliding_window,
            context_window_size=context_window_size
        )
        
        # Split into train/validation
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for debugging, increase for performance
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Store metadata
        self.vocabulary = full_dataset.vocabulary
        self.idx_to_token = full_dataset.idx_to_token
        self.metadata = full_dataset.metadata
        
        print(f"Created data loaders:")
        print(f"  - Training batches: {len(self.train_loader)}")
        print(f"  - Validation batches: {len(self.val_loader)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Mode: {'Sliding Window' if use_sliding_window else 'Legacy'}")
        if use_sliding_window:
            print(f"  - Context window: {context_window_size} balls")
        else:
            print(f"  - Max history: {max_history_length} balls")

def create_sample_batch(data_dir: str, num_samples: int = 5):
    """Create a sample batch for testing"""
    
    dataset = CricketDataset(data_dir, use_sliding_window=True, context_window_size=16)
    
    # Get first few samples
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
    # Create batch
    batch = collate_fn(samples)
    
    return batch, dataset.vocabulary, dataset.idx_to_token

def decode_predictions(predictions: torch.Tensor, idx_to_token: Dict[int, str]) -> List[List[str]]:
    """Decode model predictions back to tokens"""
    
    batch_size, seq_len = predictions.shape
    decoded_sequences = []
    
    for i in range(batch_size):
        sequence = []
        for j in range(seq_len):
            token_idx = predictions[i, j].item()
            token = idx_to_token.get(token_idx, '<UNK>')
            
            # Stop at end token
            if token == '<END>':
                break
            
            # Skip pad and start tokens
            if token not in ['<PAD>', '<START>']:
                sequence.append(token)
        
        decoded_sequences.append(sequence)
    
    return decoded_sequences

def print_sample_data(data_dir: str, num_samples: int = 3):
    """Print sample data for inspection"""
    
    batch, vocabulary, idx_to_token = create_sample_batch(data_dir, num_samples)
    
    print("=== SAMPLE DATA INSPECTION ===")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Vocabulary: {list(vocabulary.keys())}")
    print()
    
    for i in range(min(num_samples, batch['histories'].shape[0])):
        print(f"--- Sample {i+1} ---")
        print(f"History shape: {batch['histories'][i].shape}")
        print(f"Context shape: {batch['contexts'][i].shape}")
        print(f"Target input: {batch['target_inputs'][i]}")
        print(f"Target output: {batch['target_outputs'][i]}")
        
        # Decode target tokens
        target_tokens = []
        for token_idx in batch['target_outputs'][i]:
            if token_idx.item() == 0:  # PAD token
                break
            token = idx_to_token.get(token_idx.item(), '<UNK>')
            if token != '<END>':
                target_tokens.append(token)
        
        print(f"Target tokens: {target_tokens}")
        print(f"Original tokens: {batch['target_tokens'][i]}")
        print()

if __name__ == "__main__":
    # Test the data loader
    data_dir = "data_generation/processed"
    
    if Path(data_dir).exists():
        print_sample_data(data_dir)
        
        # Create data loaders
        # Test both modes
        print("=== Testing Sliding Window Mode (16 balls context) ===")
        data_loader_sliding = CricketDataLoader(
            data_dir,
            batch_size=4,
            use_sliding_window=True,
            context_window_size=16
        )
        
        print("\n=== Testing Legacy Mode ===")
        data_loader_legacy = CricketDataLoader(
            data_dir,
            batch_size=4,
            use_sliding_window=False,
            max_history_length=128
        )
        
        data_loader = data_loader_sliding  # Use sliding window for main test
        
        # Test a batch
        for batch in data_loader.train_loader:
            print("=== BATCH TEST ===")
            print(f"Histories shape: {batch['histories'].shape}")
            print(f"Contexts shape: {batch['contexts'].shape}")
            print(f"Target inputs shape: {batch['target_inputs'].shape}")
            print(f"Target outputs shape: {batch['target_outputs'].shape}")
            print(f"History mask shape: {batch['history_mask'].shape}")
            print(f"Target mask shape: {batch['target_mask'].shape}")
            break
    else:
        print(f"Processed data directory {data_dir} not found. Run data_processor.py first.")