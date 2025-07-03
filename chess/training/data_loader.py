"""
Data loader for chess training data.
Handles loading and batching of processed PGN data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import gym_chess
import gym
import chess


class ChessDataset(Dataset):
    """Dataset for chess training sequences."""
    
    def __init__(self, 
                 data_file: str,
                 max_seq_len: int = 512,
                 include_legal_moves: bool = True):
        """
        Initialize chess dataset.
        
        Args:
            data_file: Path to processed JSON data file
            max_seq_len: Maximum sequence length
            include_legal_moves: Whether to compute legal moves masks
        """
        self.max_seq_len = max_seq_len
        self.include_legal_moves = include_legal_moves
        
        # Load data
        print(f"Loading data from {data_file}...")
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.training_examples = data['training_examples']
        self.vocabulary = data['vocabulary']
        self.vocab_size = data['vocab_size']
        self.policy_size = data['policy_size']
        
        # Create reverse vocabulary mapping
        self.idx_to_token = {v: k for k, v in self.vocabulary.items()}
        
        # Initialize gym environment for legal moves computation
        if self.include_legal_moves:
            self.env = gym.make('ChessAlphaZero-v0')
            self.env.reset()
        
        print(f"Loaded {len(self.training_examples)} training examples")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Policy size: {self.policy_size}")
    
    def __len__(self) -> int:
        return len(self.training_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.training_examples[idx]
        
        # Get input sequence and convert to token indices
        input_sequence = example['input_sequence']
        input_ids = [self.vocabulary.get(token, self.vocabulary['<PAD>']) 
                    for token in input_sequence]
        
        # Truncate or pad sequence
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]  # Keep most recent moves
        
        # Pad sequence
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_len:
            input_ids.append(self.vocabulary['<PAD>'])
            attention_mask.append(0)
        
        # Target
        target_policy_index = example['target_policy_index']
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'target_policy_index': torch.tensor(target_policy_index, dtype=torch.long),
            'target_move': example['target_move']
        }
        
        # Compute legal moves mask if requested
        if self.include_legal_moves:
            legal_moves_mask = self._compute_legal_moves_mask(input_sequence[:-1])  # Exclude BOS
            result['legal_moves_mask'] = torch.tensor(legal_moves_mask, dtype=torch.float32)
        
        return result
    
    def _compute_legal_moves_mask(self, move_sequence: List[str]) -> np.ndarray:
        """
        Compute legal moves mask for a given position.
        
        Args:
            move_sequence: List of UCI moves (without BOS token)
            
        Returns:
            Binary mask of shape [4672] where 1 = legal move, 0 = illegal move
        """
        try:
            # Create board and play moves
            board = chess.Board()
            
            for move_str in move_sequence:
                if move_str not in ['<BOS>', '<EOS>', '<PAD>']:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            board.push(move)
                        else:
                            # Invalid move sequence, return empty mask
                            return np.zeros(self.policy_size, dtype=np.float32)
                    except:
                        # Invalid move, return empty mask
                        return np.zeros(self.policy_size, dtype=np.float32)
            
            # Get legal moves mask
            legal_mask = np.zeros(self.policy_size, dtype=np.float32)
            
            for move in board.legal_moves:
                try:
                    policy_index = self.env.encode(move)
                    if 0 <= policy_index < self.policy_size:
                        legal_mask[policy_index] = 1.0
                except:
                    continue
            
            return legal_mask
            
        except Exception as e:
            # If anything goes wrong, return empty mask
            return np.zeros(self.policy_size, dtype=np.float32)


def create_data_loaders(data_file: str,
                       batch_size: int = 32,
                       max_seq_len: int = 512,
                       train_split: float = 0.9,
                       include_legal_moves: bool = True,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        data_file: Path to processed JSON data file
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        train_split: Fraction of data to use for training
        include_legal_moves: Whether to compute legal moves masks
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = ChessDataset(
        data_file=data_file,
        max_seq_len=max_seq_len,
        include_legal_moves=include_legal_moves
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing Chess Data Loader...")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_file="chess/processed_data/chess_training_sample.json",
        batch_size=4,
        max_seq_len=128,
        include_legal_moves=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    # Test a batch
    for batch in train_loader:
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Target policy indices shape: {batch['target_policy_index'].shape}")
        
        if 'legal_moves_mask' in batch:
            print(f"Legal moves mask shape: {batch['legal_moves_mask'].shape}")
            print(f"Average legal moves per position: {batch['legal_moves_mask'].sum(dim=1).mean():.1f}")
        
        print(f"Sample target moves: {batch['target_move'][:2]}")
        break
    
    print("Data loader test completed successfully!")