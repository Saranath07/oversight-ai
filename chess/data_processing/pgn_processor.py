"""
PGN file processor for chess games.
Converts PGN games into sequences suitable for transformer training.
"""

import chess
import chess.pgn
import gym_chess
import gym
import numpy as np
from typing import List, Tuple, Dict, Iterator
import os
from pathlib import Path
import json
from tqdm import tqdm


class ChessPGNProcessor:
    """Process PGN files and convert games to training sequences."""
    
    def __init__(self):
        # Initialize the gym environment for move encoding
        self.env = gym.make('ChessAlphaZero-v0')
        self.env.reset()
        
        # Special tokens
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"
        self.PAD_TOKEN = "<PAD>"
        
        # Build vocabulary of UCI moves
        self.vocab = self._build_vocabulary()
        
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary mapping UCI moves to token indices."""
        vocab = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
        }
        
        # Generate all possible UCI moves by iterating through policy indices
        uci_moves = set()
        for policy_idx in range(4672):
            try:
                move = self.env.decode(policy_idx)
                if move:
                    uci_moves.add(move.uci())
            except:
                continue
        
        # Add UCI moves to vocabulary
        for i, uci_move in enumerate(sorted(uci_moves)):
            vocab[uci_move] = i + 3  # Start after special tokens
            
        return vocab
    
    def uci_to_policy_index(self, uci_move: str) -> int:
        """Convert UCI move to AlphaZero policy index."""
        try:
            move = chess.Move.from_uci(uci_move)
            return self.env.encode(move)
        except:
            return -1  # Invalid move
    
    def policy_index_to_uci(self, policy_index: int) -> str:
        """Convert AlphaZero policy index to UCI move."""
        try:
            move = self.env.decode(policy_index)
            return move.uci() if move else None
        except:
            return None
    
    def process_single_pgn_file(self, pgn_path: str, max_games: int = None) -> List[Dict]:
        """
        Process a single PGN file and extract training sequences.
        
        Returns:
            List of training examples, each containing:
            - input_sequence: List of UCI moves (history)
            - target_move: Next UCI move
            - target_policy_index: AlphaZero policy index for target move
        """
        training_examples = []
        
        with open(pgn_path, 'r') as pgn_file:
            game_count = 0
            
            while True:
                if max_games and game_count >= max_games:
                    break
                    
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                # Extract moves from the game
                moves = []
                board = game.board()
                
                for move in game.mainline_moves():
                    moves.append(move.uci())
                    board.push(move)
                
                # Skip games that are too short
                if len(moves) < 10:
                    continue
                
                # Create training sequences
                # For each position in the game, create a training example
                for i in range(1, len(moves)):
                    input_sequence = [self.BOS_TOKEN] + moves[:i]
                    target_move = moves[i]
                    target_policy_index = self.uci_to_policy_index(target_move)
                    
                    if target_policy_index >= 0:  # Valid move
                        training_examples.append({
                            'input_sequence': input_sequence,
                            'target_move': target_move,
                            'target_policy_index': int(target_policy_index),  # Convert to Python int
                            'game_id': game_count,
                            'move_number': i
                        })
                
                game_count += 1
                
                if game_count % 1000 == 0:
                    print(f"Processed {game_count} games, {len(training_examples)} examples")
        
        return training_examples
    
    def process_multiple_pgn_files(self, pgn_directory: str, 
                                 output_file: str,
                                 max_games_per_file: int = 1000,
                                 max_files: int = None) -> None:
        """Process multiple PGN files and save training data."""
        
        pgn_files = list(Path(pgn_directory).glob("*.pgn"))
        if max_files:
            pgn_files = pgn_files[:max_files]
        
        all_training_examples = []
        
        print(f"Processing {len(pgn_files)} PGN files...")
        
        for i, pgn_file in enumerate(tqdm(pgn_files)):
            print(f"\nProcessing file {i+1}/{len(pgn_files)}: {pgn_file.name}")
            
            examples = self.process_single_pgn_file(
                str(pgn_file), 
                max_games=max_games_per_file
            )
            
            all_training_examples.extend(examples)
            print(f"File {pgn_file.name}: {len(examples)} examples")
        
        # Save training data
        print(f"\nSaving {len(all_training_examples)} training examples to {output_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump({
                'training_examples': all_training_examples,
                'vocabulary': self.vocab,
                'vocab_size': len(self.vocab),
                'policy_size': 4672,
                'total_examples': len(all_training_examples)
            }, f, indent=2)
        
        print(f"Data processing complete!")
        print(f"Total training examples: {len(all_training_examples)}")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Policy size: 4672")
    
    def get_legal_moves_mask(self, board_fen: str) -> np.ndarray:
        """
        Get a binary mask for legal moves from a given board position.
        Returns a numpy array of size 4672 where 1 = legal move, 0 = illegal move.
        """
        board = chess.Board(board_fen)
        legal_mask = np.zeros(4672, dtype=np.float32)
        
        for move in board.legal_moves:
            policy_index = self.uci_to_policy_index(move.uci())
            if policy_index >= 0:
                legal_mask[policy_index] = 1.0
        
        return legal_mask


if __name__ == "__main__":
    # Example usage
    processor = ChessPGNProcessor()
    
    # Process a small sample first
    print("Processing sample PGN files...")
    processor.process_multiple_pgn_files(
        pgn_directory="chess/data/Lichess Elite Database",
        output_file="chess/processed_data/chess_training_sample.json",
        max_games_per_file=100,  # Small sample for testing
        max_files=2
    )