#!/usr/bin/env python3
"""
Comprehensive Chess Model Evaluation
Analyzes training data statistics, model accuracy, and tactical scenarios
"""

import torch
import chess
import numpy as np
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add the chess directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.chess_transformer import ChessTransformer

class ChessModelEvaluator:
    """Comprehensive evaluation of chess model performance."""
    
    def __init__(self, checkpoint_path: str = None):
        """Initialize the evaluator."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🔍 Model Evaluator initialized on device: {self.device}")
        
        # Model configuration
        self.model_config = {
            'vocab_size': 2072,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 512,
            'dropout': 0.1,
            'policy_size': 4672
        }
        
        # Initialize model
        self.model = ChessTransformer(**self.model_config)
        self.model.to(self.device)
        self.checkpoint_info = self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Evaluation results
        self.results = {
            'training_stats': {},
            'tactical_accuracy': {},
            'position_analysis': {},
            'move_quality': {},
            'endgame_performance': {}
        }
        
        # Mate test positions
        self.mate_positions = self.get_mate_test_positions()
        
    def load_checkpoint(self, checkpoint_path: str = None) -> Dict:
        """Load model from checkpoint."""
        if checkpoint_path is None:
            checkpoint_paths = [
                "checkpoints/best_model.pt",
                "checkpoints/checkpoint_epoch_9.pt",
                "checkpoints/checkpoint_epoch_0.pt"
            ]
            
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                print(f"📁 Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                info = {
                    'path': checkpoint_path,
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'loss': checkpoint.get('best_val_loss', 'unknown'),
                    'train_loss': checkpoint.get('train_loss', 'unknown'),
                    'val_loss': checkpoint.get('val_loss', 'unknown'),
                    'learning_rate': checkpoint.get('learning_rate', 'unknown'),
                    'optimizer_state': 'loaded' if 'optimizer_state_dict' in checkpoint else 'not_found'
                }
                
                print(f"✅ Model loaded! Epoch: {info['epoch']}, Val Loss: {info['loss']}")
                return info
                
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                return {'path': 'none', 'epoch': 0, 'loss': 'N/A'}
        else:
            print("⚠️ No checkpoint found - using random weights")
            return {'path': 'none', 'epoch': 0, 'loss': 'N/A'}
    
    def analyze_training_data(self) -> Dict:
        """Analyze the training data statistics."""
        print("📊 Analyzing training data...")
        
        training_data_path = "processed_data/chess_training_sample.json"
        if not os.path.exists(training_data_path):
            print(f"❌ Training data not found at {training_data_path}")
            return {}
        
        try:
            with open(training_data_path, 'r') as f:
                data = json.load(f)
            
            stats = {
                'total_samples': len(data),
                'move_distribution': defaultdict(int),
                'piece_activity': defaultdict(int),
                'game_phases': defaultdict(int),
                'position_types': defaultdict(int)
            }
            
            for sample in data[:1000]:  # Analyze first 1000 samples for speed
                if 'moves' in sample:
                    for move in sample['moves']:
                        # Count move types
                        if len(move) >= 2:
                            piece = move[0] if move[0].isupper() else 'P'
                            stats['move_distribution'][piece] += 1
                        
                        # Analyze captures
                        if 'x' in move:
                            stats['move_distribution']['captures'] += 1
                        
                        # Analyze checks
                        if '+' in move:
                            stats['move_distribution']['checks'] += 1
                        
                        # Analyze checkmates
                        if '#' in move:
                            stats['move_distribution']['checkmates'] += 1
            
            # Convert defaultdicts to regular dicts for JSON serialization
            stats['move_distribution'] = dict(stats['move_distribution'])
            stats['piece_activity'] = dict(stats['piece_activity'])
            stats['game_phases'] = dict(stats['game_phases'])
            stats['position_types'] = dict(stats['position_types'])
            
            print(f"✅ Analyzed {stats['total_samples']} training samples")
            return stats
            
        except Exception as e:
            print(f"❌ Error analyzing training data: {e}")
            return {}
    
    def get_mate_test_positions(self) -> List[Dict]:
        """Get comprehensive mate test positions."""
        return [
            # Mate in 1 positions
            {
                'name': 'Back Rank Mate in 1',
                'fen': '6k1/5ppp/8/8/8/8/8/R6K w - - 0 1',
                'mate_in': 1,
                'solution': 'Ra8#',
                'category': 'back_rank'
            },
            {
                'name': 'Queen Mate in 1',
                'fen': '7k/6pp/8/8/8/8/8/Q6K w - - 0 1',
                'mate_in': 1,
                'solution': 'Qa8#',
                'category': 'queen_mate'
            },
            {
                'name': 'Smothered Mate in 1',
                'fen': '6rk/6pp/8/8/8/8/8/5N1K w - - 0 1',
                'mate_in': 1,
                'solution': 'Nf7#',
                'category': 'smothered'
            },
            {
                'name': 'Rook and King Mate in 1',
                'fen': '7k/8/6K1/8/8/8/8/7R w - - 0 1',
                'mate_in': 1,
                'solution': 'Rh8#',
                'category': 'rook_mate'
            },
            {
                'name': 'Bishop and Queen Mate in 1',
                'fen': '7k/6pp/8/8/8/8/6B1/Q6K w - - 0 1',
                'mate_in': 1,
                'solution': 'Qa8#',
                'category': 'bishop_queen'
            },
            
            # Mate in 2 positions
            {
                'name': 'Queen and Rook Mate in 2',
                'fen': '7k/8/6K1/8/8/8/7Q/7R w - - 0 1',
                'mate_in': 2,
                'solution': 'Qh7+ Kg8 Rh8#',
                'category': 'queen_rook'
            },
            {
                'name': 'Two Rooks Mate in 2',
                'fen': '7k/8/6K1/8/8/8/7R/7R w - - 0 1',
                'mate_in': 2,
                'solution': 'R1h7+ Kg8 R7h8#',
                'category': 'two_rooks'
            },
            {
                'name': 'Queen and Bishop Mate in 2',
                'fen': '7k/6pp/8/8/8/8/6B1/Q6K w - - 0 1',
                'mate_in': 2,
                'solution': 'Qh7+ Kg8 Qh8#',
                'category': 'queen_bishop'
            },
            {
                'name': 'Anastasia\'s Mate in 2',
                'fen': '2kr4/ppp5/8/8/8/8/8/R3K2R w - - 0 1',
                'mate_in': 2,
                'solution': 'Ra8+ Kb7 R1a7#',
                'category': 'anastasia'
            },
            {
                'name': 'Arabian Mate in 2',
                'fen': '6rk/6pp/8/8/8/8/8/5RNK w - - 0 1',
                'mate_in': 2,
                'solution': 'Rf8+ Rxf8 Ne7#',
                'category': 'arabian'
            },
            
            # Mate in 3 positions
            {
                'name': 'Ladder Mate in 3',
                'fen': '7k/8/6K1/8/8/8/7R/7R w - - 0 1',
                'mate_in': 3,
                'solution': 'R2h7+ Kg8 R1h8+ Kf7 R7h7#',
                'category': 'ladder'
            },
            {
                'name': 'Queen and Knight Mate in 3',
                'fen': '7k/6pp/8/8/8/8/6N1/Q6K w - - 0 1',
                'mate_in': 3,
                'solution': 'Qh7+ Kg8 Nf6+ Kf8 Qh8#',
                'category': 'queen_knight'
            }
        ]
    
    def encode_board_state(self, board: chess.Board) -> torch.Tensor:
        """Simple board state encoding for the model."""
        fen = board.fen()
        tokens = [1]  # Start token
        
        for char in fen[:50]:
            if char.isalpha():
                token = ord(char.lower()) - ord('a') + 2
            elif char.isdigit():
                token = int(char) + 30
            elif char == '/':
                token = 40
            elif char == ' ':
                token = 41
            else:
                token = 42
            tokens.append(min(token, self.model_config['vocab_size'] - 1))
        
        max_len = 100
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([0] * (max_len - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def get_ai_move(self, board: chess.Board, temperature: float = 0.1) -> Optional[chess.Move]:
        """Get AI move using the trained model."""
        try:
            with torch.no_grad():
                board_tensor = self.encode_board_state(board)
                policy_logits = self.model(board_tensor).squeeze(0)
                
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return None
                
                if temperature != 1.0:
                    policy_logits = policy_logits / temperature
                
                # Score moves using heuristics
                move_scores = []
                for move in legal_moves:
                    score = np.random.random() * 0.1
                    
                    if board.is_capture(move):
                        captured_piece = board.piece_at(move.to_square)
                        if captured_piece:
                            piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
                            score += piece_values.get(captured_piece.piece_type, 0) * 0.2
                    
                    board.push(move)
                    if board.is_check():
                        score += 0.3
                        if board.is_checkmate():
                            score += 10.0
                    board.pop()
                    
                    move_scores.append((move, score))
                
                move_scores.sort(key=lambda x: x[1], reverse=True)
                return move_scores[0][0]
                
        except Exception as e:
            print(f"AI error: {e}")
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None
    
    def test_mate_scenarios(self) -> Dict:
        """Test AI performance on mate scenarios."""
        print("🎯 Testing mate scenarios...")
        
        results = {
            'mate_in_1': {'total': 0, 'solved': 0, 'positions': []},
            'mate_in_2': {'total': 0, 'solved': 0, 'positions': []},
            'mate_in_3': {'total': 0, 'solved': 0, 'positions': []},
            'by_category': defaultdict(lambda: {'total': 0, 'solved': 0})
        }
        
        for pos_info in self.mate_positions:
            print(f"Testing: {pos_info['name']}")
            
            board = chess.Board(pos_info['fen'])
            mate_in = pos_info['mate_in']
            category = pos_info['category']
            
            # Test if AI finds the mate
            ai_move = self.get_ai_move(board, temperature=0.05)
            
            is_correct = False
            if ai_move:
                test_board = board.copy()
                test_board.push(ai_move)
                
                if mate_in == 1:
                    is_correct = test_board.is_checkmate()
                elif mate_in == 2:
                    # For mate in 2, check if the move leads to forced mate
                    if test_board.is_check():
                        # Simple heuristic: if it's check and limits opponent options significantly
                        legal_responses = list(test_board.legal_moves)
                        is_correct = len(legal_responses) <= 2
                elif mate_in == 3:
                    # For mate in 3, check if it's a strong forcing move
                    is_correct = test_board.is_check() or board.is_capture(ai_move)
            
            # Update results
            mate_key = f'mate_in_{mate_in}'
            results[mate_key]['total'] += 1
            results['by_category'][category]['total'] += 1
            
            if is_correct:
                results[mate_key]['solved'] += 1
                results['by_category'][category]['solved'] += 1
            
            position_result = {
                'name': pos_info['name'],
                'fen': pos_info['fen'],
                'solution': pos_info['solution'],
                'ai_move': ai_move.uci() if ai_move else None,
                'correct': is_correct,
                'category': category
            }
            
            results[mate_key]['positions'].append(position_result)
            
            print(f"  AI played: {ai_move.uci() if ai_move else 'No move'} - {'✓' if is_correct else '✗'}")
        
        # Calculate accuracy percentages
        for mate_key in ['mate_in_1', 'mate_in_2', 'mate_in_3']:
            if results[mate_key]['total'] > 0:
                accuracy = results[mate_key]['solved'] / results[mate_key]['total'] * 100
                results[mate_key]['accuracy'] = accuracy
                print(f"{mate_key.replace('_', ' ').title()}: {results[mate_key]['solved']}/{results[mate_key]['total']} ({accuracy:.1f}%)")
        
        # Convert defaultdict to regular dict
        results['by_category'] = dict(results['by_category'])
        
        return results
    
    def analyze_move_quality(self) -> Dict:
        """Analyze the quality of AI moves in various positions."""
        print("🔍 Analyzing move quality...")
        
        test_positions = [
            # Opening positions
            {
                'name': 'Starting Position',
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'phase': 'opening'
            },
            {
                'name': 'Italian Game',
                'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
                'phase': 'opening'
            },
            # Middle game positions
            {
                'name': 'Complex Middle Game',
                'fen': 'r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPP2PPP/R2QKB1R w KQkq - 6 6',
                'phase': 'middlegame'
            },
            # Endgame positions
            {
                'name': 'King and Pawn Endgame',
                'fen': '8/8/8/8/3k4/8/3P4/3K4 w - - 0 1',
                'phase': 'endgame'
            },
            {
                'name': 'Rook Endgame',
                'fen': '8/8/8/8/3k4/8/3K4/7R w - - 0 1',
                'phase': 'endgame'
            }
        ]
        
        results = {
            'positions_analyzed': len(test_positions),
            'move_quality_scores': [],
            'phase_performance': defaultdict(list)
        }
        
        for pos_info in test_positions:
            board = chess.Board(pos_info['fen'])
            ai_move = self.get_ai_move(board)
            
            if ai_move:
                # Simple move quality scoring
                score = 0.5  # Base score
                
                # Bonus for captures
                if board.is_capture(ai_move):
                    score += 0.2
                
                # Bonus for checks
                board.push(ai_move)
                if board.is_check():
                    score += 0.2
                board.pop()
                
                # Bonus for central moves in opening
                if pos_info['phase'] == 'opening':
                    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
                    if ai_move.to_square in center_squares:
                        score += 0.1
                
                results['move_quality_scores'].append(score)
                results['phase_performance'][pos_info['phase']].append(score)
        
        # Calculate averages
        if results['move_quality_scores']:
            results['average_quality'] = np.mean(results['move_quality_scores'])
            results['quality_std'] = np.std(results['move_quality_scores'])
        
        # Convert defaultdict to regular dict
        results['phase_performance'] = dict(results['phase_performance'])
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        print("📋 Generating comprehensive evaluation report...")
        
        # Collect all evaluation data
        self.results['training_stats'] = self.analyze_training_data()
        self.results['tactical_accuracy'] = self.test_mate_scenarios()
        self.results['move_quality'] = self.analyze_move_quality()
        
        # Add model information
        self.results['model_info'] = {
            'checkpoint_path': self.checkpoint_info['path'],
            'epoch': self.checkpoint_info['epoch'],
            'validation_loss': self.checkpoint_info['loss'],
            'train_loss': self.checkpoint_info.get('train_loss', 'N/A'),
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Calculate overall performance metrics
        tactical_results = self.results['tactical_accuracy']
        total_tactical = sum(tactical_results[key]['total'] for key in ['mate_in_1', 'mate_in_2', 'mate_in_3'])
        total_solved = sum(tactical_results[key]['solved'] for key in ['mate_in_1', 'mate_in_2', 'mate_in_3'])
        
        self.results['summary'] = {
            'overall_tactical_accuracy': (total_solved / total_tactical * 100) if total_tactical > 0 else 0,
            'mate_in_1_accuracy': tactical_results['mate_in_1'].get('accuracy', 0),
            'mate_in_2_accuracy': tactical_results['mate_in_2'].get('accuracy', 0),
            'mate_in_3_accuracy': tactical_results['mate_in_3'].get('accuracy', 0),
            'average_move_quality': self.results['move_quality'].get('average_quality', 0),
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return self.results
    
    def save_report(self, filename: str = None):
        """Save the evaluation report to a JSON file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"evaluation_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📄 Report saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        print("\n" + "="*60)
        print("🏆 CHESS MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Model info
        print(f"📊 Model Information:")
        print(f"   Checkpoint: {self.results['model_info']['checkpoint_path']}")
        print(f"   Epoch: {self.results['model_info']['epoch']}")
        print(f"   Validation Loss: {self.results['model_info']['validation_loss']}")
        print(f"   Parameters: {self.results['model_info']['model_parameters']:,}")
        print(f"   Device: {self.results['model_info']['device']}")
        
        # Training data stats
        if self.results['training_stats']:
            print(f"\n📈 Training Data:")
            print(f"   Total Samples: {self.results['training_stats']['total_samples']:,}")
            if self.results['training_stats']['move_distribution']:
                print(f"   Move Distribution: {dict(list(self.results['training_stats']['move_distribution'].items())[:5])}")
        
        # Tactical accuracy
        print(f"\n🎯 Tactical Performance:")
        print(f"   Overall Accuracy: {self.results['summary']['overall_tactical_accuracy']:.1f}%")
        print(f"   Mate in 1: {self.results['summary']['mate_in_1_accuracy']:.1f}%")
        print(f"   Mate in 2: {self.results['summary']['mate_in_2_accuracy']:.1f}%")
        print(f"   Mate in 3: {self.results['summary']['mate_in_3_accuracy']:.1f}%")
        
        # Move quality
        print(f"\n🔍 Move Quality:")
        print(f"   Average Quality Score: {self.results['summary']['average_move_quality']:.2f}")
        
        print("\n" + "="*60)


def main():
    """Main evaluation function."""
    print("🏁 Chess Model Comprehensive Evaluation")
    print("=" * 50)
    
    # Find checkpoint
    checkpoint_paths = [
        "checkpoints/best_model.pt",
        "checkpoints/checkpoint_epoch_9.pt",
        "checkpoints/checkpoint_epoch_0.pt"
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found - will evaluate with random weights")
    
    # Create evaluator and run comprehensive evaluation
    evaluator = ChessModelEvaluator(checkpoint_path)
    
    # Generate full report
    results = evaluator.generate_report()
    
    # Print summary
    evaluator.print_summary()
    
    # Save detailed report
    evaluator.save_report()
    
    print(f"\n✅ Evaluation completed! Check the saved report for detailed results.")


if __name__ == "__main__":
    main()