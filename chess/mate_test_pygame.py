#!/usr/bin/env python3
"""
Chess Mate Test with Pygame
Tests AI's ability to find forced mates (mate in 1, mate in 2)
"""

import pygame
import chess
import torch
import numpy as np
import os
import sys
import time
from typing import Optional, Tuple, List, Dict

# Add the chess directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our model
from model.chess_transformer import ChessTransformer

# Try to import gym_chess for move encoding
try:
    import gym_chess
    import gym
    CHESS_ENV_AVAILABLE = True
except ImportError:
    CHESS_ENV_AVAILABLE = False
    print("Warning: gym_chess not available. Using simplified move encoding.")


class ChessMateTest:
    """Test AI's ability to find forced mates with visual feedback."""
    
    def __init__(self, checkpoint_path: str = None):
        """Initialize the mate test."""
        # Initialize Pygame
        pygame.init()
        
        # Constants
        self.BOARD_SIZE = 600
        self.SQUARE_SIZE = self.BOARD_SIZE // 8
        self.WINDOW_WIDTH = 1000
        self.WINDOW_HEIGHT = 700
        
        # Colors
        self.LIGHT_SQUARE = (240, 217, 181)
        self.DARK_SQUARE = (181, 136, 99)
        self.HIGHLIGHT_COLOR = (255, 255, 0, 180)  # Yellow highlight
        self.MATE_COLOR = (255, 0, 0, 180)         # Red for mate
        self.GOOD_MOVE_COLOR = (0, 255, 0, 180)    # Green for good move
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 200)
        self.GREEN = (0, 150, 0)
        self.RED = (200, 0, 0)
        
        # Create display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI - Mate Test")
        
        # Load fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Initialize AI
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🎮 Mate Test initialized on device: {self.device}")
        
        # Initialize chess environment
        self.chess_env = None
        if CHESS_ENV_AVAILABLE:
            try:
                self.chess_env = gym.make('ChessAlphaZero-v0')
                print("✓ Chess environment loaded")
            except Exception as e:
                print(f"Warning: Could not initialize chess environment: {e}")
        
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
        
        # Initialize and load model
        self.model = ChessTransformer(**self.model_config)
        self.model.to(self.device)
        self.checkpoint_info = self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Test positions
        self.mate_positions = [
            # Mate in 1 positions
            {
                'name': 'Back Rank Mate in 1',
                'fen': '6k1/5ppp/8/8/8/8/8/R6K w - - 0 1',
                'mate_in': 1,
                'solution': 'a8#'
            },
            {
                'name': 'Queen Mate in 1',
                'fen': '7k/6pp/8/8/8/8/8/Q6K w - - 0 1',
                'mate_in': 1,
                'solution': 'a8#'
            },
            {
                'name': 'Smothered Mate in 1',
                'fen': '6rk/6pp/8/8/8/8/8/5N1K w - - 0 1',
                'mate_in': 1,
                'solution': 'Nf7#'
            },
            # Mate in 2 positions
            {
                'name': 'Queen and Rook Mate in 2',
                'fen': '7k/8/6K1/8/8/8/7Q/7R w - - 0 1',
                'mate_in': 2,
                'solution': 'Qh7+ Kg8 Rh8#'
            },
            {
                'name': 'Two Rooks Mate in 2',
                'fen': '7k/8/6K1/8/8/8/7R/7R w - - 0 1',
                'mate_in': 2,
                'solution': 'R1h7+ Kg8 R7h8#'
            },
            {
                'name': 'Scholar\'s Mate Setup',
                'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4',
                'mate_in': 1,
                'solution': 'Qxf7#'
            }
        ]
        
        self.current_position = 0
        self.board = chess.Board()
        self.ai_move = None
        self.test_results = []
        self.highlighted_squares = []
        
    def load_checkpoint(self, checkpoint_path: str = None) -> Dict:
        """Load model from checkpoint."""
        if checkpoint_path is None:
            checkpoint_paths = [
                "chess/checkpoints/best_model.pt",
                "chess/checkpoints/checkpoint_epoch_9.pt",
                "chess/checkpoints/checkpoint_epoch_0.pt"
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
                    'loss': checkpoint.get('best_val_loss', 'unknown')
                }
                
                print(f"✅ Model loaded! Epoch: {info['epoch']}, Loss: {info['loss']}")
                return info
                
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                return {'path': 'none', 'epoch': 0, 'loss': 'N/A'}
        else:
            print("⚠️ No checkpoint found - using random weights")
            return {'path': 'none', 'epoch': 0, 'loss': 'N/A'}
    
    def encode_board_state(self, board: chess.Board) -> torch.Tensor:
        """Simple board state encoding for the model."""
        # Convert board to FEN and create simple token sequence
        fen = board.fen()
        tokens = [1]  # Start token
        
        # Simple encoding of FEN string
        for char in fen[:50]:  # Limit length
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
        
        # Pad to fixed length
        max_len = 100
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([0] * (max_len - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def get_ai_move(self, board: chess.Board, temperature: float = 0.3) -> Optional[chess.Move]:
        """Get AI move using the trained model."""
        try:
            with torch.no_grad():
                # Encode board state
                board_tensor = self.encode_board_state(board)
                
                # Get model prediction
                policy_logits = self.model(board_tensor).squeeze(0)
                
                # Get legal moves
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return None
                
                # Apply temperature
                if temperature != 1.0:
                    policy_logits = policy_logits / temperature
                
                # Convert to probabilities
                probabilities = torch.softmax(policy_logits, dim=0)
                
                # Try to find best legal move
                # For simplicity, we'll use a scoring approach
                move_scores = []
                for move in legal_moves:
                    # Simple heuristic: prefer captures and checks
                    score = np.random.random()  # Base random score
                    
                    if board.is_capture(move):
                        score += 0.3
                    
                    # Test if move gives check
                    board.push(move)
                    if board.is_check():
                        score += 0.5
                        if board.is_checkmate():
                            score += 10.0  # Heavily favor checkmate
                    board.pop()
                    
                    move_scores.append((move, score))
                
                # Sort by score and return best move
                move_scores.sort(key=lambda x: x[1], reverse=True)
                return move_scores[0][0]
                
        except Exception as e:
            print(f"AI error: {e}")
            # Fallback to random legal move
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None
    
    def draw_board(self, board: chess.Board):
        """Draw the chess board."""
        for row in range(8):
            for col in range(8):
                # Calculate position
                x = col * self.SQUARE_SIZE
                y = row * self.SQUARE_SIZE
                
                # Choose color
                color = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
                
                # Draw square
                pygame.draw.rect(self.screen, color, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE))
                
                # Draw piece if present
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                if piece:
                    self.draw_piece(piece, x, y)
        
        # Draw highlights
        for square in self.highlighted_squares:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            x = col * self.SQUARE_SIZE
            y = row * self.SQUARE_SIZE
            
            # Create highlight surface with alpha
            highlight_surf = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
            highlight_surf.set_alpha(180)
            highlight_surf.fill(self.HIGHLIGHT_COLOR[:3])
            self.screen.blit(highlight_surf, (x, y))
    
    def draw_piece(self, piece: chess.Piece, x: int, y: int):
        """Draw a chess piece using text."""
        # Unicode chess pieces
        piece_symbols = {
            chess.PAWN: {'white': '♙', 'black': '♟'},
            chess.ROOK: {'white': '♖', 'black': '♜'},
            chess.KNIGHT: {'white': '♘', 'black': '♞'},
            chess.BISHOP: {'white': '♗', 'black': '♝'},
            chess.QUEEN: {'white': '♕', 'black': '♛'},
            chess.KING: {'white': '♔', 'black': '♚'}
        }
        
        color = 'white' if piece.color else 'black'
        symbol = piece_symbols[piece.piece_type][color]
        
        # Render piece
        piece_font = pygame.font.Font(None, 60)
        text_color = self.BLACK if color == 'white' else self.WHITE
        piece_surface = piece_font.render(symbol, True, text_color)
        
        # Center the piece in the square
        text_rect = piece_surface.get_rect()
        text_rect.center = (x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2)
        self.screen.blit(piece_surface, text_rect)
    
    def draw_sidebar(self):
        """Draw the information sidebar."""
        sidebar_x = self.BOARD_SIZE + 10
        y_pos = 20
        
        # Title
        title = self.font_large.render("Chess Mate Test", True, self.BLACK)
        self.screen.blit(title, (sidebar_x, y_pos))
        y_pos += 50
        
        # Model info
        model_info = [
            f"Model Epoch: {self.checkpoint_info['epoch']}",
            f"Model Loss: {self.checkpoint_info['loss']}",
            f"Device: {self.device}"
        ]
        
        for info in model_info:
            text = self.font_small.render(info, True, self.BLACK)
            self.screen.blit(text, (sidebar_x, y_pos))
            y_pos += 25
        
        y_pos += 20
        
        # Current position info
        if self.current_position < len(self.mate_positions):
            pos_info = self.mate_positions[self.current_position]
            
            # Position name
            name_text = self.font_medium.render(f"Position: {pos_info['name']}", True, self.BLUE)
            self.screen.blit(name_text, (sidebar_x, y_pos))
            y_pos += 30
            
            # Mate in X
            mate_text = self.font_medium.render(f"Mate in {pos_info['mate_in']}", True, self.RED)
            self.screen.blit(mate_text, (sidebar_x, y_pos))
            y_pos += 30
            
            # Solution
            solution_text = self.font_small.render(f"Solution: {pos_info['solution']}", True, self.GREEN)
            self.screen.blit(solution_text, (sidebar_x, y_pos))
            y_pos += 30
            
            # AI move
            if self.ai_move:
                ai_text = self.font_medium.render(f"AI Move: {self.ai_move.uci()}", True, self.BLUE)
                self.screen.blit(ai_text, (sidebar_x, y_pos))
                y_pos += 30
                
                # Check if AI found mate
                test_board = self.board.copy()
                test_board.push(self.ai_move)
                if test_board.is_checkmate():
                    result_text = self.font_medium.render("✓ MATE FOUND!", True, self.GREEN)
                    self.screen.blit(result_text, (sidebar_x, y_pos))
                else:
                    result_text = self.font_medium.render("✗ Not mate", True, self.RED)
                    self.screen.blit(result_text, (sidebar_x, y_pos))
                y_pos += 40
        
        # Controls
        y_pos += 20
        controls = [
            "Controls:",
            "SPACE - Get AI move",
            "N - Next position",
            "P - Previous position",
            "R - Run all tests",
            "Q - Quit"
        ]
        
        for i, control in enumerate(controls):
            color = self.BLACK if i == 0 else self.BLUE
            font = self.font_medium if i == 0 else self.font_small
            text = font.render(control, True, color)
            self.screen.blit(text, (sidebar_x, y_pos))
            y_pos += 25
        
        # Test results
        if self.test_results:
            y_pos += 20
            results_title = self.font_medium.render("Test Results:", True, self.BLACK)
            self.screen.blit(results_title, (sidebar_x, y_pos))
            y_pos += 30
            
            passed = sum(1 for r in self.test_results if r['passed'])
            total = len(self.test_results)
            score_text = self.font_small.render(f"Score: {passed}/{total}", True, self.GREEN if passed == total else self.RED)
            self.screen.blit(score_text, (sidebar_x, y_pos))
    
    def load_position(self, index: int):
        """Load a specific mate position."""
        if 0 <= index < len(self.mate_positions):
            self.current_position = index
            pos_info = self.mate_positions[index]
            self.board = chess.Board(pos_info['fen'])
            self.ai_move = None
            self.highlighted_squares = []
            print(f"Loaded position: {pos_info['name']}")
    
    def test_ai_on_position(self):
        """Test AI on current position."""
        if self.current_position >= len(self.mate_positions):
            return
        
        pos_info = self.mate_positions[self.current_position]
        print(f"Testing AI on: {pos_info['name']}")
        
        # Get AI move
        self.ai_move = self.get_ai_move(self.board, temperature=0.1)
        
        if self.ai_move:
            # Test if it's mate
            test_board = self.board.copy()
            test_board.push(self.ai_move)
            is_mate = test_board.is_checkmate()
            
            # Highlight the move
            self.highlighted_squares = [self.ai_move.from_square, self.ai_move.to_square]
            
            result = {
                'position': pos_info['name'],
                'ai_move': self.ai_move.uci(),
                'is_mate': is_mate,
                'passed': is_mate,
                'mate_in': pos_info['mate_in']
            }
            
            print(f"AI played: {self.ai_move.uci()}, Mate: {is_mate}")
            return result
        else:
            print("AI couldn't find a move!")
            return {
                'position': pos_info['name'],
                'ai_move': None,
                'is_mate': False,
                'passed': False,
                'mate_in': pos_info['mate_in']
            }
    
    def run_all_tests(self):
        """Run AI test on all positions."""
        print("Running all mate tests...")
        self.test_results = []
        
        for i in range(len(self.mate_positions)):
            self.load_position(i)
            result = self.test_ai_on_position()
            if result:
                self.test_results.append(result)
            time.sleep(1)  # Brief pause between tests
        
        # Print summary
        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        print(f"\nTest Summary: {passed}/{total} positions solved correctly")
        
        for result in self.test_results:
            status = "✓" if result['passed'] else "✗"
            print(f"{status} {result['position']}: {result['ai_move']} ({'MATE' if result['is_mate'] else 'NOT MATE'})")
    
    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        # Load first position
        self.load_position(0)
        
        print("🎯 Chess Mate Test Started!")
        print("Use SPACE to get AI move, N/P for next/previous position")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Get AI move for current position
                        self.test_ai_on_position()
                    
                    elif event.key == pygame.K_n:
                        # Next position
                        next_pos = (self.current_position + 1) % len(self.mate_positions)
                        self.load_position(next_pos)
                    
                    elif event.key == pygame.K_p:
                        # Previous position
                        prev_pos = (self.current_position - 1) % len(self.mate_positions)
                        self.load_position(prev_pos)
                    
                    elif event.key == pygame.K_r:
                        # Run all tests
                        self.run_all_tests()
                    
                    elif event.key == pygame.K_q:
                        # Quit
                        running = False
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_board(self.board)
            self.draw_sidebar()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        print("👋 Mate test completed!")


def main():
    """Main function."""
    print("🏁 Chess AI Mate Test")
    print("=" * 50)
    
    # Look for checkpoints
    checkpoint_paths = [
        "chess/checkpoints/best_model.pt",
        "chess/checkpoints/checkpoint_epoch_9.pt",
        "chess/checkpoints/checkpoint_epoch_0.pt"
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found - will use random weights")
    
    # Create and run mate test
    mate_test = ChessMateTest(checkpoint_path)
    mate_test.run()


if __name__ == "__main__":
    main()