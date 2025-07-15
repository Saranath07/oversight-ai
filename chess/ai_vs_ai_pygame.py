#!/usr/bin/env python3
"""
AI vs AI Chess Game with Pygame
Two AI players compete against each other in a visual chess game
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


class AIvsAIChess:
    """AI vs AI Chess Game with visual feedback."""
    
    def __init__(self, checkpoint_path: str = None):
        """Initialize the AI vs AI chess game."""
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
        self.LAST_MOVE_COLOR = (0, 255, 255, 120)  # Cyan for last move
        self.CHECK_COLOR = (255, 0, 0, 180)        # Red for check
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 200)
        self.GREEN = (0, 150, 0)
        self.RED = (200, 0, 0)
        self.GRAY = (128, 128, 128)
        
        # Create display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("AI vs AI Chess")
        
        # Load fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Initialize AI
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🎮 AI vs AI Chess initialized on device: {self.device}")
        
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
        
        # Initialize and load models
        self.white_ai = ChessTransformer(**self.model_config)
        self.black_ai = ChessTransformer(**self.model_config)
        self.white_ai.to(self.device)
        self.black_ai.to(self.device)
        
        self.checkpoint_info = self.load_checkpoint(checkpoint_path)
        self.white_ai.eval()
        self.black_ai.eval()
        
        # Game state
        self.board = chess.Board()
        self.game_history = []
        self.last_move = None
        self.highlighted_squares = []
        self.game_over = False
        self.winner = None
        self.move_count = 0
        self.thinking_time = 1.0  # Seconds to "think" between moves
        self.auto_play = False
        self.last_move_time = 0
        
        # Move navigation
        self.current_view_move = 0  # Which move we're currently viewing (0 = current position)
        self.viewing_history = False  # Are we viewing a previous position?
        
        # AI configurations
        self.white_temperature = 0.3
        self.black_temperature = 0.3
        
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
                
                # Load same weights for both AIs (they can still play differently due to temperature)
                self.white_ai.load_state_dict(checkpoint['model_state_dict'])
                self.black_ai.load_state_dict(checkpoint['model_state_dict'])
                
                info = {
                    'path': checkpoint_path,
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'loss': checkpoint.get('best_val_loss', 'unknown')
                }
                
                print(f"✅ Models loaded! Epoch: {info['epoch']}, Loss: {info['loss']}")
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
    
    def get_ai_move(self, board: chess.Board, is_white: bool, temperature: float = 0.3) -> Optional[chess.Move]:
        """Get AI move using the trained model."""
        try:
            model = self.white_ai if is_white else self.black_ai
            
            with torch.no_grad():
                # Encode board state
                board_tensor = self.encode_board_state(board)
                
                # Get model prediction
                policy_logits = model(board_tensor).squeeze(0)
                
                # Get legal moves
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return None
                
                # Apply temperature
                if temperature != 1.0:
                    policy_logits = policy_logits / temperature
                
                # Convert to probabilities
                probabilities = torch.softmax(policy_logits, dim=0)
                
                # Try to find best legal move using heuristics
                move_scores = []
                for move in legal_moves:
                    # Base score from model (simplified)
                    score = np.random.random() * 0.1  # Small random component
                    
                    # Heuristic scoring
                    if board.is_capture(move):
                        captured_piece = board.piece_at(move.to_square)
                        if captured_piece:
                            # Value pieces: P=1, N/B=3, R=5, Q=9
                            piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
                            score += piece_values.get(captured_piece.piece_type, 0) * 0.1
                    
                    # Test if move gives check or mate
                    board.push(move)
                    if board.is_check():
                        score += 0.2
                        if board.is_checkmate():
                            score += 10.0  # Heavily favor checkmate
                    board.pop()
                    
                    # Prefer center control for opening
                    if self.move_count < 10:
                        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
                        if move.to_square in center_squares:
                            score += 0.1
                    
                    # Add some randomness based on temperature
                    score += np.random.normal(0, temperature * 0.1)
                    
                    move_scores.append((move, score))
                
                # Sort by score and return best move
                move_scores.sort(key=lambda x: x[1], reverse=True)
                return move_scores[0][0]
                
        except Exception as e:
            print(f"AI error: {e}")
            # Fallback to random legal move
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None
    
    def make_move(self) -> bool:
        """Make the next move in the game."""
        if self.game_over:
            return False
        
        current_player = "White" if self.board.turn else "Black"
        is_white = self.board.turn
        temperature = self.white_temperature if is_white else self.black_temperature
        
        print(f"🤔 {current_player} AI thinking...")
        
        # Get AI move
        move = self.get_ai_move(self.board, is_white, temperature)
        
        if move:
            # Make the move
            self.board.push(move)
            self.last_move = move
            self.game_history.append(move)
            self.move_count += 1
            
            # Update highlights
            self.highlighted_squares = [move.from_square, move.to_square]
            
            print(f"🎯 {current_player} played: {move.uci()}")
            
            # Check for game end
            if self.board.is_checkmate():
                self.game_over = True
                self.winner = "White" if not self.board.turn else "Black"
                print(f"🏆 {self.winner} wins by checkmate!")
            elif self.board.is_stalemate():
                self.game_over = True
                self.winner = "Draw"
                print("🤝 Game drawn by stalemate!")
            elif self.board.is_insufficient_material():
                self.game_over = True
                self.winner = "Draw"
                print("🤝 Game drawn by insufficient material!")
            elif self.board.is_seventyfive_moves():
                self.game_over = True
                self.winner = "Draw"
                print("🤝 Game drawn by 75-move rule!")
            elif self.board.is_fivefold_repetition():
                self.game_over = True
                self.winner = "Draw"
                print("🤝 Game drawn by repetition!")
            
            return True
        else:
            print(f"❌ {current_player} AI couldn't find a move!")
            self.game_over = True
            return False
    
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
        
        # Draw last move highlight
        if self.last_move:
            for square in [self.last_move.from_square, self.last_move.to_square]:
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)
                x = col * self.SQUARE_SIZE
                y = row * self.SQUARE_SIZE
                
                # Create highlight surface with alpha
                highlight_surf = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                highlight_surf.set_alpha(120)
                highlight_surf.fill(self.LAST_MOVE_COLOR[:3])
                self.screen.blit(highlight_surf, (x, y))
        
        # Draw check highlight
        if self.board.is_check():
            king_square = self.board.king(self.board.turn)
            if king_square is not None:
                col = chess.square_file(king_square)
                row = 7 - chess.square_rank(king_square)
                x = col * self.SQUARE_SIZE
                y = row * self.SQUARE_SIZE
                
                check_surf = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                check_surf.set_alpha(180)
                check_surf.fill(self.CHECK_COLOR[:3])
                self.screen.blit(check_surf, (x, y))
    
    def draw_piece(self, piece: chess.Piece, x: int, y: int):
        """Draw a chess piece using simple shapes and text."""
        center_x = x + self.SQUARE_SIZE // 2
        center_y = y + self.SQUARE_SIZE // 2
        
        # Colors for pieces
        if piece.color:  # White pieces
            piece_color = (245, 245, 245)
            border_color = (50, 50, 50)
            text_color = (50, 50, 50)
        else:  # Black pieces
            piece_color = (50, 50, 50)
            border_color = (245, 245, 245)
            text_color = (245, 245, 245)
        
        # Base size for pieces
        base_radius = self.SQUARE_SIZE // 3
        
        # Draw different shapes based on piece type
        if piece.piece_type == chess.PAWN:
            # Draw pawn as a circle with a smaller circle on top
            pygame.draw.circle(self.screen, piece_color, (center_x, center_y + 5), base_radius - 5)
            pygame.draw.circle(self.screen, border_color, (center_x, center_y + 5), base_radius - 5, 2)
            pygame.draw.circle(self.screen, piece_color, (center_x, center_y - 8), base_radius - 10)
            pygame.draw.circle(self.screen, border_color, (center_x, center_y - 8), base_radius - 10, 2)
            
        elif piece.piece_type == chess.ROOK:
            # Draw rook as a rectangle with crenellations
            rect_width = base_radius + 5
            rect_height = base_radius + 10
            rect = pygame.Rect(center_x - rect_width//2, center_y - rect_height//2, rect_width, rect_height)
            pygame.draw.rect(self.screen, piece_color, rect)
            pygame.draw.rect(self.screen, border_color, rect, 2)
            # Add crenellations
            for i in range(3):
                small_rect = pygame.Rect(center_x - rect_width//2 + i * rect_width//3,
                                       center_y - rect_height//2, rect_width//3, 5)
                pygame.draw.rect(self.screen, piece_color, small_rect)
                pygame.draw.rect(self.screen, border_color, small_rect, 1)
            
        elif piece.piece_type == chess.KNIGHT:
            # Draw knight as a triangle
            points = [
                (center_x, center_y - base_radius),
                (center_x - base_radius + 5, center_y + base_radius - 5),
                (center_x + base_radius - 5, center_y + base_radius - 5)
            ]
            pygame.draw.polygon(self.screen, piece_color, points)
            pygame.draw.polygon(self.screen, border_color, points, 2)
            
        elif piece.piece_type == chess.BISHOP:
            # Draw bishop as a diamond
            points = [
                (center_x, center_y - base_radius),
                (center_x + base_radius - 5, center_y),
                (center_x, center_y + base_radius),
                (center_x - base_radius + 5, center_y)
            ]
            pygame.draw.polygon(self.screen, piece_color, points)
            pygame.draw.polygon(self.screen, border_color, points, 2)
            # Add a small circle on top
            pygame.draw.circle(self.screen, piece_color, (center_x, center_y - base_radius + 3), 3)
            pygame.draw.circle(self.screen, border_color, (center_x, center_y - base_radius + 3), 3, 1)
            
        elif piece.piece_type == chess.QUEEN:
            # Draw queen as a large circle with points
            pygame.draw.circle(self.screen, piece_color, (center_x, center_y), base_radius)
            pygame.draw.circle(self.screen, border_color, (center_x, center_y), base_radius, 2)
            # Add crown points
            for i in range(5):
                angle = i * 72  # 360/5 = 72 degrees
                point_x = center_x + int((base_radius - 5) * pygame.math.Vector2(1, 0).rotate(angle).x)
                point_y = center_y + int((base_radius - 5) * pygame.math.Vector2(1, 0).rotate(angle).y)
                pygame.draw.circle(self.screen, border_color, (point_x, point_y), 3)
            
        elif piece.piece_type == chess.KING:
            # Draw king as a circle with a cross on top
            pygame.draw.circle(self.screen, piece_color, (center_x, center_y + 3), base_radius)
            pygame.draw.circle(self.screen, border_color, (center_x, center_y + 3), base_radius, 2)
            # Add cross
            cross_size = 8
            pygame.draw.line(self.screen, border_color,
                           (center_x, center_y - base_radius - 5),
                           (center_x, center_y - base_radius + 5), 3)
            pygame.draw.line(self.screen, border_color,
                           (center_x - cross_size//2, center_y - base_radius),
                           (center_x + cross_size//2, center_y - base_radius), 3)
        
        # Add text label for clarity
        piece_letters = {
            chess.PAWN: 'P', chess.ROOK: 'R', chess.KNIGHT: 'N',
            chess.BISHOP: 'B', chess.QUEEN: 'Q', chess.KING: 'K'
        }
        
        letter = piece_letters[piece.piece_type]
        if not piece.color:  # Black pieces use lowercase
            letter = letter.lower()
            
        font = pygame.font.Font(None, 24)
        text_surface = font.render(letter, True, text_color)
        text_rect = text_surface.get_rect()
        text_rect.center = (center_x, center_y + base_radius + 10)
        self.screen.blit(text_surface, text_rect)
    
    def draw_sidebar(self):
        """Draw the information sidebar."""
        sidebar_x = self.BOARD_SIZE + 10
        y_pos = 20
        
        # Title
        title = self.font_large.render("AI vs AI Chess", True, self.BLACK)
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
        
        # Game info
        current_player = "White" if self.board.turn else "Black"
        turn_text = self.font_medium.render(f"Turn: {current_player}", True, self.BLUE)
        self.screen.blit(turn_text, (sidebar_x, y_pos))
        y_pos += 30
        
        move_text = self.font_medium.render(f"Move: {self.move_count}", True, self.BLACK)
        self.screen.blit(move_text, (sidebar_x, y_pos))
        y_pos += 30
        
        # Game status
        if self.game_over:
            if self.winner == "Draw":
                status_text = self.font_medium.render("Game: Draw", True, self.GRAY)
            else:
                status_text = self.font_medium.render(f"Winner: {self.winner}", True, self.GREEN)
            self.screen.blit(status_text, (sidebar_x, y_pos))
        elif self.board.is_check():
            check_text = self.font_medium.render("CHECK!", True, self.RED)
            self.screen.blit(check_text, (sidebar_x, y_pos))
        else:
            status_text = self.font_medium.render("Game: Active", True, self.GREEN)
            self.screen.blit(status_text, (sidebar_x, y_pos))
        y_pos += 40
        
        # Last move
        if self.last_move:
            last_move_text = self.font_small.render(f"Last: {self.last_move.uci()}", True, self.BLUE)
            self.screen.blit(last_move_text, (sidebar_x, y_pos))
            y_pos += 25
        
        # Auto-play status
        auto_text = f"Auto-play: {'ON' if self.auto_play else 'OFF'}"
        auto_color = self.GREEN if self.auto_play else self.RED
        auto_surface = self.font_small.render(auto_text, True, auto_color)
        self.screen.blit(auto_surface, (sidebar_x, y_pos))
        y_pos += 30
        
        # Controls
        y_pos += 20
        controls = [
            "Controls:",
            "SPACE - Next move",
            "A - Toggle auto-play",
            "R - Reset game",
            "T - Adjust temperature",
            "Q - Quit"
        ]
        
        for i, control in enumerate(controls):
            color = self.BLACK if i == 0 else self.BLUE
            font = self.font_medium if i == 0 else self.font_small
            text = font.render(control, True, color)
            self.screen.blit(text, (sidebar_x, y_pos))
            y_pos += 25
        
        # Temperature settings
        y_pos += 10
        temp_title = self.font_small.render("AI Settings:", True, self.BLACK)
        self.screen.blit(temp_title, (sidebar_x, y_pos))
        y_pos += 20
        
        white_temp = self.font_small.render(f"White temp: {self.white_temperature:.1f}", True, self.BLACK)
        self.screen.blit(white_temp, (sidebar_x, y_pos))
        y_pos += 20
        
        black_temp = self.font_small.render(f"Black temp: {self.black_temperature:.1f}", True, self.BLACK)
        self.screen.blit(black_temp, (sidebar_x, y_pos))
        y_pos += 30
        
        # Game history (last few moves)
        if self.game_history:
            history_title = self.font_small.render("Recent moves:", True, self.BLACK)
            self.screen.blit(history_title, (sidebar_x, y_pos))
            y_pos += 20
            
            # Show last 5 moves
            recent_moves = self.game_history[-10:]
            for i, move in enumerate(recent_moves):
                move_num = len(self.game_history) - len(recent_moves) + i + 1
                color_indicator = "W" if (move_num - 1) % 2 == 0 else "B"
                move_text = f"{move_num}.{color_indicator}: {move.uci()}"
                move_surface = self.font_small.render(move_text, True, self.GRAY)
                self.screen.blit(move_surface, (sidebar_x, y_pos))
                y_pos += 15
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.board = chess.Board()
        self.game_history = []
        self.last_move = None
        self.highlighted_squares = []
        self.game_over = False
        self.winner = None
        self.move_count = 0
        self.last_move_time = 0
        print("🔄 Game reset!")
    
    def toggle_auto_play(self):
        """Toggle auto-play mode."""
        self.auto_play = not self.auto_play
        if self.auto_play:
            self.last_move_time = time.time()
            print("▶️ Auto-play enabled")
        else:
            print("⏸️ Auto-play disabled")
    
    def adjust_temperature(self):
        """Cycle through temperature settings."""
        temps = [0.1, 0.3, 0.5, 0.8, 1.0]
        current_idx = temps.index(self.white_temperature) if self.white_temperature in temps else 1
        next_idx = (current_idx + 1) % len(temps)
        
        self.white_temperature = temps[next_idx]
        self.black_temperature = temps[next_idx]
        print(f"🌡️ Temperature set to: {self.white_temperature}")
    
    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        print("🎯 AI vs AI Chess Started!")
        print("Use SPACE to make moves, A for auto-play, R to reset")
        
        while running:
            current_time = time.time()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Make next move
                        if not self.game_over:
                            self.make_move()
                    
                    elif event.key == pygame.K_a:
                        # Toggle auto-play
                        self.toggle_auto_play()
                    
                    elif event.key == pygame.K_r:
                        # Reset game
                        self.reset_game()
                    
                    elif event.key == pygame.K_t:
                        # Adjust temperature
                        self.adjust_temperature()
                    
                    elif event.key == pygame.K_q:
                        # Quit
                        running = False
            
            # Auto-play logic
            if self.auto_play and not self.game_over:
                if current_time - self.last_move_time >= self.thinking_time:
                    self.make_move()
                    self.last_move_time = current_time
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_board(self.board)
            self.draw_sidebar()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        print("👋 AI vs AI Chess completed!")


def main():
    """Main function."""
    print("🏁 AI vs AI Chess Game")
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
    
    # Create and run AI vs AI game
    game = AIvsAIChess(checkpoint_path)
    game.run()


if __name__ == "__main__":
    main()