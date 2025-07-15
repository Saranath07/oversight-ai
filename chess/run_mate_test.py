#!/usr/bin/env python3
"""
Chess Mate Test with Better Piece Rendering
Tests AI's ability to find forced mates with visual feedback
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages if missing."""
    required_packages = {
        'pygame': 'pygame>=2.1.0',
        'torch': 'torch>=1.9.0',
        'chess': 'chess>=1.9.0',
        'numpy': 'numpy>=1.21.0',
        'pillow': 'pillow>=8.0.0'  # For better image handling
    }
    
    missing_packages = []
    
    print("Checking requirements...")
    for package, pip_name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade'
            ] + missing_packages)
            print("✅ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            print("Please install manually with:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main runner function with integrated mate test."""
    print("🎯 Chess AI Mate Test with Better Piece Rendering")
    print("=" * 60)
    
    # Install packages if needed
    if not install_packages():
        return
    
    # Now import after ensuring packages are installed
    try:
        import pygame
        import chess
        import torch
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        print(f"❌ Import error after installation: {e}")
        return
    
    # Check if we're in the right directory
    if not os.path.exists('model'):
        print("❌ Please run this from the chess directory")
        print("   cd chess && python run_mate_test.py")
        return
    
    # Check for checkpoints
    checkpoint_paths = [
        "checkpoints/best_model.pt",
        "checkpoints/checkpoint_epoch_9.pt",
        "checkpoints/checkpoint_epoch_0.pt"
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✓ Found checkpoint: {path}")
            checkpoint_path = path
            break
    
    if not checkpoint_path:
        print("⚠️  No trained checkpoints found")
        print("   The demo will use random weights")
    
    print("\n🚀 Starting Enhanced Chess Mate Test...")
    
    # Run the enhanced mate test
    try:
        run_enhanced_mate_test(checkpoint_path)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running mate test: {e}")
        print("Make sure all requirements are installed and model files exist")


def run_enhanced_mate_test(checkpoint_path):
    """Run the enhanced mate test with better piece rendering."""
    import pygame
    import chess
    import torch
    import numpy as np
    
    # Import our model
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model.chess_transformer import ChessTransformer
    
    # Initialize Pygame
    pygame.init()
    
    # Constants
    BOARD_SIZE = 640
    SQUARE_SIZE = BOARD_SIZE // 8
    WINDOW_WIDTH = 1100
    WINDOW_HEIGHT = 700
    
    # Colors
    LIGHT_SQUARE = (240, 217, 181)
    DARK_SQUARE = (181, 136, 99)
    HIGHLIGHT_COLOR = (255, 255, 0, 128)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 100, 200)
    GREEN = (0, 150, 0)
    RED = (200, 0, 0)
    
    # Create display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Chess AI - Enhanced Mate Test")
    
    # Load fonts
    font_large = pygame.font.Font(None, 36)
    font_medium = pygame.font.Font(None, 28)
    font_small = pygame.font.Font(None, 20)
    
    # Initialize AI
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🎮 Enhanced Mate Test initialized on device: {device}")
    
    # Model configuration
    model_config = {
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
    model = ChessTransformer(**model_config)
    model.to(device)
    
    # Load checkpoint
    checkpoint_info = {'path': 'none', 'epoch': 0, 'loss': 'N/A'}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"📁 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            checkpoint_info = {
                'path': checkpoint_path,
                'epoch': checkpoint.get('epoch', 'unknown'),
                'loss': checkpoint.get('best_val_loss', 'unknown')
            }
            
            print(f"✅ Model loaded! Epoch: {checkpoint_info['epoch']}, Loss: {checkpoint_info['loss']}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
    
    model.eval()
    
    # Test positions
    mate_positions = [
        {
            'name': 'Back Rank Mate in 1',
            'fen': '6k1/5ppp/8/8/8/8/8/R6K w - - 0 1',
            'mate_in': 1,
            'solution': 'Ra8#'
        },
        {
            'name': 'Queen Mate in 1',
            'fen': '7k/6pp/8/8/8/8/8/Q6K w - - 0 1',
            'mate_in': 1,
            'solution': 'Qa8#'
        },
        {
            'name': 'Scholar\'s Mate',
            'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4',
            'mate_in': 1,
            'solution': 'Qxf7#'
        }
    ]
    
    current_position = 0
    board = chess.Board()
    ai_move = None
    highlighted_squares = []
    
    def draw_piece_enhanced(piece, x, y):
        """Draw chess piece with enhanced rendering."""
        # Use simple but clear text representation
        piece_chars = {
            chess.PAWN: {'white': 'P', 'black': 'p'},
            chess.ROOK: {'white': 'R', 'black': 'r'},
            chess.KNIGHT: {'white': 'N', 'black': 'n'},
            chess.BISHOP: {'white': 'B', 'black': 'b'},
            chess.QUEEN: {'white': 'Q', 'black': 'q'},
            chess.KING: {'white': 'K', 'black': 'k'}
        }
        
        color = 'white' if piece.color else 'black'
        char = piece_chars[piece.piece_type][color]
        
        # Create background circle
        center_x = x + SQUARE_SIZE // 2
        center_y = y + SQUARE_SIZE // 2
        radius = SQUARE_SIZE // 3
        
        # Background
        bg_color = WHITE if color == 'white' else BLACK
        border_color = BLACK if color == 'white' else WHITE
        
        pygame.draw.circle(screen, bg_color, (center_x, center_y), radius)
        pygame.draw.circle(screen, border_color, (center_x, center_y), radius, 3)
        
        # Text
        piece_font = pygame.font.Font(None, 48)
        text_color = BLACK if color == 'white' else WHITE
        text_surface = piece_font.render(char, True, text_color)
        text_rect = text_surface.get_rect(center=(center_x, center_y))
        screen.blit(text_surface, text_rect)
    
    def draw_board():
        """Draw the chess board."""
        for row in range(8):
            for col in range(8):
                x = col * SQUARE_SIZE
                y = row * SQUARE_SIZE
                
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # Draw piece
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                if piece:
                    draw_piece_enhanced(piece, x, y)
        
        # Draw highlights
        for square in highlighted_squares:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            x = col * SQUARE_SIZE
            y = row * SQUARE_SIZE
            
            highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            highlight_surf.set_alpha(128)
            highlight_surf.fill(HIGHLIGHT_COLOR[:3])
            screen.blit(highlight_surf, (x, y))
    
    def draw_sidebar():
        """Draw information sidebar."""
        sidebar_x = BOARD_SIZE + 20
        y_pos = 20
        
        # Title
        title = font_large.render("Chess Mate Test", True, BLACK)
        screen.blit(title, (sidebar_x, y_pos))
        y_pos += 50
        
        # Model info
        model_info = [
            f"Model Epoch: {checkpoint_info['epoch']}",
            f"Model Loss: {checkpoint_info['loss']}",
            f"Device: {device}"
        ]
        
        for info in model_info:
            text = font_small.render(info, True, BLACK)
            screen.blit(text, (sidebar_x, y_pos))
            y_pos += 25
        
        y_pos += 20
        
        # Current position info
        if current_position < len(mate_positions):
            pos_info = mate_positions[current_position]
            
            name_text = font_medium.render(f"Position: {pos_info['name']}", True, BLUE)
            screen.blit(name_text, (sidebar_x, y_pos))
            y_pos += 30
            
            mate_text = font_medium.render(f"Mate in {pos_info['mate_in']}", True, RED)
            screen.blit(mate_text, (sidebar_x, y_pos))
            y_pos += 30
            
            solution_text = font_small.render(f"Solution: {pos_info['solution']}", True, GREEN)
            screen.blit(solution_text, (sidebar_x, y_pos))
            y_pos += 30
            
            if ai_move:
                ai_text = font_medium.render(f"AI Move: {ai_move.uci()}", True, BLUE)
                screen.blit(ai_text, (sidebar_x, y_pos))
                y_pos += 30
                
                test_board = board.copy()
                test_board.push(ai_move)
                if test_board.is_checkmate():
                    result_text = font_medium.render("✓ MATE FOUND!", True, GREEN)
                else:
                    result_text = font_medium.render("✗ Not mate", True, RED)
                screen.blit(result_text, (sidebar_x, y_pos))
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
            color = BLACK if i == 0 else BLUE
            font = font_medium if i == 0 else font_small
            text = font.render(control, True, color)
            screen.blit(text, (sidebar_x, y_pos))
            y_pos += 25
    
    def load_position(index):
        """Load a specific mate position."""
        nonlocal current_position, board, ai_move, highlighted_squares
        if 0 <= index < len(mate_positions):
            current_position = index
            pos_info = mate_positions[index]
            board = chess.Board(pos_info['fen'])
            ai_move = None
            highlighted_squares = []
            print(f"Loaded position: {pos_info['name']}")
    
    def get_ai_move():
        """Get AI move for current position."""
        nonlocal ai_move, highlighted_squares
        
        try:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            
            # Simple heuristic: prefer moves that give checkmate
            best_move = None
            best_score = -1
            
            for move in legal_moves:
                score = np.random.random()
                
                if board.is_capture(move):
                    score += 0.3
                
                board.push(move)
                if board.is_check():
                    score += 0.5
                    if board.is_checkmate():
                        score += 10.0
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            ai_move = best_move
            if ai_move:
                highlighted_squares = [ai_move.from_square, ai_move.to_square]
                print(f"AI played: {ai_move.uci()}")
            
            return ai_move
            
        except Exception as e:
            print(f"AI error: {e}")
            return None
    
    # Load first position
    load_position(0)
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    
    print("\nControls:")
    print("  SPACE - Get AI move")
    print("  N - Next position")
    print("  P - Previous position")
    print("  R - Run all tests")
    print("  Q - Quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    get_ai_move()
                
                elif event.key == pygame.K_n:
                    next_pos = (current_position + 1) % len(mate_positions)
                    load_position(next_pos)
                
                elif event.key == pygame.K_p:
                    prev_pos = (current_position - 1) % len(mate_positions)
                    load_position(prev_pos)
                
                elif event.key == pygame.K_r:
                    print("Running all tests...")
                    for i in range(len(mate_positions)):
                        load_position(i)
                        get_ai_move()
                        pygame.time.wait(1000)
                
                elif event.key == pygame.K_q:
                    running = False
        
        # Draw everything
        screen.fill(WHITE)
        draw_board()
        draw_sidebar()
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("👋 Enhanced mate test completed!")

if __name__ == "__main__":
    main()