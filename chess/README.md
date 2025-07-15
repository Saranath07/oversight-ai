# Chess AI - Transformer-based Chess Engine

A decoder-only transformer model trained to play chess, inspired by AlphaZero's policy network architecture. This project demonstrates how modern transformer architectures can be applied to chess move prediction and tactical pattern recognition.

## 🏗️ Architecture Overview

### Model Architecture: Decoder-Only Transformer

The chess AI uses a **decoder-only transformer architecture** similar to GPT, specifically designed for sequential move prediction:

```
Input Sequence → Token Embedding → Positional Encoding → Transformer Blocks → Policy Head → Move Probabilities
```

#### Key Components:

1. **Token Embedding Layer**
   - Vocabulary size: 2,072 tokens
   - Embedding dimension: 512
   - Converts chess moves (UCI format) to dense vectors

2. **Positional Encoding**
   - Sinusoidal positional embeddings
   - Maximum sequence length: 512 tokens
   - Helps model understand move order/sequence

3. **Transformer Decoder Blocks** (6 layers)
   - **Multi-Head Self-Attention**: 8 attention heads
   - **Feed-Forward Network**: 2,048 hidden units
   - **Layer Normalization** and **Residual Connections**
   - **Causal Masking**: Prevents looking at future moves

4. **Policy Head**
   - Linear projection to 4,672 possible moves
   - Outputs probability distribution over all legal chess moves
   - AlphaZero-style policy network

### Model Parameters
```python
{
    'vocab_size': 2072,        # Chess move vocabulary
    'd_model': 512,            # Model dimension
    'n_heads': 8,              # Attention heads
    'n_layers': 6,             # Transformer layers
    'd_ff': 2048,              # Feed-forward dimension
    'max_seq_len': 512,        # Maximum sequence length
    'dropout': 0.1,            # Dropout rate
    'policy_size': 4672        # Total possible moves
}
```

**Total Parameters**: ~22M parameters

## 📥 Input Format

### Input: Game History Sequence
The model takes a sequence of chess moves in UCI (Universal Chess Interface) notation:

```python
# Example input sequence
input_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
```

### Tokenization Process:
1. **Move Encoding**: Each UCI move is converted to a token ID
2. **Sequence Building**: Moves are concatenated chronologically
3. **Padding/Truncation**: Sequences are padded to fixed length (100-512 tokens)
4. **Special Tokens**: Start token (1) and padding tokens (0)

### Input Tensor Shape:
```python
input_ids: torch.Tensor  # Shape: [batch_size, sequence_length]
# Example: [1, 100] for batch_size=1, max_length=100
```

## 📤 Output Format

### Output: Policy Logits
The model outputs a probability distribution over all possible chess moves:

```python
policy_logits: torch.Tensor  # Shape: [batch_size, 4672]
```

### Move Selection Process:
1. **Logits to Probabilities**: Apply softmax to get move probabilities
2. **Legal Move Filtering**: Mask illegal moves for current position
3. **Temperature Sampling**: Apply temperature for exploration vs exploitation
4. **Move Selection**: Sample from filtered probability distribution

### Policy Index Mapping:
- **4,672 total moves**: Covers all possible chess moves
- **Move Encoding**: Uses gym-chess environment for move↔index mapping
- **Legal Filtering**: Only legal moves for current position are considered

## 🎯 Training Details

### Training Data:
- **Source**: Lichess Elite Database (high-rated games)
- **Format**: PGN files processed into move sequences
- **Size**: Thousands of professional chess games
- **Preprocessing**: Games converted to UCI move sequences with policy targets

### Training Process:
- **Loss Function**: Cross-entropy loss with illegal move penalty
- **Optimizer**: AdamW with cosine annealing scheduler
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Training Device**: GPU/MPS/CPU auto-detection

### Checkpoints:
- `best_model.pt`: Best validation loss model
- `checkpoint_epoch_X.pt`: Regular training checkpoints

## 🚀 Usage

### 1. Install Dependencies
```bash
cd chess
pip install -r pygame_requirements.txt
```

### 2. Run Mate Test Demo
```bash
python run_mate_test.py
```

### 3. Demo Controls
- **SPACE**: Get AI move for current position
- **N**: Next test position
- **P**: Previous test position  
- **R**: Run all tests automatically
- **Q**: Quit

### 4. Test Positions
The demo includes several tactical positions:
- **Mate in 1**: Back rank mate, Queen mate, Scholar's mate
- **Mate in 2**: Queen+Rook combinations
- **Tactical Puzzles**: Various checkmate patterns

## 🧠 Model Capabilities

### What the Model Learns:
1. **Move Patterns**: Common opening, middlegame, and endgame moves
2. **Tactical Motifs**: Pins, forks, skewers, discovered attacks
3. **Positional Understanding**: Piece coordination and board control
4. **Endgame Technique**: Basic checkmate patterns

### Current Performance:
- ✅ **Mate in 1**: Successfully finds simple checkmates
- ✅ **Basic Tactics**: Recognizes common tactical patterns
- ⚠️ **Complex Positions**: Limited by training data and model size
- ⚠️ **Strategic Planning**: Focuses more on tactics than long-term strategy

## 🔧 Technical Implementation

### Key Files:
- `model/chess_transformer.py`: Core transformer architecture
- `training/trainer.py`: Training loop and checkpointing
- `run_mate_test.py`: Interactive demo with visual board
- `data_processing/`: PGN processing and tokenization

### Model Loading:
```python
from model.chess_transformer import ChessTransformer

model = ChessTransformer(**model_config)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inference:
```python
# Encode game history
input_ids = encode_game_history(move_sequence)

# Get move predictions
with torch.no_grad():
    policy_logits = model(input_ids)
    
# Apply legal move filtering and sample
move_probs = torch.softmax(policy_logits, dim=-1)
selected_move = sample_legal_move(move_probs, legal_moves)
```

## 🎮 Demo Features

### Visual Board:
- **Clear Piece Rendering**: Enhanced piece symbols with backgrounds
- **Move Highlighting**: Shows AI's selected moves
- **Position Information**: Displays current test position details

### AI Testing:
- **Tactical Positions**: Pre-loaded mate-in-X positions
- **Real-time Analysis**: Watch AI think and select moves
- **Success Tracking**: See which positions AI solves correctly

### Interactive Controls:
- **Step-by-step**: Analyze individual positions
- **Batch Testing**: Run all tests automatically
- **Visual Feedback**: Immediate results with color coding

## 🔮 Future Improvements

### Model Enhancements:
- **Larger Model**: Increase parameters for better performance
- **Value Head**: Add position evaluation alongside move prediction
- **MCTS Integration**: Combine with Monte Carlo Tree Search
- **Self-Play Training**: Generate training data through self-play

### Technical Upgrades:
- **Better Tokenization**: More sophisticated move encoding
- **Attention Visualization**: Show what the model focuses on
- **Opening Book**: Integration with chess opening databases
- **Endgame Tablebase**: Perfect endgame play integration

## 📊 Model Statistics

```
Architecture: Decoder-Only Transformer
Parameters: ~25M
Training Time: ~10 epochs
Vocabulary: 2,072 move tokens
Policy Space: 4,672 possible moves
Context Length: 512 tokens
Attention Heads: 8
Transformer Layers: 6
```

## 🏆 Acknowledgments

- **AlphaZero**: Inspiration for policy network architecture
- **Lichess**: High-quality chess game database
- **python-chess**: Excellent chess library for Python
- **PyTorch**: Deep learning framework
- **Pygame**: Graphics and user interface

---

*This chess AI demonstrates the power of transformer architectures applied to strategic games. While not yet at grandmaster level, it shows promising tactical pattern recognition and provides a solid foundation for further development.*