# Cricket Transformer Training

A complete training pipeline for the cricket ball-by-ball prediction transformer model, optimized for Apple Silicon Macs with MPS support.

## 🏗️ Architecture Overview

The Cricket Transformer is an encoder-decoder model designed for cricket ball-by-ball prediction:

- **Encoder**: Processes match history sequences (variable length, up to 128 balls)
- **Decoder**: Generates next over predictions using current context
- **Input Features**: 18-dimensional ball vectors + 20-dimensional context vectors
- **Output**: Sequence of cricket outcome tokens (24 vocabulary size)

### Model Components

1. **CricketTransformerEncoder**
   - Processes match history with self-attention
   - Positional encoding for sequence order
   - Variable length sequence handling with masking

2. **CricketTransformerDecoder**
   - Generates over predictions with causal masking
   - Cross-attention to encoder outputs
   - Context injection for current match state

3. **Training Modes**
   - **Teacher Forcing**: Uses ground truth tokens during training
   - **Autonomous Training**: Uses model's own predictions

## 📊 Data Requirements

Based on your tensor analysis:
- **Training Sequences**: 37,397
- **Ball Vector Dimension**: 18
- **Context Vector Dimension**: 20
- **Vocabulary Size**: 24 tokens
- **Max Sequence Length**: 128 balls

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Verify MPS availability (Mac only)
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

### 2. Test Model Architecture

```bash
# Test model creation and basic functionality
python test_model.py
```

### 3. Start Training

```bash
# Train with default configuration
python trainer.py

# Or use a specific configuration
python -c "
from trainer import main
from config import get_config
import sys
sys.argv = ['trainer.py', '--config', 'small']  # or 'large', 'fast'
main()
"
```

## ⚙️ Configuration Options

### Available Configurations

| Config | Model Size | Parameters | Use Case |
|--------|------------|------------|----------|
| `small` | 256d, 4 layers | ~2M | Testing/Development |
| `base` | 512d, 6 layers | ~10M | Production Training |
| `large` | 768d, 8 layers | ~25M | Maximum Performance |
| `fast` | 256d, 3 layers | ~1.5M | Quick Experiments |

### Key Parameters

```python
# Model Architecture
'd_model': 512,              # Hidden dimension
'nhead': 8,                  # Attention heads
'num_encoder_layers': 6,     # Encoder depth
'num_decoder_layers': 6,     # Decoder depth

# Training
'batch_size': 16,            # Optimized for MPS
'learning_rate': 1e-4,       # Learning rate
'num_epochs': 100,           # Total epochs
'teacher_forcing_epochs': 50, # TF epochs before autonomous
```

## 🎯 Training Process

### Phase 1: Teacher Forcing (Epochs 1-50)
- Model learns from ground truth sequences
- Stable training with known targets
- Builds foundational understanding

### Phase 2: Autonomous Training (Epochs 51-100)
- Model uses its own predictions
- Learns to handle prediction errors
- Improves real-world performance

### Monitoring

Training metrics are logged to TensorBoard:

```bash
# View training progress
tensorboard --logdir runs/
```

Key metrics tracked:
- Training/Validation Loss
- Perplexity
- Token Accuracy
- Exact Match Accuracy

## 🖥️ Apple Silicon Optimization

### MPS (Metal Performance Shaders) Support

The trainer automatically detects and uses Apple's Neural Processing Unit:

```python
if torch.backends.mps.is_available():
    device = torch.device('mps')
```

### MPS-Specific Optimizations

1. **Tensor Compatibility**: Automatic float64 → float32 conversion
2. **Memory Management**: Smaller batch sizes for stability
3. **Error Handling**: Graceful fallback for MPS-specific issues
4. **Contiguous Tensors**: Ensures optimal memory layout

### Performance Tips

- **Batch Size**: Start with 16, adjust based on memory
- **Model Size**: Use 'base' config for good performance/memory balance
- **Memory**: Monitor Activity Monitor for memory usage
- **Temperature**: Keep MPS cool for sustained performance

## 📈 Expected Performance

Based on your data analysis:

### Training Feasibility: ✅ EXCELLENT

- **Dataset Size**: 37K+ sequences (substantial)
- **Memory Usage**: <100MB per batch (manageable)
- **Training Time**: 2-4 hours on Apple Silicon
- **Parameters**: 10-50M (reasonable size)

### Accuracy Expectations

- **Token Accuracy**: 70-85% (individual ball predictions)
- **Exact Match**: 20-40% (complete over predictions)
- **Perplexity**: <3.0 (good language modeling)

## 🔧 Troubleshooting

### Common Issues

1. **MPS Memory Errors**
   ```bash
   # Reduce batch size in config
   'batch_size': 8  # Instead of 16
   ```

2. **Data Not Found**
   ```bash
   # Ensure data generation completed
   ls ../data_generation/processed/
   ```

3. **Import Errors**
   ```bash
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Performance Issues

1. **Slow Training**: Verify MPS is being used
2. **High Memory**: Reduce batch size or model size
3. **Poor Convergence**: Adjust learning rate or increase teacher forcing epochs

## 📁 File Structure

```
training/
├── cricket_transformer.py    # Model architecture
├── trainer.py               # Training script
├── test_model.py           # Testing utilities
├── config.py               # Configuration management
├── requirements.txt        # Dependencies
├── README.md              # This file
├── checkpoints/           # Saved models (created during training)
└── runs/                 # TensorBoard logs (created during training)
```

## 🎮 Usage Examples

### Basic Training

```python
from trainer import CricketTrainer
from cricket_transformer import create_cricket_transformer
from data_generation.data_loader import CricketDataLoader
from config import get_config

# Load configuration
config = get_config('base')

# Create data loader
data_loader = CricketDataLoader('../data_generation/processed')

# Create model
model = create_cricket_transformer(config)

# Train
trainer = CricketTrainer(model, data_loader, config)
trainer.train(num_epochs=50)
```

### Custom Configuration

```python
# Custom config
config = {
    'd_model': 384,
    'nhead': 6,
    'num_encoder_layers': 5,
    'batch_size': 12,
    'learning_rate': 5e-5
}

# Train with custom config
trainer = CricketTrainer(model, data_loader, config)
trainer.train(num_epochs=75, teacher_forcing_epochs=40)
```

### Generation/Inference

```python
# Load trained model
model = create_cricket_transformer(config)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate predictions
predictions = model.generate(
    histories=match_history,
    contexts=current_context,
    start_token_id=vocabulary['<START>'],
    end_token_id=vocabulary['<END>'],
    temperature=0.8
)
```

## 📊 Monitoring Training

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should follow training loss
3. **Perplexity**: Lower is better (<3.0 target)
4. **Token Accuracy**: Should increase over time
5. **Learning Rate**: Automatically adjusted by scheduler

### Early Stopping Indicators

- Validation loss stops improving for 10+ epochs
- Training loss much lower than validation (overfitting)
- Token accuracy plateaus below 60%

## 🎯 Next Steps

After training:

1. **Evaluate**: Test on held-out data
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Deploy**: Create inference pipeline
4. **Iterate**: Collect more data and retrain

## 🤝 Contributing

To extend the model:

1. **New Features**: Add to ball vector or context dimensions
2. **Architecture**: Modify transformer components
3. **Training**: Implement new training strategies
4. **Evaluation**: Add cricket-specific metrics

---

**Ready to train your cricket prediction model!** 🏏

Run `python test_model.py` to verify everything works, then `python trainer.py` to start training.