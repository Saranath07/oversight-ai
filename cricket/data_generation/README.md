# OverSight AI - Data Generation Pipeline

This directory contains the complete data generation pipeline for the OverSight AI cricket ball-by-ball prediction transformer model.

## Overview

The OverSight AI project aims to build a state-of-the-art deep learning model that predicts ball-by-ball outcomes for cricket overs using a Sequence-to-Sequence (Seq2Seq) approach.

### Problem Formulation

- **Input (X)**: 
  - High-Fidelity Match History: Sequence of ball vectors representing the match story up to a certain point
  - Current Over Context: Rich feature vector capturing game state, player stats, and match conditions

- **Output (Y)**: 
  - Predicted sequence of 6+ ball outcomes for the next over
  - Vocabulary: `['0', '1', '2', '3', '4', '6', 'W', 'wd', 'nb', 'lb', 'b', ...]`

## Data Structure

### Input Data
- **Match JSON Files**: Ball-by-ball data in Cricsheet format
- **Player Statistics CSV**: Comprehensive player performance statistics

### Generated Training Data
- **Match Histories**: Sequences of ball vectors (each ball = 18-dimensional vector)
- **Context Vectors**: Rich over context (19-dimensional vector)
- **Target Sequences**: Ball outcome token sequences for next over

## Pipeline Components

### 1. Data Processor (`data_processor.py`)
Converts raw cricket data into training sequences:

```python
from data_processor import CricketDataProcessor

processor = CricketDataProcessor(
    json_dir="data/ipl_json",
    player_stats_file="data/comprehensive_player_stats.csv"
)
processor.save_processed_data("processed/")
```

**Key Features:**
- Ball-by-ball event parsing
- Player statistics integration
- Comprehensive feature engineering
- Vocabulary creation and token mapping
- Sequence generation for transformer training

### 2. Data Loader (`data_loader.py`)
PyTorch-compatible data loading utilities:

```python
from data_loader import CricketDataLoader

data_loader = CricketDataLoader(
    data_dir="processed/",
    batch_size=32,
    max_history_length=500
)

# Use in training
for batch in data_loader.train_loader:
    histories = batch['histories']      # [batch_size, seq_len, 18]
    contexts = batch['contexts']        # [batch_size, 19]
    targets = batch['target_outputs']   # [batch_size, 6+]
```

**Key Features:**
- Variable-length sequence handling
- Automatic padding and masking
- Train/validation splits
- Batch collation for transformer training

### 3. Data Analyzer (`data_analyzer.py`)
Comprehensive data analysis and visualization:

```python
from data_analyzer import CricketDataAnalyzer

analyzer = CricketDataAnalyzer(
    json_dir="data/ipl_json",
    player_stats_file="data/comprehensive_player_stats.csv",
    processed_dir="processed/"
)
analyzer.generate_comprehensive_report()
```

**Analysis Features:**
- Raw data statistics
- Ball outcome distributions
- Over pattern analysis
- Player performance insights
- Training data validation
- Visualization generation

### 4. Main Pipeline (`generate_data.py`)
Orchestrates the complete data generation process:

```bash
python generate_data.py --json_dir data/ipl_json --player_stats data/comprehensive_player_stats.csv
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure you have:
- `data/ipl_json/`: Directory with match JSON files
- `data/comprehensive_player_stats.csv`: Player statistics file

### 3. Generate Training Data
```bash
cd data_generation
python generate_data.py
```

### 4. Verify Results
The pipeline will create:
- `processed/`: Training data files
- `analysis/`: Analysis results and visualizations

## Data Format Details

### Ball Vector (18 dimensions)
```
[over, ball, runs_batter, runs_extras, runs_total, cumulative_score, 
 cumulative_wickets, is_wicket, is_wide, is_noball, is_legbye, is_bye,
 batter_avg, batter_sr, batter_runs_norm, bowler_avg, bowler_sr, bowler_wickets_norm]
```

### Context Vector (19 dimensions)
```
[innings, over_number, score_norm, wickets, balls_remaining_norm, run_rate,
 required_rate, venue_hash, season_norm, striker_avg, striker_sr, striker_runs_norm,
 non_striker_avg, non_striker_sr, non_striker_runs_norm, bowler_avg, bowler_sr,
 bowler_wickets_norm, h2h_placeholder]
```

### Target Vocabulary
- **Runs**: `'0', '1', '2', '3', '4', '5', '6'`
- **Wicket**: `'W'`
- **Extras**: `'wd', 'nb', 'lb', 'b'`
- **Extras with runs**: `'wd1', 'wd2', 'nb1', 'nb4', 'nb6'`, etc.
- **Special tokens**: `'<PAD>', '<START>', '<END>'`

## Output Files

### Processed Data Directory
```
processed/
├── match_histories.pkl    # List of ball vector sequences
├── contexts.pkl          # List of context vectors  
├── targets.pkl           # List of target token sequences
├── vocabulary.json       # Token to index mapping
└── metadata.json         # Dataset statistics
```

### Analysis Directory
```
analysis/
├── ball_outcomes.png     # Ball outcome distribution
├── player_stats.png      # Player statistics distributions
└── analysis_report.txt   # Comprehensive analysis report
```

## Configuration Options

### Command Line Arguments
```bash
python generate_data.py \
    --json_dir data/ipl_json \
    --player_stats data/comprehensive_player_stats.csv \
    --output_dir processed \
    --analysis_dir analysis \
    --batch_size 32 \
    --max_history_length 500 \
    --skip_processing \
    --skip_analysis
```

### Key Parameters
- `max_history_length`: Maximum balls in match history (default: 500)
- `batch_size`: Training batch size (default: 32)
- `train_split`: Train/validation split ratio (default: 0.8)

## Data Statistics

### Expected Dataset Size
- **Matches**: ~200 IPL matches
- **Training Sequences**: ~4,000-6,000 sequences
- **Vocabulary Size**: ~25 tokens
- **Average Sequence Length**: ~100-200 balls
- **File Sizes**: ~50-100 MB total

### Performance Metrics
- **Processing Time**: ~2-5 minutes for full dataset
- **Memory Usage**: ~1-2 GB during processing
- **Storage**: ~100 MB for processed data

## Integration with Transformer Model

The generated data is designed for transformer training:

```python
# Pseudo-code for transformer training
for batch in data_loader.train_loader:
    # Encoder input: match history
    encoder_input = batch['histories']        # [B, seq_len, 18]
    encoder_mask = batch['history_mask']      # [B, seq_len]
    
    # Decoder context: over context + target input
    decoder_context = batch['contexts']       # [B, 19]
    decoder_input = batch['target_inputs']    # [B, target_len]
    decoder_mask = batch['target_mask']       # [B, target_len]
    
    # Target: next over outcomes
    targets = batch['target_outputs']         # [B, target_len]
    
    # Forward pass
    predictions = model(encoder_input, encoder_mask, 
                       decoder_context, decoder_input, decoder_mask)
    loss = criterion(predictions, targets)
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `max_history_length` or process in smaller batches
2. **Missing Files**: Ensure all JSON files and CSV are in correct locations
3. **Empty Sequences**: Check JSON file format and data quality
4. **Vocabulary Issues**: Verify ball outcome parsing logic

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with small subset
processor = CricketDataProcessor(json_dir, player_stats)
sequences = processor.process_match(single_match_file)
```

## Next Steps

1. **Model Architecture**: Implement transformer encoder-decoder
2. **Training Loop**: Create training script with the data loader
3. **Evaluation**: Implement metrics for sequence prediction
4. **Inference**: Build prediction pipeline for new matches

## Contributing

When adding new features:
1. Update the corresponding module
2. Add tests for new functionality
3. Update documentation
4. Regenerate sample data for validation

## License

This project is part of the OverSight AI cricket prediction system.