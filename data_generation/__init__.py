"""
OverSight AI Data Generation Package
===================================

This package provides comprehensive data processing utilities for the OverSight AI
cricket ball-by-ball prediction transformer model.

Components:
- data_processor: Convert raw cricket data to training sequences
- data_loader: PyTorch-compatible data loading utilities  
- data_analyzer: Comprehensive data analysis and visualization
- generate_data: Main pipeline orchestration script
- test_pipeline: Validation and testing utilities

Usage:
    from data_generation import CricketDataProcessor, CricketDataLoader
    
    # Process raw data
    processor = CricketDataProcessor(json_dir, player_stats_file)
    processor.save_processed_data(output_dir)
    
    # Load for training
    data_loader = CricketDataLoader(processed_dir, batch_size=32)
    for batch in data_loader.train_loader:
        # Train your model
        pass
"""

from .data_processor import CricketDataProcessor, BallEvent, OverContext
from .data_loader import CricketDataLoader, CricketDataset, create_sample_batch, decode_predictions
from .data_analyzer import CricketDataAnalyzer

__version__ = "1.0.0"
__author__ = "OverSight AI Team"

__all__ = [
    'CricketDataProcessor',
    'BallEvent', 
    'OverContext',
    'CricketDataLoader',
    'CricketDataset',
    'create_sample_batch',
    'decode_predictions',
    'CricketDataAnalyzer'
]