"""
Cricket Transformer Configuration
================================

Configuration settings for different training scenarios and model sizes.
"""

# Base configuration
BASE_CONFIG = {
    # Data dimensions (from your analysis)
    'ball_vector_dim': 22,
    'context_dim': 20,
    'vocab_size': 24,  # Will be updated from actual vocabulary
    
    # Model architecture
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'max_target_length': 15,
    
    # Training parameters
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'batch_size': 16,  # Optimized for MPS
    'max_history_length': 8,
    
    # Training schedule
    'num_epochs': 100,
    'teacher_forcing_epochs': 50,
    'save_every': 10,
    'validate_every': 1,
    
    # Generation parameters
    'temperature': 0.8,
    'top_k': 10,
    'top_p': 0.9
}

# Small model for testing/development
SMALL_CONFIG = {
    **BASE_CONFIG,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'dim_feedforward': 1024,
    'batch_size': 128,
    'num_epochs': 20,
    'teacher_forcing_epochs': 10
}

# Large model for production
LARGE_CONFIG = {
    **BASE_CONFIG,
    'd_model': 768,
    'nhead': 12,
    'num_encoder_layers': 8,
    'num_decoder_layers': 8,
    'dim_feedforward': 3072,
    'batch_size': 8,  # Smaller batch for larger model
    'learning_rate': 5e-5,  # Lower learning rate for stability
}

# Fast training configuration
FAST_CONFIG = {
    **BASE_CONFIG,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dim_feedforward': 1024,
    'batch_size': 32,
    'num_epochs': 50,
    'teacher_forcing_epochs': 25,
    'learning_rate': 2e-4
}

# Configuration for different scenarios
CONFIGS = {
    'base': BASE_CONFIG,
    'small': SMALL_CONFIG,
    'large': LARGE_CONFIG,
    'fast': FAST_CONFIG
}

def get_config(config_name: str = 'base'):
    """Get configuration by name"""
    if config_name not in CONFIGS:
        print(f"Warning: Config '{config_name}' not found. Using 'base' config.")
        config_name = 'base'
    
    return CONFIGS[config_name].copy()

def print_config(config: dict):
    """Print configuration in a readable format"""
    print("Configuration:")
    print("-" * 40)
    
    sections = {
        'Model Architecture': ['d_model', 'nhead', 'num_encoder_layers', 'num_decoder_layers', 
                              'dim_feedforward', 'dropout'],
        'Data': ['ball_vector_dim', 'context_dim', 'vocab_size', 'max_history_length', 
                'max_target_length'],
        'Training': ['learning_rate', 'weight_decay', 'label_smoothing', 'batch_size'],
        'Schedule': ['num_epochs', 'teacher_forcing_epochs', 'save_every', 'validate_every'],
        'Generation': ['temperature', 'top_k', 'top_p']
    }
    
    for section, keys in sections.items():
        print(f"\n{section}:")
        for key in keys:
            if key in config:
                print(f"  {key}: {config[key]}")

def estimate_parameters(config: dict) -> int:
    """Estimate number of parameters for given configuration"""
    
    d_model = config['d_model']
    vocab_size = config['vocab_size']
    ball_dim = config['ball_vector_dim']
    context_dim = config['context_dim']
    dim_ff = config['dim_feedforward']
    n_enc = config['num_encoder_layers']
    n_dec = config['num_decoder_layers']
    
    # Encoder parameters
    encoder_params = (
        ball_dim * d_model +  # Input projection
        n_enc * (
            4 * d_model * d_model +  # Self-attention (Q, K, V, O)
            2 * d_model * dim_ff +   # Feed-forward
            4 * d_model              # Layer norms and biases
        )
    )
    
    # Decoder parameters
    decoder_params = (
        vocab_size * d_model +     # Token embedding
        context_dim * d_model +    # Context projection
        n_dec * (
            8 * d_model * d_model +  # Self + cross attention
            2 * d_model * dim_ff +   # Feed-forward
            6 * d_model              # Layer norms and biases
        ) +
        d_model * vocab_size       # Output projection
    )
    
    total_params = encoder_params + decoder_params
    return int(total_params)

def estimate_memory(config: dict) -> dict:
    """Estimate memory requirements"""
    
    batch_size = config['batch_size']
    max_seq_len = config['max_history_length']
    ball_dim = config['ball_vector_dim']
    context_dim = config['context_dim']
    max_target_len = config['max_target_length']
    
    # Memory per batch (in MB)
    encoder_memory = batch_size * max_seq_len * ball_dim * 4 / (1024**2)
    context_memory = batch_size * context_dim * 4 / (1024**2)
    target_memory = batch_size * max_target_len * 4 / (1024**2)
    
    total_batch_memory = encoder_memory + context_memory + target_memory
    
    # Estimate total GPU memory with gradients and activations
    total_gpu_memory = total_batch_memory * 4  # Rough estimate
    
    return {
        'batch_memory_mb': total_batch_memory,
        'estimated_gpu_memory_mb': total_gpu_memory,
        'parameters': estimate_parameters(config)
    }

if __name__ == "__main__":
    # Test configurations
    for name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"CONFIG: {name.upper()}")
        print(f"{'='*60}")
        
        print_config(config)
        
        memory_info = estimate_memory(config)
        print(f"\nEstimated Requirements:")
        print(f"  Parameters: {memory_info['parameters']:,}")
        print(f"  Batch Memory: {memory_info['batch_memory_mb']:.1f} MB")
        print(f"  GPU Memory: {memory_info['estimated_gpu_memory_mb']:.1f} MB")