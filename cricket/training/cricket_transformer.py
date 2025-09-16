"""
Cricket Transformer Model
========================

A transformer-based encoder-decoder model for cricket ball-by-ball prediction.
Uses match history as encoder input and current over context for decoder initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CricketTransformerEncoder(nn.Module):
    """Transformer encoder for match history processing"""
    
    def __init__(self, 
                 input_dim: int = 22,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """
        Args:
            src: [batch_size, seq_len, input_dim] - Match history sequences
            src_mask: [batch_size, seq_len] - Padding mask
        """
        # Project input to model dimension
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        src = self.dropout(src)
        
        # Handle masking based on device compatibility
        if src_mask is not None:
            # Check if we're on MPS (which has issues with nested tensors)
            if src.device.type == 'mps':
                # For MPS: Apply mask directly to embeddings
                mask_expanded = src_mask.unsqueeze(-1).expand_as(src)
                src = src * mask_expanded
                # Use transformer without key padding mask
                output = self.transformer_encoder(src)
            else:
                # For CUDA/CPU: Use proper attention masking
                attention_mask = (src_mask == 0)  # True = ignore
                output = self.transformer_encoder(src, src_key_padding_mask=attention_mask)
        else:
            # No masking needed
            output = self.transformer_encoder(src)
        
        return output

class CricketTransformerDecoder(nn.Module):
    """Transformer decoder for over prediction"""
    
    def __init__(self,
                 vocab_size: int = 24,
                 context_dim: int = 20,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_target_length: int = 15):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_target_length = max_target_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Context projection
        self.context_projection = nn.Linear(context_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_target_length)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                tgt: torch.Tensor,
                memory: torch.Tensor,
                context: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None):
        """
        Args:
            tgt: [batch_size, tgt_len] - Target token indices
            memory: [batch_size, src_len, d_model] - Encoder output
            context: [batch_size, context_dim] - Current over context
            tgt_mask: [batch_size, tgt_len] - Target padding mask
            memory_mask: [batch_size, src_len] - Memory padding mask
        """
        batch_size, tgt_len = tgt.shape
        
        # Token embeddings
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Add context information to first token
        context_emb = self.context_projection(context).unsqueeze(1)  # [batch, 1, d_model]
        tgt_emb[:, 0:1, :] = tgt_emb[:, 0:1, :] + context_emb
        
        tgt_emb = self.dropout(tgt_emb)
        
        # Create causal mask for target sequence
        causal_mask = self._generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Handle masking based on device compatibility
        if tgt_emb.device.type == 'mps':
            # For MPS: Apply masks directly to embeddings to avoid nested tensor issues
            if tgt_mask is not None:
                mask_expanded = tgt_mask.unsqueeze(-1).expand_as(tgt_emb)
                tgt_emb = tgt_emb * mask_expanded
            
            if memory_mask is not None:
                memory_mask_expanded = memory_mask.unsqueeze(-1).expand_as(memory)
                memory = memory * memory_mask_expanded
            
            # Use transformer decoder without key padding masks
            output = self.transformer_decoder(
                tgt_emb,
                memory,
                tgt_mask=causal_mask
            )
        else:
            # For CUDA/CPU: Use proper attention masking
            tgt_key_padding_mask = (tgt_mask == 0) if tgt_mask is not None else None
            memory_key_padding_mask = (memory_mask == 0) if memory_mask is not None else None
            
            output = self.transformer_decoder(
                tgt_emb,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz: int):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class CricketTransformer(nn.Module):
    """Complete Cricket Transformer Model"""
    
    def __init__(self,
                 # Data dimensions
                 ball_vector_dim: int = 18,
                 context_dim: int = 20,
                 vocab_size: int = 24,
                 
                 # Model dimensions
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_target_length: int = 15):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_target_length = max_target_length
        
        # Encoder for match history
        self.encoder = CricketTransformerEncoder(
            input_dim=ball_vector_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Decoder for over prediction
        self.decoder = CricketTransformerDecoder(
            vocab_size=vocab_size,
            context_dim=context_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_target_length=max_target_length
        )
    
    def forward(self, 
                histories: torch.Tensor,
                contexts: torch.Tensor,
                target_inputs: torch.Tensor,
                history_mask: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for training with teacher forcing
        
        Args:
            histories: [batch_size, seq_len, ball_vector_dim] - Match histories
            contexts: [batch_size, context_dim] - Current over contexts
            target_inputs: [batch_size, tgt_len] - Target input tokens (with <START>)
            history_mask: [batch_size, seq_len] - History padding mask
            target_mask: [batch_size, tgt_len] - Target padding mask
        """
        
        # Encode match history
        memory = self.encoder(histories, history_mask)
        
        # Decode target sequence
        logits = self.decoder(
            target_inputs,
            memory,
            contexts,
            target_mask,
            history_mask
        )
        
        return logits
    
    def generate(self,
                 histories: torch.Tensor,
                 contexts: torch.Tensor,
                 start_token_id: int,
                 end_token_id: int,
                 max_length: int = 15,
                 history_mask: Optional[torch.Tensor] = None,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None):
        """
        Generate predictions without teacher forcing
        
        Args:
            histories: [batch_size, seq_len, ball_vector_dim]
            contexts: [batch_size, context_dim]
            start_token_id: ID of start token
            end_token_id: ID of end token
            max_length: Maximum generation length
            history_mask: [batch_size, seq_len]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
        """
        batch_size = histories.shape[0]
        device = histories.device
        
        # Encode match history
        memory = self.encoder(histories, history_mask)
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token_id, 
                              dtype=torch.long, device=device)
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self.decoder(
                generated,
                memory,
                contexts,
                memory_mask=history_mask
            )
            
            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply sampling strategies
            if top_k is not None:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            if top_p is not None:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Add to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end tokens
            finished = finished | (next_token.squeeze(-1) == end_token_id)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int):
        """Apply top-k filtering"""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float):
        """Apply top-p (nucleus) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_cricket_transformer(config: Dict) -> CricketTransformer:
    """Factory function to create cricket transformer with config"""
    
    return CricketTransformer(
        ball_vector_dim=config.get('ball_vector_dim', 18),
        context_dim=config.get('context_dim', 20),
        vocab_size=config.get('vocab_size', 24),
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        max_target_length=config.get('max_target_length', 15)
    )

if __name__ == "__main__":
    # Test model creation
    config = {
        'ball_vector_dim': 18,
        'context_dim': 20,
        'vocab_size': 24,
        'd_model': 256,  # Smaller for testing
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1
    }
    
    model = create_cricket_transformer(config)
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    tgt_len = 6
    
    histories = torch.randn(batch_size, seq_len, 18)
    contexts = torch.randn(batch_size, 20)
    target_inputs = torch.randint(0, 24, (batch_size, tgt_len))
    
    with torch.no_grad():
        logits = model(histories, contexts, target_inputs)
        print(f"Output shape: {logits.shape}")
        
        # Test generation
        generated = model.generate(
            histories, contexts, 
            start_token_id=1, end_token_id=2, 
            max_length=10
        )
        print(f"Generated shape: {generated.shape}")