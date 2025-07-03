"""
Chess Transformer with AlphaZero-style Policy Head
Following the architecture described in the prompt.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
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


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(context)
        return output


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class ChessTransformer(nn.Module):
    """
    Chess Transformer with AlphaZero-style Policy Head.
    
    Architecture:
    - Input: Sequence of UCI move tokens
    - Embedding: Token + Positional embeddings
    - Transformer: N decoder blocks with causal attention
    - Output: Linear layer projecting to 4672 policy logits
    """
    
    def __init__(self, 
                 vocab_size: int = 2072,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 policy_size: int = 4672):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.policy_size = policy_size
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head - AlphaZero style policy head
        self.policy_head = nn.Linear(d_model, policy_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for decoder-only architecture."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            policy_logits: Logits for 4672 possible moves [batch_size, 4672]
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Combine causal mask with attention mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            causal_mask = causal_mask * attention_mask
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Take the last token's representation for next move prediction
        last_token_hidden = x[:, -1, :]  # [batch_size, d_model]
        
        # Project to policy space
        policy_logits = self.policy_head(last_token_hidden)  # [batch_size, 4672]
        
        return policy_logits
    
    def generate_move(self, 
                     input_ids: torch.Tensor,
                     legal_moves_mask: Optional[torch.Tensor] = None,
                     temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
        """
        Generate next move given input sequence.
        
        Args:
            input_ids: Input token sequence [1, seq_len]
            legal_moves_mask: Binary mask for legal moves [4672]
            temperature: Sampling temperature
            
        Returns:
            policy_index: Selected move index
            probabilities: Move probabilities [4672]
        """
        self.eval()
        with torch.no_grad():
            # Get policy logits
            policy_logits = self.forward(input_ids)  # [1, 4672]
            policy_logits = policy_logits.squeeze(0)  # [4672]
            
            # Apply legal moves mask if provided
            if legal_moves_mask is not None:
                # Set illegal moves to very low probability
                policy_logits = policy_logits + (legal_moves_mask - 1) * 1e9
            
            # Apply temperature
            if temperature != 1.0:
                policy_logits = policy_logits / temperature
            
            # Convert to probabilities
            probabilities = F.softmax(policy_logits, dim=0)
            
            # Sample from distribution
            policy_index = torch.multinomial(probabilities, 1).item()
            
            return policy_index, probabilities


class ChessLoss(nn.Module):
    """
    Combined loss function for chess training.
    Includes cross-entropy loss and illegal move penalty.
    """
    
    def __init__(self, illegal_move_penalty: float = 1.0):
        super().__init__()
        self.illegal_move_penalty = illegal_move_penalty
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, 
                policy_logits: torch.Tensor,
                target_policy_indices: torch.Tensor,
                legal_moves_masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            policy_logits: Model output [batch_size, 4672]
            target_policy_indices: Target move indices [batch_size]
            legal_moves_masks: Legal moves masks [batch_size, 4672]
            
        Returns:
            Dictionary with loss components
        """
        # Cross-entropy loss
        ce_loss = self.cross_entropy(policy_logits, target_policy_indices)
        
        total_loss = ce_loss
        loss_dict = {
            'total_loss': total_loss,
            'cross_entropy_loss': ce_loss,
        }
        
        # Illegal move penalty if masks provided
        if legal_moves_masks is not None and self.illegal_move_penalty > 0:
            # Convert logits to probabilities
            probabilities = F.softmax(policy_logits, dim=1)
            
            # Calculate probability mass on illegal moves
            illegal_prob_mass = torch.sum(
                probabilities * (1 - legal_moves_masks), 
                dim=1
            ).mean()
            
            illegal_loss = self.illegal_move_penalty * illegal_prob_mass
            total_loss = ce_loss + illegal_loss
            
            loss_dict.update({
                'total_loss': total_loss,
                'illegal_move_loss': illegal_loss,
                'illegal_prob_mass': illegal_prob_mass
            })
        
        return loss_dict


if __name__ == "__main__":
    # Test the model
    print("Testing Chess Transformer...")
    
    # Model parameters
    vocab_size = 2072
    batch_size = 4
    seq_len = 20
    
    # Create model
    model = ChessTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    policy_logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {policy_logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test loss
    loss_fn = ChessLoss()
    target_indices = torch.randint(0, 4672, (batch_size,))
    
    loss_dict = loss_fn(policy_logits, target_indices)
    print(f"Loss: {loss_dict['total_loss'].item():.4f}")
    
    print("Model test completed successfully!")