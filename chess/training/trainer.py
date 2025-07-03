"""
Training script for Chess Transformer with AlphaZero-style Policy Head.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
# Import wandb optionally
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.chess_transformer import ChessTransformer, ChessLoss
from training.data_loader import create_data_loaders


class ChessTrainer:
    """Trainer for Chess Transformer model."""
    
    def __init__(self,
                 model: ChessTransformer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 illegal_move_penalty: float = 1.0,
                 save_dir: str = "chess/checkpoints",
                 use_wandb: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: Chess transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            illegal_move_penalty: Penalty for illegal moves
            save_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.loss_fn = ChessLoss(illegal_move_penalty=illegal_move_penalty)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,  # Will be updated based on training steps
            eta_min=learning_rate * 0.1
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Initialize wandb if requested
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="chess-transformer",
                config={
                    "model_params": sum(p.numel() for p in model.parameters()),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "illegal_move_penalty": illegal_move_penalty,
                    "vocab_size": model.vocab_size,
                    "d_model": model.d_model,
                    "policy_size": model.policy_size
                }
            )
        elif self.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb not available, disabling logging")
            self.use_wandb = False
        
        # Load checkpoint if exists
        self.load_checkpoint()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_illegal_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train]")
        
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            target_indices = batch['target_policy_index'].to(self.device)
            legal_moves_masks = batch.get('legal_moves_mask', None)
            if legal_moves_masks is not None:
                legal_moves_masks = legal_moves_masks.to(self.device)
            
            # Forward pass
            policy_logits = self.model(input_ids)
            
            # Calculate loss
            loss_dict = self.loss_fn(policy_logits, target_indices, legal_moves_masks)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(policy_logits, dim=1)
            accuracy = (predictions == target_indices).float().mean()
            
            # Update metrics
            total_loss += loss.item()
            total_ce_loss += loss_dict['cross_entropy_loss'].item()
            if 'illegal_move_loss' in loss_dict:
                total_illegal_loss += loss_dict['illegal_move_loss'].item()
            total_accuracy += accuracy.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy.item():.3f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and self.global_step % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/ce_loss': loss_dict['cross_entropy_loss'].item(),
                    'train/accuracy': accuracy.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'global_step': self.global_step
                })
                
                if 'illegal_move_loss' in loss_dict:
                    wandb.log({
                        'train/illegal_loss': loss_dict['illegal_move_loss'].item(),
                        'train/illegal_prob_mass': loss_dict['illegal_prob_mass'].item(),
                        'global_step': self.global_step
                    })
        
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_illegal_loss = total_illegal_loss / num_batches if total_illegal_loss > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'illegal_loss': avg_illegal_loss,
            'accuracy': avg_accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_illegal_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]")
            
            for batch in pbar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                target_indices = batch['target_policy_index'].to(self.device)
                legal_moves_masks = batch.get('legal_moves_mask', None)
                if legal_moves_masks is not None:
                    legal_moves_masks = legal_moves_masks.to(self.device)
                
                # Forward pass
                policy_logits = self.model(input_ids)
                
                # Calculate loss
                loss_dict = self.loss_fn(policy_logits, target_indices, legal_moves_masks)
                loss = loss_dict['total_loss']
                
                # Calculate accuracy
                predictions = torch.argmax(policy_logits, dim=1)
                accuracy = (predictions == target_indices).float().mean()
                
                # Update metrics
                total_loss += loss.item()
                total_ce_loss += loss_dict['cross_entropy_loss'].item()
                if 'illegal_move_loss' in loss_dict:
                    total_illegal_loss += loss_dict['illegal_move_loss'].item()
                total_accuracy += accuracy.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy.item():.3f}"
                })
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_illegal_loss = total_illegal_loss / num_batches if total_illegal_loss > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'illegal_loss': avg_illegal_loss,
            'accuracy': avg_accuracy
        }
    
    def load_checkpoint(self):
        """Load checkpoint if exists."""
        best_model_path = os.path.join(self.save_dir, "best_model.pt")
        
        if os.path.exists(best_model_path):
            print(f"Loading checkpoint from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            print(f"Resumed from epoch {self.epoch + 1}, best val loss: {self.best_val_loss:.4f}")
        else:
            print("No checkpoint found, starting from scratch")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{self.epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training on device: {self.device}")
        
        start_epoch = self.epoch + 1 if self.epoch > 0 else 0
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'val/loss': val_metrics['loss'],
                    'val/ce_loss': val_metrics['ce_loss'],
                    'val/accuracy': val_metrics['accuracy'],
                })
                
                if val_metrics['illegal_loss'] > 0:
                    wandb.log({
                        'val/illegal_loss': val_metrics['illegal_loss']
                    })
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_file': 'chess/processed_data/chess_training_sample.json',
        'batch_size': 16,
        'max_seq_len': 256,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'illegal_move_penalty': 1.0,
        'num_epochs': 10,
        'save_every': 2,
        'use_wandb': False,  # Set to True to enable wandb logging
        
        # Model config
        'vocab_size': 2072,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'policy_size': 4672
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_file=config['data_file'],
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        include_legal_moves=True,
        num_workers=0  # Use 0 to avoid multiprocessing issues
    )
    
    # Create model
    print("Creating model...")
    model = ChessTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        policy_size=config['policy_size']
    )
    
    # Create trainer
    trainer = ChessTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        illegal_move_penalty=config['illegal_move_penalty'],
        use_wandb=config['use_wandb']
    )
    
    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        save_every=config['save_every']
    )


if __name__ == "__main__":
    main()