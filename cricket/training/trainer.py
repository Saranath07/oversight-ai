"""
Cricket Transformer Trainer
===========================

Training script for the cricket transformer model with teacher forcing
and autonomous training capabilities. Optimized for Apple Silicon Macs using MPS.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cricket_transformer import CricketTransformer, create_cricket_transformer
from data_generation.data_loader import CricketDataLoader, decode_predictions

class CricketTrainer:
    """Trainer class for cricket transformer model"""
    
    def __init__(self, 
                 model: CricketTransformer,
                 data_loader: CricketDataLoader,
                 config: Dict,
                 save_dir: str = "checkpoints"):
        
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training components - prioritize MPS for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon MPS (Neural Processing Unit)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using CUDA GPU")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignore padding tokens
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Tensorboard logging
        self.writer = SummaryWriter(log_dir=f"runs/cricket_transformer_{int(time.time())}")
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Vocabulary for decoding
        self.vocabulary = data_loader.vocabulary
        self.idx_to_token = data_loader.idx_to_token
        
        print(f"Trainer initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - Model parameters: {self.model.count_parameters():,}")
        print(f"  - Training batches: {len(self.data_loader.train_loader)}")
        print(f"  - Validation batches: {len(self.data_loader.val_loader)}")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device with MPS compatibility"""
        moved_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # For MPS, ensure tensor is contiguous and proper dtype
                if self.device.type == 'mps':
                    if v.dtype == torch.float64:
                        v = v.float()  # MPS doesn't support float64
                    v = v.contiguous()
                moved_batch[k] = v.to(self.device)
            else:
                moved_batch[k] = v
        return moved_batch
    
    def train_epoch(self, use_teacher_forcing: bool = True) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        for batch_idx, batch in enumerate(self.data_loader.train_loader):
            # Move batch to device with MPS compatibility
            batch = self._move_batch_to_device(batch)
            
            self.optimizer.zero_grad()
            
            try:
                if use_teacher_forcing:
                    # Teacher forcing: use ground truth tokens as input
                    logits = self.model(
                        histories=batch['histories'],
                        contexts=batch['contexts'],
                        target_inputs=batch['target_inputs'],
                        history_mask=batch['history_mask'],
                        target_mask=batch['target_mask']
                    )
                    
                    # Calculate loss
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        batch['target_outputs'].reshape(-1)
                    )
                else:
                    # Autonomous training: use model's own predictions
                    loss = self._autonomous_training_step(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                total_tokens += (batch['target_mask'].sum().item() if 'target_mask' in batch else 
                               batch['target_outputs'].numel())
                
                # Log progress
                if batch_idx % 100 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"  Batch {batch_idx}/{len(self.data_loader.train_loader)}, "
                          f"Loss: {avg_loss:.4f}")
                    
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                
                self.global_step += 1
                
            except RuntimeError as e:
                if "MPS" in str(e):
                    print(f"MPS Error in batch {batch_idx}: {e}")
                    print("Skipping batch and continuing...")
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / len(self.data_loader.train_loader)
        perplexity = np.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens
        }
    
    def _autonomous_training_step(self, batch: Dict) -> torch.Tensor:
        """Training step without teacher forcing"""
        
        batch_size = batch['histories'].shape[0]
        start_token_id = self.vocabulary['<START>']
        end_token_id = self.vocabulary['<END>']
        
        # Generate sequences
        with torch.no_grad():
            generated = self.model.generate(
                histories=batch['histories'],
                contexts=batch['contexts'],
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                max_length=batch['target_outputs'].shape[1],
                history_mask=batch['history_mask'],
                temperature=0.8
            )
        
        # Ensure generated sequences are the same length as target inputs for training
        target_seq_len = batch['target_inputs'].shape[1]
        
        # Use generated sequence as input (except last token)
        # Truncate or pad to match the target input length
        if generated.shape[1] - 1 > target_seq_len:
            # Truncate if too long
            target_inputs = generated[:, :target_seq_len]
        else:
            # Pad with end tokens if too short
            padding_needed = target_seq_len - (generated.shape[1] - 1)
            padding = torch.full((batch_size, padding_needed), end_token_id,
                               dtype=torch.long, device=generated.device)
            target_inputs = torch.cat([generated[:, :-1], padding], dim=1)
        
        # Forward pass with generated inputs
        logits = self.model(
            histories=batch['histories'],
            contexts=batch['contexts'],
            target_inputs=target_inputs,
            history_mask=batch['history_mask'],
            target_mask=batch['target_mask']  # Add target mask for proper padding handling
        )
        
        # Calculate loss against ground truth
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            batch['target_outputs'].reshape(-1)
        )
        
        return loss
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.data_loader.val_loader:
                try:
                    # Move batch to device with MPS compatibility
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    logits = self.model(
                        histories=batch['histories'],
                        contexts=batch['contexts'],
                        target_inputs=batch['target_inputs'],
                        history_mask=batch['history_mask'],
                        target_mask=batch['target_mask']
                    )
                    
                    # Calculate loss
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        batch['target_outputs'].reshape(-1)
                    )
                    
                    total_loss += loss.item()
                    total_tokens += (batch['target_mask'].sum().item() if 'target_mask' in batch else 
                                   batch['target_outputs'].numel())
                    
                    # Collect predictions for analysis
                    pred_tokens = torch.argmax(logits, dim=-1)
                    predictions.extend(decode_predictions(pred_tokens, self.idx_to_token))
                    targets.extend(batch['target_tokens'])
                    
                except RuntimeError as e:
                    if "MPS" in str(e):
                        print(f"MPS Error in validation: {e}")
                        print("Skipping batch and continuing...")
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / len(self.data_loader.val_loader)
        perplexity = np.exp(avg_loss)
        
        # Calculate accuracy metrics
        accuracy = self._calculate_accuracy(predictions, targets)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'tokens': total_tokens,
            'predictions': predictions[:5],  # Sample predictions
            'targets': targets[:5]  # Sample targets
        }
    
    def _calculate_accuracy(self, predictions: List[List[str]], 
                          targets: List[List[str]]) -> Dict[str, float]:
        """Calculate various accuracy metrics"""
        
        exact_matches = 0
        token_matches = 0
        total_tokens = 0
        
        for pred, target in zip(predictions, targets):
            # Exact sequence match
            if pred == target:
                exact_matches += 1
            
            # Token-level accuracy
            min_len = min(len(pred), len(target))
            for i in range(min_len):
                if pred[i] == target[i]:
                    token_matches += 1
            
            total_tokens += max(len(pred), len(target))
        
        return {
            'exact_match': exact_matches / len(predictions) if predictions else 0.0,
            'token_accuracy': token_matches / total_tokens if total_tokens > 0 else 0.0
        }
    
    def train(self, 
              num_epochs: int,
              teacher_forcing_epochs: int = None,
              save_every: int = 10,
              validate_every: int = 1):
        """Main training loop"""
        
        if teacher_forcing_epochs is None:
            teacher_forcing_epochs = num_epochs // 2
        
        print(f"\nStarting training for {num_epochs} epochs:")
        print(f"  - Teacher forcing: epochs 1-{teacher_forcing_epochs}")
        print(f"  - Autonomous training: epochs {teacher_forcing_epochs+1}-{num_epochs}")
        
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            start_time = time.time()
            
            # Determine training mode
            use_teacher_forcing = epoch <= teacher_forcing_epochs
            mode = "Teacher Forcing" if use_teacher_forcing else "Autonomous"
            
            print(f"\nEpoch {epoch}/{num_epochs} ({mode})")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(use_teacher_forcing)
            
            # Validate
            if epoch % validate_every == 0:
                val_metrics = self.validate()
                
                # Log metrics
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Perplexity', train_metrics['perplexity'], epoch)
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Perplexity', val_metrics['perplexity'], epoch)
                self.writer.add_scalar('Val/ExactMatch', val_metrics['accuracy']['exact_match'], epoch)
                self.writer.add_scalar('Val/TokenAccuracy', val_metrics['accuracy']['token_accuracy'], epoch)
                
                # Update learning rate
                self.scheduler.step(val_metrics['loss'])
                
                # Print results
                epoch_time = time.time() - start_time
                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Train PPL: {train_metrics['perplexity']:.2f}, "
                      f"Val PPL: {val_metrics['perplexity']:.2f}")
                print(f"Exact Match: {val_metrics['accuracy']['exact_match']:.3f}, "
                      f"Token Acc: {val_metrics['accuracy']['token_accuracy']:.3f}")
                print(f"Time: {epoch_time:.1f}s")
                
                # Show sample predictions
                print("\nSample Predictions:")
                for i, (pred, target) in enumerate(zip(val_metrics['predictions'][:3], 
                                                     val_metrics['targets'][:3])):
                    print(f"  {i+1}. Pred: {pred}")
                    print(f"     True: {target}")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
                    print("  → New best model saved!")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Close tensorboard writer
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'vocabulary': self.vocabulary
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")

def main():
    """Main training function"""
    
    # Configuration - optimized for Apple Silicon with configurable context windows
    config = {
        # Model architecture
        'ball_vector_dim': 18,
        'context_dim': 20,
        'vocab_size': 24,
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_target_length': 15,
        
        # Training parameters - adjusted for MPS
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'batch_size': 16,  # Smaller batch size for MPS stability
        
        # Context window configuration (NEW APPROACH)
        'use_sliding_window': True,  # Use realistic short context windows
        'context_window_size': 16,  # Number of recent balls as context (try 8, 16, 32)
        'max_history_length': 128,  # Legacy mode fallback (if sliding window disabled)
        
        # Training schedule
        'num_epochs': 100,
        'teacher_forcing_epochs': 50,
        'save_every': 10,
        'validate_every': 1
    }
    
    # Data directory (adjust path based on current working directory)
    if Path("training").exists():
        # Running from root directory
        data_dir = "data_generation/processed"
    else:
        # Running from training directory
        data_dir = "../data_generation/processed"
    
    if not Path(data_dir).exists():
        print(f"Error: Data directory {data_dir} not found!")
        print("Please run the data generation pipeline first.")
        return
    
    # Create data loader with new sliding window approach
    print("Loading data...")
    data_loader = CricketDataLoader(
        data_dir=data_dir,
        batch_size=config['batch_size'],
        max_history_length=config['max_history_length'],
        train_split=0.8,
        use_sliding_window=config['use_sliding_window'],
        context_window_size=config['context_window_size']
    )
    
    print(f"\n=== Training Configuration ===")
    print(f"Context approach: {'Sliding Window' if config['use_sliding_window'] else 'Legacy'}")
    if config['use_sliding_window']:
        print(f"Context window size: {config['context_window_size']} balls")
        print(f"Benefits: Prevents data leakage, more realistic training")
    else:
        print(f"Max history length: {config['max_history_length']} balls")
    print(f"Batch size: {config['batch_size']}")
    
    # Update config with actual vocabulary size
    config['vocab_size'] = len(data_loader.vocabulary)
    
    # Create model
    print("Creating model...")
    model = create_cricket_transformer(config)
    
    # Create trainer
    trainer = CricketTrainer(model, data_loader, config)
    
    # Start training
    trainer.train(
        num_epochs=config['num_epochs'],
        teacher_forcing_epochs=config['teacher_forcing_epochs'],
        save_every=config['save_every'],
        validate_every=config['validate_every']
    )

if __name__ == "__main__":
    main()