"""
Training logic for uncertainty probes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import wandb
from tqdm import tqdm
from pathlib import Path
import json

from model import ModelWithProbe
from uncertainty import SupervisionSignalGenerator


class ProbeTrainer:
    """Trainer for uncertainty probes."""
    
    def __init__(
        self,
        model: ModelWithProbe,
        train_loader: DataLoader,
        val_loaders: Dict[str, DataLoader],
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.config = config
        self.device = device
        
        # Setup optimizer (only train probe components)
        trainable_params = [
            p for p in model.parameters() if p.requires_grad
        ]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * config['num_epochs']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # Loss functions
        self.use_regression = config.get('use_regression', False)
        if self.use_regression:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Supervision signal generator
        self.signal_generator = SupervisionSignalGenerator(
            model=model,
            tokenizer=model.tokenizer,
            num_generations=config.get('num_generations', 10),
            temperature=config.get('temperature', 1.0)
        )
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup wandb
        if config.get('use_wandb', True):
            wandb.init(
                project=config.get('wandb_project', 'attention-probing'),
                config=config,
                name=config.get('run_name', None)
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answers = batch['answers']  # List of lists
            
            # Generate supervision signals
            with torch.no_grad():
                supervision = self.signal_generator.generate_supervision_signals(
                    input_ids=input_ids,
                    ground_truth_answers=answers,
                    methods=['semantic_entropy', 'correctness']
                )
            
            # Prepare labels
            if self.use_regression:
                labels = supervision['semantic_entropy'].unsqueeze(-1)
            else:
                # Binary classification based on semantic entropy threshold
                labels = self.signal_generator.create_binary_labels(
                    supervision['semantic_entropy'],
                    method='median'
                )
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_probe_logits=True
            )
            
            probe_logits = outputs['probe_logits']
            
            # Compute loss
            loss = self.criterion(probe_logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('max_grad_norm', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            if not self.use_regression:
                predictions = probe_logits.argmax(dim=-1)
                total_correct += (predictions == labels).sum().item()
            
            total_samples += input_ids.shape[0]
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb
            if self.config.get('use_wandb', True) and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })
            
            self.global_step += 1
            
            # Periodic validation
            if self.config.get('val_every_n_steps', None) and \
               self.global_step % self.config['val_every_n_steps'] == 0:
                val_metrics = self.validate()
                self.model.train()
        
        # Epoch metrics
        metrics = {
            'loss': total_loss / len(self.train_loader)
        }
        
        if not self.use_regression:
            metrics['accuracy'] = total_correct / total_samples
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on all validation sets."""
        self.model.eval()
        
        all_metrics = {}
        
        for split_name, val_loader in self.val_loaders.items():
            split_loss = 0.0
            split_correct = 0
            split_samples = 0
            
            for batch in tqdm(val_loader, desc=f'Validating {split_name}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                answers = batch['answers']
                
                # Generate supervision
                supervision = self.signal_generator.generate_supervision_signals(
                    input_ids=input_ids,
                    ground_truth_answers=answers,
                    methods=['semantic_entropy']
                )
                
                # Prepare labels
                if self.use_regression:
                    labels = supervision['semantic_entropy'].unsqueeze(-1)
                else:
                    labels = self.signal_generator.create_binary_labels(
                        supervision['semantic_entropy']
                    )
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_probe_logits=True
                )
                
                probe_logits = outputs['probe_logits']
                loss = self.criterion(probe_logits, labels)
                
                split_loss += loss.item()
                
                if not self.use_regression:
                    predictions = probe_logits.argmax(dim=-1)
                    split_correct += (predictions == labels).sum().item()
                
                split_samples += input_ids.shape[0]
            
            # Compute metrics
            avg_loss = split_loss / len(val_loader)
            all_metrics[f'{split_name}/loss'] = avg_loss
            
            if not self.use_regression:
                accuracy = split_correct / split_samples
                all_metrics[f'{split_name}/accuracy'] = accuracy
        
        # Log to wandb
        if self.config.get('use_wandb', True):
            wandb.log({f'val/{k}': v for k, v in all_metrics.items()})
        
        return all_metrics
    
    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch} - Train metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Validate
            val_metrics = self.validate()
            
            print(f"Epoch {epoch} - Validation metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Save checkpoint
            avg_val_loss = sum(
                v for k, v in val_metrics.items() if 'loss' in k
            ) / len([k for k in val_metrics.keys() if 'loss' in k])
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % self.config.get('save_every_n_epochs', 5) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("Training complete!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")