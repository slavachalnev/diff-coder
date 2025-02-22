from diffcoder import DiffCoder
from buffer import Buffer
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import wandb
from typing import List, Dict
import copy

class MultiTrainer:
    def __init__(self, base_cfg: Dict, diffcoder_cfgs: List[Dict], model_A=None, model_B=None, all_tokens=None):
        """
        Initialize multiple DiffCoders with different configs that share the same buffer.
        
        Args:
            base_cfg: Base configuration containing shared settings like buffer config
            diffcoder_cfgs: List of configs for different DiffCoders, each extending base_cfg
            model_A, model_B, all_tokens: Models and data for buffer initialization
        """
        self.base_cfg = base_cfg
        self.buffer = Buffer(base_cfg, model_A, model_B, all_tokens)
        
        # Initialize wandb run for the entire training session
        wandb.init(
            project=base_cfg["wandb_project"],
            config={"base_cfg": base_cfg, "num_diffcoders": len(diffcoder_cfgs)}
        )
        
        # Initialize DiffCoders, optimizers, and schedulers for each config
        self.trainers = []
        for i, dc_cfg in enumerate(diffcoder_cfgs):
            # Merge base config with specific diffcoder config
            cfg = copy.deepcopy(base_cfg)
            cfg.update(dc_cfg)
            
            # Create trainer instance
            trainer = Trainer(cfg, buffer=self.buffer)
            trainer.wandb_group = f"diffcoder_{i}"
            
            # Log this specific DiffCoder's config
            wandb.config.update({f"diffcoder_{i}_cfg": dc_cfg}, allow_val_change=True)
            
            self.trainers.append(trainer)

    def train(self):
        """Train all DiffCoders in parallel using the same buffer data."""
        try:
            for i in tqdm.trange(self.base_cfg["num_tokens"] // self.base_cfg["batch_size"]):
                # Get shared batch of activations
                acts = self.buffer.next()
                
                # Train each DiffCoder on this batch
                for trainer in self.trainers:
                    loss_dict = trainer.train_step(acts)
                    
                    if i % self.base_cfg["log_every"] == 0:
                        trainer.log(loss_dict)
                    
                    if (i + 1) % self.base_cfg["save_every"] == 0:
                        trainer.save()
        
        finally:
            # Save all models at the end
            for trainer in self.trainers:
                trainer.save()
            wandb.finish()

class Trainer:
    def __init__(self, cfg, model_A=None, model_B=None, all_tokens=None, buffer=None):
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.diffcoder = DiffCoder(cfg)
        self.buffer = buffer if buffer is not None else Buffer(cfg, model_A, model_B, all_tokens)
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]
        self.wandb_group = None  # For use with MultiTrainer

        # Separate optimizers for encoder+decoder1 and decoder2
        self.optimizer_enc_dec1 = torch.optim.Adam(
            [
                {'params': self.diffcoder.W_enc},
                {'params': self.diffcoder.b_enc},
                {'params': self.diffcoder.W_dec1},
                {'params': self.diffcoder.b_dec1}
            ],
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        
        self.optimizer_dec2 = torch.optim.Adam(
            [
                {'params': self.diffcoder.W_dec2},
                {'params': self.diffcoder.b_dec2}
            ],
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )

        # Separate schedulers for each optimizer
        self.scheduler_enc_dec1 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_enc_dec1, self.lr_lambda
        )
        self.scheduler_dec2 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_dec2, self.lr_lambda
        )
        
        self.step_counter = 0

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def train_step(self, acts):
        """Single training step that can be called externally with provided activations"""
        losses = self.diffcoder.get_losses(acts)
        
        # First update: encoder and decoder1
        self.optimizer_enc_dec1.zero_grad()
        loss_enc_dec1 = (
            losses.mse1 - 
            self.diffcoder.alpha * losses.mse2 + 
            self.get_l1_coeff() * losses.l1_loss
        )
        loss_enc_dec1.backward(retain_graph=True)
        clip_grad_norm_(
            [self.diffcoder.W_enc, self.diffcoder.b_enc, 
             self.diffcoder.W_dec1, self.diffcoder.b_dec1], 
            max_norm=1.0
        )
        self.optimizer_enc_dec1.step()
        
        # Second update: decoder2
        self.optimizer_dec2.zero_grad()
        loss_dec2 = losses.mse2
        loss_dec2.backward()
        clip_grad_norm_(
            [self.diffcoder.W_dec2, self.diffcoder.b_dec2],
            max_norm=1.0
        )
        self.optimizer_dec2.step()
        
        # Update learning rates
        self.scheduler_enc_dec1.step()
        self.scheduler_dec2.step()

        loss_dict = {
            "total_loss": loss_enc_dec1.item() + loss_dec2.item(),
            "mse1": losses.mse1.item(),
            "mse2": losses.mse2.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler_enc_dec1.get_last_lr()[0],
        }
        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        if self.wandb_group:
            wandb.log({f"{self.wandb_group}/{k}": v for k, v in loss_dict.items()}, step=self.step_counter)
        else:
            wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.diffcoder.save()

    def train(self):
        """Main training loop when running a single trainer"""
        self.step_counter = 0
        try:
            for i in tqdm.trange(self.total_steps):
                acts = self.buffer.next()
                loss_dict = self.train_step(acts)
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            self.save()