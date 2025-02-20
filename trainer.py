from diffcoder import DiffCoder
from buffer import Buffer
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import wandb

class Trainer:
    def __init__(self, cfg, model_A, model_B, all_tokens):
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.diffcoder = DiffCoder(cfg)
        self.buffer = Buffer(cfg, model_A, model_B, all_tokens)
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]

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
        wandb.init(project=cfg["wandb_project"])

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def step(self):
        acts = self.buffer.next()
        losses = self.diffcoder.get_losses(acts)
        
        # First update: encoder and decoder1
        # Goal: minimize mse1 - alpha * mse2 + l1_regularization
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
        # Goal: minimize mse2
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
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.diffcoder.save()

    def train(self):
        self.step_counter = 0
        try:
            for i in tqdm.trange(self.total_steps):
                loss_dict = self.step()
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            self.save()