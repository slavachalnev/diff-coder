import torch
from torch import nn
import pprint
import einops
import json
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import NamedTuple

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace/crosscoder-model-diff-replication/checkpoints")

class LossOutput(NamedTuple):
    mse1: torch.Tensor  # MSE for target 1 (want to minimize)
    mse2: torch.Tensor  # MSE for target 2 (want to maximize)
    l1_loss: torch.Tensor  # Sparsity regularization
    l0_loss: torch.Tensor  # Activation sparsity

class DiffCoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        self.alpha = self.cfg.get("alpha", 0.5)  # Trade-off parameter
        torch.manual_seed(self.cfg["seed"])
        
        # Encoder
        self.W_enc = nn.Parameter(
            torch.empty(2, d_in, d_hidden, dtype=self.dtype)
        )
        
        # Two separate decoders for y1 and y2
        self.W_dec1 = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(d_hidden, d_in, dtype=self.dtype)
            )
        )
        self.W_dec2 = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(d_hidden, d_in, dtype=self.dtype)
            )
        )
        
        # Initialize decoder norms
        self.W_dec1.data = (
            self.W_dec1.data / self.W_dec1.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        self.W_dec2.data = (
            self.W_dec2.data / self.W_dec2.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        
        # Initialize both parts of encoder using decoder1's weights
        self.W_enc.data = torch.stack([
            self.W_dec1.data.clone().T,  # Transpose of decoder1 for first input
            self.W_dec1.data.clone().T   # Also use decoder1's transpose for second input
        ], dim=0)
        
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec1 = nn.Parameter(torch.zeros(d_in, dtype=self.dtype))
        self.b_dec2 = nn.Parameter(torch.zeros(d_in, dtype=self.dtype))
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            acts = F.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts

    def decode1(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = einops.einsum(
            acts,
            self.W_dec1,
            "batch d_hidden, d_hidden d_model -> batch d_model",
        )
        return acts_dec + self.b_dec1

    def decode2(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = einops.einsum(
            acts,
            self.W_dec2,
            "batch d_hidden, d_hidden d_model -> batch d_model",
        )
        return acts_dec + self.b_dec2

    def forward(self, x):
        # x: [batch, n_models, d_model]
        acts = self.encode(x)
        y1_pred = self.decode1(acts)
        y2_pred = self.decode2(acts)
        return y1_pred, y2_pred

    def get_losses(self, x):
        # x: [batch, n_models, d_model]
        x = x.to(self.dtype)
        acts = self.encode(x)
        
        # Get predictions for both targets
        y1_pred = self.decode1(acts)
        y2_pred = self.decode2(acts)
        
        # Target 1 - want to minimize this error
        diff1 = y1_pred.float() - x[:, 0].float()  # Compare to first model
        mse1 = diff1.pow(2).mean()
        
        # Target 2 - want to maximize this error
        diff2 = y2_pred.float() - x[:, 1].float()  # Compare to second model
        mse2 = diff2.pow(2).mean()
        
        # L1 regularization on activations with decoder norms
        decoder1_norm = self.W_dec1.norm(dim=-1)
        decoder2_norm = self.W_dec2.norm(dim=-1)
        total_decoder_norm = decoder1_norm + decoder2_norm
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)
        
        # L0 sparsity - fraction of neurons that fire
        l0_loss = (acts>0).float().sum(-1).mean()

        return LossOutput(
            mse1=mse1,
            mse2=mse2,
            l1_loss=l1_loss,
            l0_loss=l0_loss,
        )

    def create_save_dir(self):
        base_dir = Path("/workspace/crosscoder-model-diff-replication/checkpoints")
        version_list = [
            int(file.name.split("_")[1])
            for file in list(SAVE_DIR.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = base_dir / f"version_{version}"
        self.save_dir.mkdir(parents=True)

    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f)

        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1

    @classmethod
    def load(cls, version_dir, checkpoint_version):
        save_dir = Path("/workspace/crosscoder-model-diff-replication/checkpoints") / str(version_dir)
        cfg_path = save_dir / f"{str(checkpoint_version)}_cfg.json"
        weight_path = save_dir / f"{str(checkpoint_version)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(weight_path))
        return self