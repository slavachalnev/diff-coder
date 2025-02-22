# %%
from trainer import MultiTrainer
from transformer_lens import HookedTransformer
from utils import load_pile_lmsys_mixed_tokens, arg_parse_update_cfg
import torch
from pathlib import Path
from config import get_activations_cache_dir
# %%
device = 'cuda:0'

# base_model = HookedTransformer.from_pretrained(
#     "gemma-2-2b", 
#     device=device,
#     # dtype=torch.bfloat16,
#     dtype=torch.float16,
# )

# chat_model = HookedTransformer.from_pretrained(
#     "gemma-2-2b-it", 
#     device=device,
#     # dtype=torch.bfloat16,
#     dtype=torch.float16,
# )

# # %%
# all_tokens = load_pile_lmsys_mixed_tokens()


# %%
# Base configuration with shared settings
base_cfg = {
    "seed": 49,
    "batch_size": 8192,
    "buffer_mult": 128,
    "num_tokens": 4_000_000,
    "d_in": 2304,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "gemma-2-2b",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 2,
    "log_every": 100,
    "save_every": 30000,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "gemma-diff-coder",
    # Cache settings
    "use_cache": True,
    "cache_size_gb": 40,
    "cache_dir": str(get_activations_cache_dir()),
}

# Different configurations for DiffCoders to train in parallel
diffcoder_cfgs = [
    {
        "dict_size": 1024,
        "lr": 5e-5,
        "l1_coeff": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "dec_init_norm": 0.08,
    },
    {
        "dict_size": 1024,
        "lr": 5e-5,
        "l1_coeff": 0.005,
        "beta1": 0.9,
        "beta2": 0.999,
        "dec_init_norm": 0.08,
    },
    {
        "dict_size": 1024,
        "lr": 5e-5,
        "l1_coeff": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "dec_init_norm": 0.08,
    },
]

base_cfg = arg_parse_update_cfg(base_cfg)

# Initialize MultiTrainer with different configs
trainer = MultiTrainer(base_cfg, diffcoder_cfgs)
trainer.train()
# %%