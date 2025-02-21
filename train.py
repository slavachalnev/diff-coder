# %%
from trainer import Trainer
from transformer_lens import HookedTransformer
from utils import load_pile_lmsys_mixed_tokens, arg_parse_update_cfg
import torch
from pathlib import Path
from config import get_activations_cache_dir
# %%
device = 'cuda:0'

base_model = HookedTransformer.from_pretrained(
    "gemma-2-2b", 
    device=device,
    # dtype=torch.bfloat16,
    dtype=torch.float16,
)

chat_model = HookedTransformer.from_pretrained(
    "gemma-2-2b-it", 
    device=device,
    # dtype=torch.bfloat16,
    dtype=torch.float16,
)

# %%
all_tokens = load_pile_lmsys_mixed_tokens()

# %%
default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    # "dict_size": 2**14,
    "dict_size": 512,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "gemma-2-2b",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 2,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "gemma-diff-coder",
    # Cache settings
    "use_cache": True,
    "cache_size_gb": 40,
    "cache_dir": str(get_activations_cache_dir()),
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()
# %%