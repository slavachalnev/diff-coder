# copied from https://github.com/ckkissane/crosscoder-model-diff-replication


import torch
import numpy as np
from transformer_lens import ActivationCache
import tqdm
import einops
import os
from pathlib import Path

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. 
    It can either generate activations on the fly or load them from a cache.
    """

    def __init__(self, cfg, model_A=None, model_B=None, all_tokens=None):
        self.cfg = cfg
        self.use_cache = cfg.get("use_cache", False)
        self.cache_dir = Path(cfg["cache_dir"])
        self.cache_file = self.cache_dir / "activations_cache.pt"
        self.normalize = True
        
        # If using cache and it exists, we only need minimal initialization
        if self.use_cache and self._cache_exists():
            print(f"Loading cache from {self.cache_file}")
            self._load_cache()
            self.buffer_pointer = 0
            return
            
        # For new cache generation or non-cache usage, we need models and tokens
        if model_A is None or model_B is None or all_tokens is None:
            raise ValueError("model_A, model_B, and all_tokens are required when cache doesn't exist or cache is not used")
            
        assert model_A.cfg.d_model == model_B.cfg.d_model
        self.model_A = model_A
        self.model_B = model_B
        self.all_tokens = all_tokens
        self.model_batch_size = cfg["model_batch_size"]
        self.seq_len = cfg["seq_len"]
        
        # Calculate max number of samples that fit in cache
        self.cache_size = 40 * (1024**3)  # Convert GB to bytes
        bytes_per_expanded_sample = 2 * model_A.cfg.d_model * 2  # 2 models, 2 for bfloat16
        samples_per_batch = self.model_batch_size * (self.seq_len - 1)  # Each batch expands to this many samples
        total_samples = int(self.cache_size / bytes_per_expanded_sample)
        self.max_samples = total_samples - (total_samples % samples_per_batch)  # Round down to nearest batch multiple
        
        print(f"Initializing buffer with {self.max_samples} samples")
        self.buffer = torch.zeros(
            (self.max_samples, 2, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        )
        
        self.token_pointer = 0
        self.buffer_pointer = 0
        
        # Compute normalization factors
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)
        
        self.normalisation_factor = torch.tensor(
            [estimated_norm_scaling_factor_A, estimated_norm_scaling_factor_B],
            dtype=torch.float32,
        )

        if self.use_cache:
            print(f"Cache not found at {self.cache_file}. Generating cache...")
            self._generate_cache()
        else:
            self.refresh()

    def _cache_exists(self):
        """Check if cache file exists"""
        return self.cache_file.exists()

    def _load_cache(self):
        """Load the entire cache into buffer"""
        self.buffer = torch.load(self.cache_file)  # Keep on CPU
        
    @torch.no_grad()
    def _generate_cache(self):
        """Generate activation cache and save to single file"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Move models to device
        self.model_A.to(self.cfg["device"])
        self.model_B.to(self.cfg["device"])
        
        buffer_pointer = 0
        samples_per_batch = self.model_batch_size * (self.seq_len - 1)
        num_full_batches = self.max_samples // samples_per_batch
        
        with torch.autocast("cuda", torch.bfloat16):
            for _ in tqdm.trange(num_full_batches):
                if self.token_pointer >= len(self.all_tokens):
                    break
                        
                tokens = self.all_tokens[
                    self.token_pointer : self.token_pointer + self.model_batch_size
                ]
                    
                acts = self._get_batch_activations(tokens)
                
                # Store in buffer
                next_pointer = buffer_pointer + acts.shape[0]
                self.buffer[buffer_pointer:next_pointer] = acts
                buffer_pointer = next_pointer
                self.token_pointer += self.model_batch_size
        
        # Trim buffer to actual size and save
        if buffer_pointer < self.max_samples:
            self.buffer = self.buffer[:buffer_pointer]
        
        torch.save(self.buffer, self.cache_file)  # Buffer is already on CPU
        
        # Move models back to CPU
        self.model_A.cpu()
        self.model_B.cpu()
        torch.cuda.empty_cache()
        
        print(f"Generated cache with {buffer_pointer} samples, total size: {(buffer_pointer * 2 * self.model_A.cfg.d_model * 2) / (1024**3):.2f} GB")

    def _get_batch_activations(self, tokens):
        """Get activations for a batch of tokens from both models.
        Handles device management and memory cleanup consistently."""
        # Process model A and move activation to CPU
        _, cache_A = self.model_A.run_with_cache(
                tokens, names_filter=self.cfg["hook_point"]
            )
        acts_A = cache_A[self.cfg["hook_point"]][:, 1:, :].cpu()  # Drop BOS and move to CPU
        del cache_A  # Clear cache to free GPU memory
        
        # Process model B and move activation to CPU
        _, cache_B = self.model_B.run_with_cache(
            tokens, names_filter=self.cfg["hook_point"]
        )
        acts_B = cache_B[self.cfg["hook_point"]][:, 1:, :].cpu()  # Drop BOS and move to CPU
        del cache_B  # Clear cache to free GPU memory
        
        # Stack and reshape on CPU
        acts = torch.stack([acts_A, acts_B], dim=0)
        del acts_A, acts_B
        
        acts = einops.rearrange(
            acts,
            "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
        )

        # Apply normalization if enabled
        if self.normalize:
            acts = acts * self.normalisation_factor[None, :, None]
        
        return acts

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, n_batches_for_norm_estimate: int = 100):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            _, cache = model.run_with_cache(
                tokens,
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            acts = cache[self.cfg["hook_point"]]
            # TODO: maybe drop BOS here
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm

        return scaling_factor

    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        print("Refreshing the buffer!")
        
        # Move buffer to CPU for processing
        self.buffer = self.buffer.cpu()
        
        if self.use_cache:
            # Load next chunk from cache
            chunk = self._load_next_chunk()
            # Shuffle the chunk
            chunk = chunk[torch.randperm(chunk.shape[0])]
            # Update buffer with as much data as it can hold
            self.buffer[:min(self.buffer.shape[0], chunk.shape[0])] = chunk[:min(self.buffer.shape[0], chunk.shape[0])]
        else:
            # Original refresh logic for non-cached operation
            self.model_A.to(self.cfg["device"])
            self.model_B.to(self.cfg["device"])
            
            with torch.autocast("cuda", torch.bfloat16):
                if self.first:
                    num_batches = self.buffer_batches
                else:
                    num_batches = self.buffer_batches // 2
                self.first = False
                for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                    tokens = self.all_tokens[
                        self.token_pointer : min(
                            self.token_pointer + self.cfg["model_batch_size"], num_batches
                        )
                    ]
                    
                    acts = self._get_batch_activations(tokens)
                    
                    self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                    self.pointer += acts.shape[0]
                    self.token_pointer += self.cfg["model_batch_size"]

            self.pointer = 0
            # Shuffle on CPU
            self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]
            
            # Move models back to CPU
            self.model_A.cpu()
            self.model_B.cpu()
            torch.cuda.empty_cache()  # Clear any remaining GPU memory
        
        # Finally move buffer back to GPU
        self.buffer = self.buffer.to(self.cfg["device"])

    @torch.no_grad()
    def next(self):
        if self.use_cache:
            # Get batch from CPU buffer and move to GPU
            out = self.buffer[self.buffer_pointer:self.buffer_pointer + self.cfg["batch_size"]].to(self.cfg["device"]).float()
            self.buffer_pointer += self.cfg["batch_size"]
            
            # If we've reached the end, reshuffle the entire buffer on CPU
            if self.buffer_pointer + self.cfg["batch_size"] > len(self.buffer):
                self.buffer = self.buffer[torch.randperm(len(self.buffer))]
                self.buffer_pointer = 0
                out = self.buffer[self.buffer_pointer:self.buffer_pointer + self.cfg["batch_size"]].to(self.cfg["device"]).float()
                self.buffer_pointer += self.cfg["batch_size"]
        else:
            out = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]].to(self.cfg["device"]).float()
            self.pointer += self.cfg["batch_size"]
            
            # Refresh when buffer is half empty
            if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
                self.refresh()
            
        return out