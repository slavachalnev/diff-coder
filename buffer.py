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

    def __init__(self, cfg, model_A, model_B, all_tokens):
        assert model_A.cfg.d_model == model_B.cfg.d_model
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        )#.to(cfg["device"]) # hardcoding 2 for model diffing
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        
        # Cache related attributes
        self.cache_dir = Path(cfg["cache_dir"])
        self.use_cache = cfg.get("use_cache", False)
        self.cache_size = cfg.get("cache_size_gb", 40) * (1024**3) # Convert GB to bytes
        self.chunk_size = min(self.buffer_size * 10, self.cache_size // (2 * model_A.cfg.d_model * 2)) # Size that fits in memory
        self.current_chunk = 0
        self.total_chunks = self.cache_size // (self.chunk_size * 2 * model_A.cfg.d_model * 2) # 2 for both models, 2 for bfloat16
        
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)
        
        self.normalisation_factor = torch.tensor(
            [estimated_norm_scaling_factor_A, estimated_norm_scaling_factor_B],
            device="cuda:0",
            dtype=torch.float32,
        )

        if self.use_cache:
            if not self._cache_exists():
                print(f"Cache not found at {self.cache_dir}. Generating cache...")
                self._generate_cache()
            else:
                print(f"Loading from cache at {self.cache_dir}")
                self._load_cache_metadata()
        else:
            self.refresh()

    def _cache_exists(self):
        """Check if cache exists and is complete"""
        if not self.cache_dir.exists():
            return False
        metadata_path = self.cache_dir / "metadata.pt"
        if not metadata_path.exists():
            return False
        # Check if all expected chunk files exist
        for i in range(self.total_chunks):
            if not (self.cache_dir / f"chunk_{i}.pt").exists():
                return False
        return True

    def _save_cache_metadata(self):
        """Save normalization factors and other metadata"""
        metadata = {
            "total_chunks": self.total_chunks,
            "chunk_size": self.chunk_size,
            "d_model": self.model_A.cfg.d_model,
        }
        torch.save(metadata, self.cache_dir / "metadata.pt")

    def _load_cache_metadata(self):
        """Load normalization factors and other metadata"""
        metadata = torch.load(self.cache_dir / "metadata.pt")
        self.total_chunks = metadata["total_chunks"]
        self.chunk_size = metadata["chunk_size"]
        assert self.model_A.cfg.d_model == metadata["d_model"], "Model dimension mismatch with cache"

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
        
        return acts

    @torch.no_grad()
    def _generate_cache(self):
        """Generate and save activation cache to disk"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_buffer = torch.zeros(
            (self.chunk_size, 2, self.model_A.cfg.d_model),
            dtype=torch.bfloat16
        )
        
        total_processed = 0
        chunk_idx = 0
        
        print(f"Generating cache with {self.total_chunks} chunks of size {self.chunk_size}")
        
        # Move models to device at start
        self.model_A.to(self.cfg["device"])
        self.model_B.to(self.cfg["device"])
        
        with torch.autocast("cuda", torch.bfloat16):
            while total_processed < self.cache_size and chunk_idx < self.total_chunks:
                chunk_pointer = 0
                
                # Fill current chunk
                for _ in tqdm.trange(0, self.chunk_size, self.cfg["model_batch_size"]):
                    if self.token_pointer >= len(self.all_tokens):
                        break
                            
                    tokens = self.all_tokens[
                        self.token_pointer : min(
                            self.token_pointer + self.cfg["model_batch_size"],
                            len(self.all_tokens)
                        )
                    ]
                        
                    acts = self._get_batch_activations(tokens)
                        
                    # Store in chunk buffer
                    chunk_buffer[chunk_pointer:chunk_pointer + acts.shape[0]] = acts
                    chunk_pointer += acts.shape[0]
                    self.token_pointer += self.cfg["model_batch_size"]
                        
                    if chunk_pointer >= self.chunk_size:
                        break
                
                # Save filled chunk
                if chunk_pointer > 0:  # Only save if we actually filled something
                    torch.save(
                        chunk_buffer[:chunk_pointer], 
                        self.cache_dir / f"chunk_{chunk_idx}.pt"
                    )
                    total_processed += chunk_pointer * 2 * self.model_A.cfg.d_model * 2
                    chunk_idx += 1
                    
                if self.token_pointer >= len(self.all_tokens):
                    break
        
        # Move models back to CPU at end
        self.model_A.cpu()
        self.model_B.cpu()
        torch.cuda.empty_cache()
        
        self._save_cache_metadata()
        print(f"Generated {chunk_idx} chunks, total size: {total_processed / (1024**3):.2f} GB")

    def _load_next_chunk(self):
        """Load next chunk from cache"""
        chunk = torch.load(self.cache_dir / f"chunk_{self.current_chunk}.pt")
        self.current_chunk = (self.current_chunk + 1) % self.total_chunks
        return chunk

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
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        self.pointer += self.cfg["batch_size"]
        
        # Refresh when buffer is half empty
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()
            
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out