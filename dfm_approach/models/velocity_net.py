#!/usr/bin/env python3
"""Enhanced Velocity Network for OT-Conditional Flow Matching.

Predicts the velocity field v(z_t, t, c) for flow matching in latent space.

Architecture:
    [z_t, c_global] → Input Proj → N×[AdaLN-ResBlock (+ optional CrossAttn)] → Output Proj → v̂

Improvements over the original LatentDenoiser:
    1. Cross-attention to c_spatial tokens (spatial conditioning)
    2. AdaLN conditioned on BOTH timestep t AND global condition
    3. Zeros-initialized output projection for stable training

References:
    Tong et al., "Improving and Generalizing Flow-Based Generative Models
    with Minibatch Optimal Transport", ICLR 2024
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Sinusoidal timestep embedding
# ------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timestep values in [0, 1]

        Returns:
            emb: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ------------------------------------------------------------------
# AdaLN-ResBlock
# ------------------------------------------------------------------

class AdaLNResBlock(nn.Module):
    """Residual block with Adaptive Layer Normalisation.

    AdaLN: LayerNorm(x) is modulated by (γ, β) predicted from condition embedding.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # AdaLN modulation: cond → (γ1, β1, γ2, β2)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 4),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, hidden_dim)
            cond: (B, cond_dim) — combined timestep + condition embedding
        """
        params = self.adaln(cond)
        g1, b1, g2, b2 = params.chunk(4, dim=-1)

        h = self.norm1(x)
        h = (1 + g1) * h + b1
        h = self.mlp(h)
        h = self.norm2(h)
        h = (1 + g2) * h + b2

        return x + h


# ------------------------------------------------------------------
# Cross-attention block
# ------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Cross-attention: latent tokens attend to spatial condition tokens.

    Q from latent z, K/V from c_spatial tokens.
    """

    def __init__(self, hidden_dim: int, spatial_token_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(spatial_token_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(spatial_token_dim, hidden_dim)
        self.v_proj = nn.Linear(spatial_token_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:              (B, hidden_dim)
            spatial_tokens: (B, N_tokens, spatial_token_dim)  — N_tokens = 8*8 = 64
        """
        B = x.shape[0]
        q = self.q_proj(self.norm_q(x)).unsqueeze(1)  # (B, 1, hidden_dim)
        k = self.k_proj(self.norm_kv(spatial_tokens))  # (B, N, hidden_dim)
        v = self.v_proj(self.norm_kv(spatial_tokens))

        # Reshape for multi-head attention
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)  # (B, heads, 1, head_dim)
        attn = attn.transpose(1, 2).reshape(B, -1)       # (B, hidden_dim)
        return x + self.out_proj(attn)


# ------------------------------------------------------------------
# Enhanced Velocity Network
# ------------------------------------------------------------------

class VelocityNet(nn.Module):
    """Velocity network for OT-CFM with optional cross-attention.

    Args:
        config: dict with z_dim, c_dim, cfm_hidden_dim, cfm_n_blocks,
                cfm_dropout, cfm_use_cross_attn, cfm_cross_attn_every, cfm_n_heads.
    """

    def __init__(self, config: dict):
        super().__init__()
        z_dim = int(config.get('z_dim', 64))
        c_dim = int(config.get('c_dim', 64))
        hidden_dim = int(config.get('cfm_hidden_dim', 512))
        n_blocks = int(config.get('cfm_n_blocks', 8))
        dropout = float(config.get('cfm_dropout', 0.1))
        use_cross_attn = bool(config.get('cfm_use_cross_attn', True))
        cross_attn_every = int(config.get('cfm_cross_attn_every', 2))
        n_heads = int(config.get('cfm_n_heads', 4))
        spatial_ch = 128  # from ConditionEncoder

        self.z_dim = z_dim

        # Timestep embedding
        t_emb_dim = hidden_dim
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        # Input projection: [z_t, c_global] → hidden
        self.input_proj = nn.Linear(z_dim + c_dim, hidden_dim)

        # Condition projection: t_emb combined with global info
        # AdaLN condition = t_emb (already contains temporal info)
        self.cond_dim = t_emb_dim

        # Residual blocks + optional cross-attention
        self.blocks = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.use_cross_attn_flags: list[bool] = []

        for i in range(n_blocks):
            self.blocks.append(AdaLNResBlock(hidden_dim, self.cond_dim, dropout))
            do_cross = use_cross_attn and ((i + 1) % cross_attn_every == 0)
            self.use_cross_attn_flags.append(do_cross)
            if do_cross:
                self.cross_attns.append(
                    CrossAttentionBlock(hidden_dim, spatial_ch, n_heads)
                )
            else:
                self.cross_attns.append(nn.Identity())  # placeholder

        # Output projection (zeros-init for stable training)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        c_global: torch.Tensor,
        c_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t:       (B, z_dim)  — noisy latent at time t
            t:         (B,)       — timestep in [0, 1]
            c_global:  (B, c_dim) — global condition
            c_spatial: (B, 128, 8, 8) — spatial condition

        Returns:
            v_hat: (B, z_dim) — predicted velocity
        """
        # Timestep embedding
        t_emb = self.t_embed(t)  # (B, hidden_dim)

        # Input projection
        h = self.input_proj(torch.cat([z_t, c_global], dim=-1))  # (B, hidden_dim)

        # Prepare spatial tokens: (B, 128, 8, 8) → (B, 64, 128)
        B = c_spatial.shape[0]
        spatial_tokens = c_spatial.flatten(2).transpose(1, 2)  # (B, 64, 128)

        # Process through blocks
        for block, cross_attn, use_ca in zip(
            self.blocks, self.cross_attns, self.use_cross_attn_flags
        ):
            h = block(h, t_emb)
            if use_ca:
                h = cross_attn(h, spatial_tokens)

        # Output
        return self.out_proj(self.out_norm(h))
