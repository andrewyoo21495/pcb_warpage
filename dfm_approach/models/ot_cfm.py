#!/usr/bin/env python3
"""Module B-2: OT-Conditional Flow Matching wrapper.

Handles the OT-CFM training loop logic and ODE-based inference.

Training:
    1. Encode residuals → z₁ (using frozen CAE encoder)
    2. Sample z₀ ~ N(0, I)
    3. Minibatch OT coupling (optional)
    4. Interpolate z_t, compute target velocity v* = z₁ - z₀
    5. Predict v̂ = VelocityNet(z_t, t, c)
    6. Loss = MSE(v̂, v*)

Inference:
    1. Sample z₀ ~ N(0, I)
    2. ODE integrate: dz/dt = v̂(z_t, t, c),  t: 0→1
    3. Decode z₁ via CAE decoder → residual ε̂
"""

import torch
import torch.nn as nn

from .velocity_net import VelocityNet


# ------------------------------------------------------------------
# Minibatch OT coupling
# ------------------------------------------------------------------

def minibatch_ot_coupling(
    z0: torch.Tensor,
    z1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute minibatch optimal transport coupling.

    Finds the permutation of z1 that minimises total squared distance to z0.

    Args:
        z0: (B, D) source (noise)
        z1: (B, D) target (encoded data)

    Returns:
        z0, z1_permuted — OT-coupled pairs
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Fallback: random coupling (no OT)
        return z0, z1

    with torch.no_grad():
        cost = torch.cdist(z0, z1, p=2).pow(2)  # (B, B)
        cost_np = cost.cpu().numpy()
        _, col_idx = linear_sum_assignment(cost_np)
        col_idx = torch.from_numpy(col_idx).to(z1.device)

    return z0, z1[col_idx]


# ------------------------------------------------------------------
# ODE Solvers
# ------------------------------------------------------------------

def _euler_step(
    vel_fn, z: torch.Tensor, t: float, dt: float,
    c_global: torch.Tensor, c_spatial: torch.Tensor,
) -> torch.Tensor:
    t_tensor = torch.full((z.shape[0],), t, device=z.device, dtype=z.dtype)
    return z + dt * vel_fn(z, t_tensor, c_global, c_spatial)


def _midpoint_step(
    vel_fn, z: torch.Tensor, t: float, dt: float,
    c_global: torch.Tensor, c_spatial: torch.Tensor,
) -> torch.Tensor:
    t_tensor = torch.full((z.shape[0],), t, device=z.device, dtype=z.dtype)
    t_mid = torch.full_like(t_tensor, t + 0.5 * dt)
    k1 = vel_fn(z, t_tensor, c_global, c_spatial)
    z_mid = z + 0.5 * dt * k1
    return z + dt * vel_fn(z_mid, t_mid, c_global, c_spatial)


def _rk4_step(
    vel_fn, z: torch.Tensor, t: float, dt: float,
    c_global: torch.Tensor, c_spatial: torch.Tensor,
) -> torch.Tensor:
    B = z.shape[0]
    t0 = torch.full((B,), t, device=z.device, dtype=z.dtype)
    t_half = torch.full((B,), t + 0.5 * dt, device=z.device, dtype=z.dtype)
    t1 = torch.full((B,), t + dt, device=z.device, dtype=z.dtype)

    k1 = vel_fn(z, t0, c_global, c_spatial)
    k2 = vel_fn(z + 0.5 * dt * k1, t_half, c_global, c_spatial)
    k3 = vel_fn(z + 0.5 * dt * k2, t_half, c_global, c_spatial)
    k4 = vel_fn(z + dt * k3, t1, c_global, c_spatial)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


_ODE_SOLVERS = {
    'euler': _euler_step,
    'midpoint': _midpoint_step,
    'rk4': _rk4_step,
}


# ------------------------------------------------------------------
# OT-CFM Module
# ------------------------------------------------------------------

class OTCFM(nn.Module):
    """OT-Conditional Flow Matching for latent space generation.

    Args:
        config: dict with cfm_* keys and z_dim, c_dim.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.velocity_net = VelocityNet(config)
        self.z_dim = int(config.get('z_dim', 64))
        self.sigma_min = float(config.get('cfm_sigma_min', 0.001))
        self.use_ot = bool(config.get('cfm_use_ot', True))
        self.ode_steps = int(config.get('cfm_ode_steps', 20))
        self.ode_solver = str(config.get('cfm_ode_solver', 'midpoint'))

    def compute_loss(
        self,
        z1: torch.Tensor,
        c_global: torch.Tensor,
        c_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OT-CFM training loss.

        Args:
            z1:        (B, z_dim)  — encoded data latent
            c_global:  (B, c_dim)  — global condition
            c_spatial: (B, 128, 8, 8) — spatial condition

        Returns:
            loss: scalar MSE between predicted and target velocity
        """
        B = z1.shape[0]
        z0 = torch.randn_like(z1)

        # Minibatch OT coupling
        if self.use_ot:
            z0, z1 = minibatch_ot_coupling(z0, z1)

        # Random timestep
        t = torch.rand(B, device=z1.device, dtype=z1.dtype)
        t = t.clamp(self.sigma_min, 1.0 - self.sigma_min)

        # Interpolate
        t_expand = t.unsqueeze(-1)  # (B, 1)
        z_t = (1.0 - t_expand) * z0 + t_expand * z1

        # Target velocity (constant for linear interpolation)
        v_target = z1 - z0

        # Predict velocity
        v_pred = self.velocity_net(z_t, t, c_global, c_spatial)

        return nn.functional.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        c_global: torch.Tensor,
        c_spatial: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate latent samples via ODE integration.

        Args:
            c_global:    (1, c_dim) or (B, c_dim) — condition
            c_spatial:   (1, 128, 8, 8) or (B, ...)
            num_samples: number of samples to generate
            temperature: noise scale (1.0 = standard)

        Returns:
            z1: (num_samples, z_dim) — generated latent vectors
        """
        device = c_global.device

        # Expand conditions if needed
        if c_global.shape[0] == 1 and num_samples > 1:
            c_global = c_global.expand(num_samples, -1)
            c_spatial = c_spatial.expand(num_samples, -1, -1, -1)

        B = c_global.shape[0]

        # Initial noise
        z = torch.randn(B, self.z_dim, device=device) * temperature

        # ODE integration: t from 0 to 1
        solver_fn = _ODE_SOLVERS.get(self.ode_solver, _midpoint_step)
        dt = 1.0 / self.ode_steps

        for step in range(self.ode_steps):
            t = step * dt
            z = solver_fn(self.velocity_net, z, t, dt, c_global, c_spatial)

        return z
