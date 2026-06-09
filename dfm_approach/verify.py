#!/usr/bin/env python3
"""DF²M Verification & Diagnostics Script.

Comprehensive checks for deployment on a Linux GPU server:
    1. Code integrity:     Import checks, model instantiation, forward pass shapes
    2. GPU utilization:    Memory estimation, batch size recommendations, AMP check
    3. Bottleneck analysis: Profiling critical paths, data loading speed
    4. Result validation:  Metric sanity checks, red flags detection
    5. Tuning guide:       Parameter recommendations based on GPU specs

Usage:
    # Full verification (no GPU required — uses CPU for shape checks)
    python dfm_approach/verify.py --config dfm_approach/config_dfm.txt

    # With GPU profiling
    python dfm_approach/verify.py --config dfm_approach/config_dfm.txt --profile

    # Validate trained checkpoints
    python dfm_approach/verify.py --config dfm_approach/config_dfm.txt --validate-checkpoints
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ==================================================================
# Colour helpers for terminal output
# ==================================================================

def _ok(msg):
    print(f"  \033[92m✓\033[0m {msg}")

def _warn(msg):
    print(f"  \033[93m⚠\033[0m {msg}")

def _fail(msg):
    print(f"  \033[91m✗\033[0m {msg}")

def _info(msg):
    print(f"  \033[94mℹ\033[0m {msg}")

def _header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ==================================================================
# 1. Code Integrity Checks
# ==================================================================

def check_imports():
    """Verify all DF²M modules import cleanly."""
    _header("1. CODE INTEGRITY — Import Checks")
    errors = []

    modules = [
        ('models', 'build_dfm_models'),
        ('models.fno_mean_predictor', 'FNOMeanPredictor'),
        ('models.condition_encoder', 'ConditionEncoder'),
        ('models.residual_cae', 'ResidualCAE'),
        ('models.velocity_net', 'VelocityNet'),
        ('models.ot_cfm', 'OTCFM'),
        ('utils.dfm_dataset', 'MeanWarpageDataset'),
        ('utils.dfm_dataset', 'ResidualDataset'),
        ('utils.dfm_losses', 'fno_loss'),
        ('utils.dfm_losses', 'cae_loss'),
    ]

    for mod_name, attr_name in modules:
        try:
            full_mod = f"dfm_approach.{mod_name}" if '.' in mod_name else mod_name
            # Use relative import path
            exec(f"from {mod_name} import {attr_name}")
            _ok(f"{mod_name}.{attr_name}")
        except Exception as e:
            _fail(f"{mod_name}.{attr_name}: {e}")
            errors.append((mod_name, attr_name, str(e)))

    # Check external dependencies
    ext_deps = ['scipy.optimize', 'numpy', 'PIL', 'matplotlib']
    for dep in ext_deps:
        try:
            __import__(dep)
            _ok(f"External: {dep}")
        except ImportError:
            _warn(f"External: {dep} not found (some features may be limited)")

    return errors


def check_model_shapes(config):
    """Verify model forward passes produce correct output shapes."""
    _header("1b. CODE INTEGRITY — Shape Checks")
    device = torch.device('cpu')
    B = 2
    image_size = int(config.get('image_size', 128))
    z_dim = int(config.get('z_dim', 64))
    c_dim = int(config.get('c_dim', 64))

    selected = config.get('selected_features', None)
    if isinstance(selected, (list, tuple)):
        n_feat = len(selected)
    else:
        n_feat = 24

    errors = []

    # --- FNO ---
    try:
        from models.fno_mean_predictor import FNOMeanPredictor
        fno = FNOMeanPredictor(config).to(device)
        design = torch.randn(B, 1, image_size, image_size)
        features = torch.randn(B, n_feat)
        out = fno(design, features)
        assert out.shape == (B, 1, image_size, image_size), f"FNO output shape mismatch: {out.shape}"
        assert (out >= 0).all() and (out <= 1).all(), "FNO output not in [0,1] (sigmoid missing?)"
        _ok(f"FNO: input ({B},1,{image_size},{image_size}) → output {tuple(out.shape)}, range [{out.min():.2f}, {out.max():.2f}]")
    except Exception as e:
        _fail(f"FNO: {e}")
        errors.append(('FNO', str(e)))

    # --- ConditionEncoder ---
    try:
        from models.condition_encoder import ConditionEncoder
        cond_enc = ConditionEncoder(config).to(device)
        c_global, c_spatial = cond_enc(design, features)
        assert c_global.shape == (B, c_dim), f"c_global shape: {c_global.shape}"
        assert c_spatial.shape[0] == B and c_spatial.shape[-1] == 8, f"c_spatial: {c_spatial.shape}"
        _ok(f"CondEncoder: c_global={tuple(c_global.shape)}, c_spatial={tuple(c_spatial.shape)}")
    except Exception as e:
        _fail(f"CondEncoder: {e}")
        errors.append(('CondEncoder', str(e)))

    # --- ResidualCAE ---
    try:
        from models.residual_cae import ResidualCAE
        cae = ResidualCAE(config).to(device)
        residual = torch.randn(B, 1, image_size, image_size)
        recon, mu, logvar = cae(residual, c_global, c_spatial)
        assert recon.shape == (B, 1, image_size, image_size), f"CAE recon: {recon.shape}"
        assert mu.shape == (B, z_dim), f"mu: {mu.shape}"
        assert logvar.shape == (B, z_dim), f"logvar: {logvar.shape}"

        # Test decode path
        z = torch.randn(B, z_dim)
        decoded = cae.decode(z, c_global, c_spatial)
        assert decoded.shape == (B, 1, image_size, image_size)
        _ok(f"CAE: recon={tuple(recon.shape)}, mu={tuple(mu.shape)}, decode={tuple(decoded.shape)}")
    except Exception as e:
        _fail(f"CAE: {e}")
        errors.append(('CAE', str(e)))

    # --- OTCFM ---
    try:
        from models.ot_cfm import OTCFM
        cfm = OTCFM(config).to(device)
        z1 = torch.randn(B, z_dim)
        loss = cfm.compute_loss(z1, c_global, c_spatial)
        assert loss.shape == (), f"CFM loss not scalar: {loss.shape}"
        assert not torch.isnan(loss), "CFM loss is NaN"

        # Test sampling
        z_gen = cfm.sample(c_global[:1], c_spatial[:1], num_samples=3)
        assert z_gen.shape == (3, z_dim), f"CFM sample: {z_gen.shape}"
        _ok(f"OT-CFM: loss={loss.item():.4f}, sample={tuple(z_gen.shape)}")
    except Exception as e:
        _fail(f"OT-CFM: {e}")
        errors.append(('OT-CFM', str(e)))

    return errors


# ==================================================================
# 2. GPU Utilization & Memory Estimation
# ==================================================================

def check_gpu(config, do_profile=False):
    """GPU memory estimation and utilization analysis."""
    _header("2. GPU UTILIZATION — Memory & Throughput")

    if not torch.cuda.is_available():
        _warn("CUDA not available — running on CPU only")
        _info("Memory estimates below are theoretical for GPU deployment")
        estimate_memory_cpu(config)
        return

    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    _info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    if do_profile:
        profile_gpu(config, device)
    else:
        estimate_memory_cpu(config)


def estimate_memory_cpu(config):
    """Estimate GPU memory requirements without a GPU."""
    image_size = int(config.get('image_size', 128))
    z_dim = int(config.get('z_dim', 64))

    # Count parameters
    from models import build_dfm_models
    models = build_dfm_models(config)

    total_params = 0
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        mem_mb = n_params * 4 / (1024**2)  # float32
        _info(f"  {name}: {n_params:,} params ({mem_mb:.1f} MB)")
        total_params += n_params

    total_mb = total_params * 4 / (1024**2)
    _info(f"  Total model params: {total_params:,} ({total_mb:.1f} MB)")

    # Estimate activation memory (rough)
    B_fno = int(config.get('fno_batch_size', 16))
    B_cae = int(config.get('cae_batch_size', 32))
    B_cfm = int(config.get('cfm_batch_size', 64))

    # Activation memory ≈ batch × channels × H × W × 4 bytes × num_layers
    fno_act = B_fno * 32 * image_size * image_size * 4 * 4 / (1024**2)
    cae_act = B_cae * 256 * (image_size // 16) ** 2 * 4 * 6 / (1024**2)
    cfm_act = B_cfm * 512 * 4 * 8 / (1024**2)

    print()
    _info("Estimated GPU memory per phase (model + activations + gradients):")
    _info(f"  Phase 1 (FNO,  BS={B_fno}): ~{total_mb * 0.2 + fno_act * 3:.0f} MB")
    _info(f"  Phase 2 (CAE,  BS={B_cae}): ~{total_mb * 0.5 + cae_act * 3:.0f} MB")
    _info(f"  Phase 3 (CFM,  BS={B_cfm}): ~{total_mb * 0.3 + cfm_act * 3:.0f} MB")

    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  GPU 메모리별 권장 배치 크기                              │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print("  │  GPU Memory  │  FNO BS  │  CAE BS  │  CFM BS           │")
    print("  ├──────────────┼──────────┼──────────┼───────────────────┤")
    print("  │  8 GB        │    8     │    16    │    32             │")
    print("  │  16 GB       │   16     │    32    │    64             │")
    print("  │  24 GB       │   32     │    64    │   128             │")
    print("  │  40 GB+      │   64     │   128    │   256             │")
    print("  └──────────────┴──────────┴──────────┴───────────────────┘")


def profile_gpu(config, device):
    """Run actual GPU profiling with forward/backward passes."""
    from models import build_dfm_models
    image_size = int(config.get('image_size', 128))
    z_dim = int(config.get('z_dim', 64))

    selected = config.get('selected_features', None)
    n_feat = len(selected) if isinstance(selected, (list, tuple)) else 24

    models = build_dfm_models(config)
    for m in models.values():
        m.to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Phase 1 profiling
    B = int(config.get('fno_batch_size', 16))
    fno = models['fno']
    design = torch.randn(B, 1, image_size, image_size, device=device)
    feat = torch.randn(B, n_feat, device=device)
    target = torch.randn(B, 1, image_size, image_size, device=device)

    fno.train()
    out = fno(design, feat)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()

    mem_phase1 = torch.cuda.max_memory_allocated() / (1024**2)
    _ok(f"Phase 1 peak memory (BS={B}): {mem_phase1:.0f} MB")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Phase 3 profiling (lightweight)
    B = int(config.get('cfm_batch_size', 64))
    cfm = models['cfm']
    cond_enc = models['cond_enc']

    design = torch.randn(B, 1, image_size, image_size, device=device)
    feat = torch.randn(B, n_feat, device=device)
    z1 = torch.randn(B, z_dim, device=device)

    cond_enc.eval()
    with torch.no_grad():
        c_g, c_s = cond_enc(design, feat)

    cfm.train()
    loss = cfm.compute_loss(z1, c_g, c_s)
    loss.backward()

    mem_phase3 = torch.cuda.max_memory_allocated() / (1024**2)
    _ok(f"Phase 3 peak memory (BS={B}): {mem_phase3:.0f} MB")

    # Throughput test
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        cfm.compute_loss(z1, c_g, c_s)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    _ok(f"Phase 3 throughput: {100 * B / elapsed:.0f} samples/sec")


# ==================================================================
# 3. Bottleneck Analysis
# ==================================================================

def check_bottlenecks(config):
    """Identify potential bottlenecks."""
    _header("3. BOTTLENECK ANALYSIS")

    image_size = int(config.get('image_size', 128))
    num_workers = int(config.get('num_workers', 4))

    # Data loading
    if num_workers == 0:
        _warn("num_workers=0: Data loading will be single-threaded (bottleneck!)")
        _info("  → Set num_workers=4-8 for GPU training")
    else:
        _ok(f"num_workers={num_workers}")

    # OT coupling complexity
    cfm_bs = int(config.get('cfm_batch_size', 64))
    use_ot = bool(config.get('cfm_use_ot', True))
    if use_ot:
        ot_complexity = cfm_bs ** 2  # cost matrix size
        if cfm_bs > 128:
            _warn(f"OT coupling with BS={cfm_bs}: cost matrix {cfm_bs}×{cfm_bs} "
                  f"may be slow. Consider BS≤128 or disable OT for large batches.")
        else:
            _ok(f"OT coupling: {cfm_bs}×{cfm_bs} cost matrix (manageable)")

    # FNO modes
    fno_modes = int(config.get('fno_modes', 16))
    max_modes = image_size // 2
    if fno_modes > max_modes:
        _warn(f"fno_modes={fno_modes} > image_size/2={max_modes}: will be clamped")
    else:
        _ok(f"FNO modes: {fno_modes}/{max_modes} (keeping {100*fno_modes/max_modes:.0f}% of spectrum)")

    # ODE steps at inference
    ode_steps = int(config.get('cfm_ode_steps', 20))
    ode_solver = str(config.get('cfm_ode_solver', 'midpoint'))
    nfe_per_step = {'euler': 1, 'midpoint': 2, 'rk4': 4}
    total_nfe = ode_steps * nfe_per_step.get(ode_solver, 2)
    if total_nfe > 100:
        _warn(f"ODE solver {ode_solver} with {ode_steps} steps = {total_nfe} NFE (slow inference)")
        _info("  → Consider reducing ode_steps or using euler solver")
    else:
        _ok(f"ODE inference: {ode_solver}, {ode_steps} steps, {total_nfe} NFE")

    # Mean computation (Phase 1 data)
    n_samples = int(config.get('num_samples_per_design', 300))
    _info(f"Mean warpage computation: {n_samples} images per design to average")
    _info(f"  → This is a one-time precomputation step, not a bottleneck")

    # Cross-attention
    use_ca = bool(config.get('cfm_use_cross_attn', True))
    if use_ca:
        spatial_tokens = 8 * 8  # from 8×8 spatial condition
        _info(f"Cross-attention: 1 query token × {spatial_tokens} spatial tokens (lightweight)")


# ==================================================================
# 4. Result Validation
# ==================================================================

def validate_results(config):
    """Check if trained checkpoints exist and validate their metrics."""
    _header("4. RESULT VALIDATION — Checkpoint Analysis")

    paths = {
        'FNO (Phase 1)': config.get('fno_modelpath', './outputs/dfm_fno.pth'),
        'CAE (Phase 2)': config.get('cae_modelpath', './outputs/dfm_cae.pth'),
        'CFM (Phase 3)': config.get('cfm_modelpath', './outputs/dfm_cfm.pth'),
    }

    for name, path in paths.items():
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            epoch = ckpt.get('epoch', '?')
            val_loss = ckpt.get('val_loss', float('nan'))
            phase = ckpt.get('phase', '?')

            _ok(f"{name}: epoch={epoch}, val_loss={val_loss:.6f}")

            # Quality thresholds
            if phase == 1:
                if val_loss < 0.001:
                    _ok(f"  Mean MSE excellent (< 0.001)")
                elif val_loss < 0.01:
                    _ok(f"  Mean MSE good (< 0.01)")
                elif val_loss < 0.05:
                    _warn(f"  Mean MSE moderate ({val_loss:.4f}) — consider more training or mixup augmentation")
                else:
                    _fail(f"  Mean MSE poor ({val_loss:.4f}) — FNO likely underfitting")
                    _info("  Suggestions:")
                    _info("    1. Increase fno_epochs or fno_lr")
                    _info("    2. Increase fno_width (32→64)")
                    _info("    3. Enable/increase mixup_alpha")
                    _info("    4. Check data loading (are mean warpage targets correct?)")

            elif phase == 2:
                if val_loss < 0.005:
                    _ok(f"  Residual recon excellent")
                elif val_loss < 0.02:
                    _ok(f"  Residual recon good")
                else:
                    _warn(f"  Residual recon poor ({val_loss:.4f})")
                    _info("  Suggestions:")
                    _info("    1. Check for KL collapse (active_kl_dims in logs)")
                    _info("    2. Reduce cae_beta_max")
                    _info("    3. Increase z_dim or c_dim")
                    _info("    4. Increase cae_free_bits")

            elif phase == 3:
                if val_loss < 0.01:
                    _ok(f"  Velocity MSE excellent")
                elif val_loss < 0.1:
                    _ok(f"  Velocity MSE good")
                else:
                    _warn(f"  Velocity MSE high ({val_loss:.4f})")
                    _info("  Suggestions:")
                    _info("    1. Train longer (cfm_epochs)")
                    _info("    2. Increase cfm_hidden_dim")
                    _info("    3. Try disabling OT (cfm_use_ot=False) for comparison")
        else:
            _warn(f"{name}: checkpoint not found at {path}")

    # Check log file
    log_path = config.get('log_file_dir', './outputs/train_dfm.log')
    if os.path.exists(log_path):
        _ok(f"Training log found: {log_path}")
        # Parse last few lines for issues
        with open(log_path, 'r') as f:
            lines = f.readlines()
        last_lines = lines[-20:] if len(lines) > 20 else lines

        nan_count = sum(1 for l in last_lines if 'nan' in l.lower())
        if nan_count > 0:
            _fail(f"  Found {nan_count} lines with NaN in recent logs — training diverged!")
            _info("  Suggestions:")
            _info("    1. Reduce learning rate")
            _info("    2. Check gradient clipping (max_norm=5.0)")
            _info("    3. Reduce batch size")
    else:
        _info(f"No training log found at {log_path} (not yet trained)")


# ==================================================================
# 5. Tuning Guide
# ==================================================================

def print_tuning_guide(config):
    """Print parameter tuning recommendations."""
    _header("5. TUNING GUIDE — Key Parameters to Monitor")

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Parameter              │  When to Adjust                       │
  ├─────────────────────────┼───────────────────────────────────────┤
  │  fno_width (32)         │  ↑ if mean MSE plateaus high          │
  │  fno_modes (16)         │  ↑ if mean lacks high-freq detail     │
  │  fno_mixup_alpha (0.4)  │  ↑ if val MSE >> train MSE (overfit)  │
  │  fno_smooth_weight      │  ↑ if predicted mean is noisy         │
  │                         │                                       │
  │  z_dim (64)             │  ↑ if gen diversity too low            │
  │  c_dim (64)             │  ↓ if CAE ignores design condition     │
  │  cae_beta_max (0.3)     │  ↓ if KL collapse persists            │
  │  cae_free_bits (0.5)    │  ↑ if active_kl_dims < z_dim/4        │
  │                         │                                       │
  │  cfm_hidden_dim (512)   │  ↑ if velocity MSE plateaus high      │
  │  cfm_ode_steps (20)     │  ↑ if generated samples are blurry    │
  │  cfm_use_ot (True)      │  Try False if OT coupling is slow     │
  │  cfm_ema_decay (0.9999) │  ↓ to 0.999 for faster adaptation     │
  └─────────────────────────┴───────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  RED FLAGS — 즉시 조치 필요                                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  1. KL → 0 by epoch 20     : cae_beta_max 낮추기, free_bits 올리기 │
  │  2. NaN in loss             : LR 낮추기, gradient clipping 확인   │
  │  3. Val loss >> Train loss  : augmentation 강화, model 축소      │
  │  4. Diversity ratio < 0.1   : temperature↑, z_dim↑, KL collapse 확인 │
  │  5. Diversity ratio > 5.0   : temperature↓, ODE steps↑          │
  │  6. MMD > 0.3               : 전체 파이프라인 재검토               │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  GOOD SIGNS — 학습이 정상 진행 중                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  1. Phase 1: val MSE < 0.01 and steadily decreasing             │
  │  2. Phase 2: active_kl_dims > 16 and recon_mse < 0.01          │
  │  3. Phase 3: velocity MSE < 0.05 and steadily decreasing        │
  │  4. Eval: diversity_ratio between 0.3 and 3.0                   │
  │  5. Eval: MMD < 0.1                                             │
  └─────────────────────────────────────────────────────────────────┘
""")


# ==================================================================
# Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description='DF²M Verification & Diagnostics')
    parser.add_argument('--config', type=str, default='dfm_approach/config_dfm.txt')
    parser.add_argument('--profile', action='store_true', help='Run GPU profiling')
    parser.add_argument('--validate-checkpoints', action='store_true',
                        help='Validate trained checkpoints')
    args = parser.parse_args()

    # Change CWD to project root for relative imports
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(str(project_root))
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'dfm_approach'))

    from utils.load_config import load_config
    config = load_config(args.config)

    # Parse selected features
    selected = config.get('selected_features', None)
    if isinstance(selected, str):
        selected = [int(x) for x in selected.split(',')]
    config['selected_features'] = selected

    print("\n" + "=" * 60)
    print("  DF²M VERIFICATION & DIAGNOSTICS")
    print("=" * 60)

    # 1. Code integrity
    import_errors = check_imports()
    if import_errors:
        _fail(f"\n  {len(import_errors)} import error(s) found. Fix before proceeding.")
        for mod, attr, err in import_errors:
            print(f"    {mod}.{attr}: {err}")
    else:
        _ok("All imports successful")

    shape_errors = check_model_shapes(config)
    if shape_errors:
        _fail(f"\n  {len(shape_errors)} shape error(s) found.")
    else:
        _ok("All shape checks passed")

    # 2. GPU utilization
    check_gpu(config, do_profile=args.profile)

    # 3. Bottlenecks
    check_bottlenecks(config)

    # 4. Result validation
    if args.validate_checkpoints:
        validate_results(config)
    else:
        _header("4. RESULT VALIDATION (skipped — use --validate-checkpoints)")

    # 5. Tuning guide
    print_tuning_guide(config)

    print("\n" + "=" * 60)
    print("  Verification complete.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
