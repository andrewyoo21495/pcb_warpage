#!/usr/bin/env python3
"""Analyse training and evaluation results for PCB Warpage models.

Parses training logs (CVAE/LDM/LFM) and evaluation logs across all folds,
produces a diagnostic report with issue detection and improvement suggestions,
and optionally generates a summary visualisation.

Usage:
  python analyze_results.py                         # analyse all models
  python analyze_results.py --model cvae            # CVAE only
  python analyze_results.py --outputs-dir outputs/  # custom outputs dir
  python analyze_results.py --no-plot               # skip visualisation
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MODEL_TYPES: list[str] = ['cvae', 'ldm', 'lfm']

# Diagnostic thresholds
DIVERSITY_RATIO_LOW: float = 0.3
DIVERSITY_RATIO_HIGH: float = 3.0
MMD_GOOD: float = 0.1
MMD_HIGH: float = 0.3
ACTIVE_DIM_FRAC: float = 0.25
OVERFIT_RATIO: float = 2.0
CONVERGENCE_WINDOW: int = 10
CONVERGENCE_DROP_FRAC: float = 0.02


# ------------------------------------------------------------------
# Regex patterns -- derived from train.py and evaluate.py format strings
# ------------------------------------------------------------------

# CVAE training epoch (with active dims)
RE_CVAE_EPOCH = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+'
    r'beta=([\d.]+)\s+'
    r'train\[loss=([\d.]+)\s+recon=([\d.]+)\s+kl=([\d.]+)\s+active=(\d+)/(\d+)\]\s+'
    r'val\[loss=([\d.]+)\s+recon=([\d.]+)\s+kl=([\d.]+)\s+active=(\d+)/(\d+)\]\s+'
    r'lr=([\d.eE+-]+)'
)

# CVAE training epoch (legacy format without active dims)
RE_CVAE_EPOCH_LEGACY = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+'
    r'beta=([\d.]+)\s+'
    r'train\[loss=([\d.]+)\s+recon=([\d.]+)\s+kl=([\d.]+)\]\s+'
    r'val\[loss=([\d.]+)\s+recon=([\d.]+)\s+kl=([\d.]+)\]\s+'
    r'lr=([\d.eE+-]+)'
)

# LDM training epoch
RE_LDM_EPOCH = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+'
    r'train_noise-pred=([\d.]+)\s+'
    r'val_noise-pred=([\d.]+)\s+'
    r'lr=([\d.eE+-]+)'
)

# LFM training epoch
RE_LFM_EPOCH = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+'
    r'train_velocity=([\d.]+)\s+'
    r'val_velocity=([\d.]+)\s+'
    r'lr=([\d.eE+-]+)'
)

# Checkpoint and completion markers
RE_CHECKPOINT_CVAE = re.compile(r'->\s*Checkpoint saved \(val_recon=([\d.]+)\)')
RE_CHECKPOINT_LATENT = re.compile(
    r'->\s*Checkpoint saved \(val_(?:noise-pred|velocity)=([\d.]+)\)')
RE_EARLY_STOP = re.compile(r'Early stop at epoch (\d+)')
RE_TRAINING_COMPLETE = re.compile(r'Training complete\.')
RE_BEST_VAL = re.compile(r'Best val (?:recon|noise-pred|velocity) loss:\s*([\d.]+)')

# Evaluation log patterns
RE_EVAL_FOLD = re.compile(r'Fold (\d+)\s+--\s+held-out design:\s+(\S+)')
RE_TRAIN_RECON_MSE = re.compile(r'Train Recon MSE\s*:\s*([\d.]+)')
RE_REAL_DIV_TRAIN = re.compile(r'Real Diversity\s*:\s*([\d.]+)\s+\(baseline')
RE_VAL_RECON_MSE = re.compile(r'Val Recon MSE\s*:\s*([\d.]+)')
RE_ACTIVE_KL = re.compile(r'Active KL dims\s*:\s*(\d+)/(\d+)')
RE_REAL_DIV_VAL = re.compile(r'Real Diversity\s*:\s*([\d.]+)\s+\(val split')
RE_GEN_DIV = re.compile(
    r'Gen\s+Diversity\s*:\s*([\d.]+)\s+\(ratio vs real:\s*([\d.]+)x')
RE_MMD = re.compile(r'MMD\s*:\s*([\d.]+)')


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class TrainingEpoch:
    """Metrics for a single training epoch."""
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    beta: float | None = None
    train_recon: float | None = None
    train_kl: float | None = None
    train_active: int | None = None
    val_recon: float | None = None
    val_kl: float | None = None
    val_active: int | None = None
    z_dim: int | None = None
    lr: float | None = None


@dataclass
class TrainingRun:
    """Parsed results of a complete training run."""
    model_type: str
    fold: int
    epochs: list[TrainingEpoch] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float | None = None
    early_stopped: bool = False
    early_stop_epoch: int | None = None
    completed: bool = False


@dataclass
class EvalResult:
    """Parsed results from an evaluation log for one fold."""
    model_type: str
    fold: int
    design_name: str = ''
    train_recon_mse: float | None = None
    train_real_diversity: float | None = None
    val_recon_mse: float | None = None
    active_dims: int | None = None
    z_dim: int | None = None
    real_diversity: float | None = None
    gen_diversity: float | None = None
    diversity_ratio: float | None = None
    mmd: float | None = None


@dataclass
class Issue:
    """A detected diagnostic issue with recommendation."""
    severity: str           # 'CRITICAL' or 'WARNING'
    model_type: str
    fold: int | None
    description: str
    recommendation: str


# ------------------------------------------------------------------
# Report writer -- simultaneous stdout + buffer
# ------------------------------------------------------------------

class ReportWriter:
    """Write report lines to both stdout and a buffer for file output."""

    def __init__(self):
        self.lines: list[str] = []

    def write(self, text: str = '') -> None:
        print(text)
        self.lines.append(text)

    def get_text(self) -> str:
        return '\n'.join(self.lines)


# ------------------------------------------------------------------
# Log discovery
# ------------------------------------------------------------------

def discover_log_files(
    outputs_dir: Path,
    model_types: list[str],
) -> dict[str, dict[str, dict[int, Path]]]:
    """Scan outputs directory for training and evaluation log files.

    Returns:
        {model_type: {'train': {fold: Path}, 'eval': {fold: Path}}}
    """
    result: dict[str, dict[str, dict[int, Path]]] = {}
    re_train = re.compile(r'^fold(\d+)\.log$')
    re_eval = re.compile(r'^eval_fold(\d+)\.log$')

    for model in model_types:
        train_logs: dict[int, Path] = {}
        eval_logs: dict[int, Path] = {}
        log_dir = outputs_dir / f'logs_{model}'

        if log_dir.is_dir():
            for f in sorted(log_dir.iterdir()):
                m = re_train.match(f.name)
                if m:
                    train_logs[int(m.group(1))] = f
                    continue
                m = re_eval.match(f.name)
                if m:
                    eval_logs[int(m.group(1))] = f

        result[model] = {'train': train_logs, 'eval': eval_logs}

    return result


# ------------------------------------------------------------------
# Training log parsing
# ------------------------------------------------------------------

def parse_cvae_training_log(log_path: Path) -> TrainingRun:
    """Parse a CVAE training log file into structured data."""
    run = TrainingRun(model_type='cvae', fold=-1)
    text = log_path.read_text(encoding='utf-8', errors='replace')

    best_val_recon: float | None = None

    for line in text.splitlines():
        # Try full format (with active dims) first
        m = RE_CVAE_EPOCH.search(line)
        if m:
            epoch = TrainingEpoch(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                train_loss=float(m.group(4)),
                val_loss=float(m.group(9)),
                beta=float(m.group(3)),
                train_recon=float(m.group(5)),
                train_kl=float(m.group(6)),
                train_active=int(m.group(7)),
                val_recon=float(m.group(10)),
                val_kl=float(m.group(11)),
                val_active=int(m.group(12)),
                z_dim=int(m.group(8)),
                lr=float(m.group(14)),
            )
            run.epochs.append(epoch)
            continue

        # Try legacy format (without active dims)
        m = RE_CVAE_EPOCH_LEGACY.search(line)
        if m:
            epoch = TrainingEpoch(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                train_loss=float(m.group(4)),
                val_loss=float(m.group(7)),
                beta=float(m.group(3)),
                train_recon=float(m.group(5)),
                train_kl=float(m.group(6)),
                val_recon=float(m.group(8)),
                val_kl=float(m.group(9)),
                lr=float(m.group(10)),
            )
            run.epochs.append(epoch)
            continue

        m = RE_CHECKPOINT_CVAE.search(line)
        if m:
            val_recon = float(m.group(1))
            if best_val_recon is None or val_recon < best_val_recon:
                best_val_recon = val_recon
                run.best_val_loss = val_recon
                run.best_epoch = run.epochs[-1].epoch if run.epochs else None

        m = RE_EARLY_STOP.search(line)
        if m:
            run.early_stopped = True
            run.early_stop_epoch = int(m.group(1))

        if RE_TRAINING_COMPLETE.search(line):
            run.completed = True

        m = RE_BEST_VAL.search(line)
        if m and run.best_val_loss is None:
            run.best_val_loss = float(m.group(1))

    return run


def parse_latent_training_log(log_path: Path, model_type: str) -> TrainingRun:
    """Parse an LDM or LFM training log file into structured data."""
    run = TrainingRun(model_type=model_type, fold=-1)
    text = log_path.read_text(encoding='utf-8', errors='replace')

    pattern = RE_LDM_EPOCH if model_type == 'ldm' else RE_LFM_EPOCH
    best_loss: float | None = None

    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            epoch = TrainingEpoch(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                train_loss=float(m.group(3)),
                val_loss=float(m.group(4)),
                lr=float(m.group(5)),
            )
            run.epochs.append(epoch)
            continue

        m = RE_CHECKPOINT_LATENT.search(line)
        if m:
            val_loss = float(m.group(1))
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                run.best_val_loss = val_loss
                run.best_epoch = run.epochs[-1].epoch if run.epochs else None

        m = RE_EARLY_STOP.search(line)
        if m:
            run.early_stopped = True
            run.early_stop_epoch = int(m.group(1))

        if RE_TRAINING_COMPLETE.search(line):
            run.completed = True

        m = RE_BEST_VAL.search(line)
        if m and run.best_val_loss is None:
            run.best_val_loss = float(m.group(1))

    return run


# ------------------------------------------------------------------
# Evaluation log parsing
# ------------------------------------------------------------------

def parse_eval_log(log_path: Path, model_type: str) -> EvalResult:
    """Parse an evaluation log file into structured data."""
    result = EvalResult(model_type=model_type, fold=-1)
    text = log_path.read_text(encoding='utf-8', errors='replace')

    # Track whether we've seen Step 2 to distinguish train vs val Real Diversity
    in_step2 = False

    for line in text.splitlines():
        m = RE_EVAL_FOLD.search(line)
        if m:
            result.fold = int(m.group(1))
            result.design_name = m.group(2)
            continue

        if '[Step 2]' in line:
            in_step2 = True

        m = RE_TRAIN_RECON_MSE.search(line)
        if m:
            result.train_recon_mse = float(m.group(1))

        if not in_step2:
            m = RE_REAL_DIV_TRAIN.search(line)
            if m:
                result.train_real_diversity = float(m.group(1))

        m = RE_VAL_RECON_MSE.search(line)
        if m:
            result.val_recon_mse = float(m.group(1))

        m = RE_ACTIVE_KL.search(line)
        if m:
            result.active_dims = int(m.group(1))
            result.z_dim = int(m.group(2))

        if in_step2:
            m = RE_REAL_DIV_VAL.search(line)
            if m:
                result.real_diversity = float(m.group(1))

        m = RE_GEN_DIV.search(line)
        if m:
            result.gen_diversity = float(m.group(1))
            result.diversity_ratio = float(m.group(2))

        m = RE_MMD.search(line)
        if m:
            result.mmd = float(m.group(1))

    return result


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

def _check_convergence(epochs: list[TrainingEpoch]) -> bool:
    """Return True if training appears converged.

    Checks whether val_loss dropped meaningfully in the final window.
    """
    if len(epochs) < CONVERGENCE_WINDOW:
        return True
    recent = epochs[-CONVERGENCE_WINDOW:]
    start_loss = recent[0].val_loss
    end_loss = recent[-1].val_loss
    if start_loss <= 0:
        return True
    relative_drop = (start_loss - end_loss) / start_loss
    return relative_drop < CONVERGENCE_DROP_FRAC


def diagnose_training_run(run: TrainingRun) -> list[Issue]:
    """Detect issues in a single training run."""
    issues: list[Issue] = []
    if not run.epochs:
        return issues

    last = run.epochs[-1]

    # Not converged
    if not _check_convergence(run.epochs):
        total = last.total_epochs
        issues.append(Issue(
            severity='WARNING',
            model_type=run.model_type,
            fold=run.fold,
            description=f'Training may not have converged '
                        f'(val loss still decreasing at epoch {last.epoch}/{total})',
            recommendation=f'Increase training_epochs (current: {total})',
        ))

    # Overfitting
    if run.model_type == 'cvae' and last.val_recon is not None and last.train_recon is not None:
        if last.train_recon > 0:
            ratio = last.val_recon / last.train_recon
            if ratio > OVERFIT_RATIO:
                issues.append(Issue(
                    severity='WARNING',
                    model_type=run.model_type,
                    fold=run.fold,
                    description=f'Overfitting: val_recon/train_recon = {ratio:.1f}x',
                    recommendation='Strengthen augmentation (aug_d4, aug_elev_noise_std), '
                                   'increase weight_decay, or reduce model capacity',
                ))
    elif last.train_loss > 0:
        ratio = last.val_loss / last.train_loss
        if ratio > OVERFIT_RATIO:
            issues.append(Issue(
                severity='WARNING',
                model_type=run.model_type,
                fold=run.fold,
                description=f'Overfitting: val_loss/train_loss = {ratio:.1f}x',
                recommendation='Strengthen augmentation or increase weight_decay',
            ))

    # Posterior collapse (CVAE only)
    if run.model_type == 'cvae' and last.val_active is not None and last.z_dim is not None:
        if last.val_active < last.z_dim * ACTIVE_DIM_FRAC:
            issues.append(Issue(
                severity='CRITICAL',
                model_type=run.model_type,
                fold=run.fold,
                description=f'Posterior collapse: only {last.val_active}/{last.z_dim} '
                            f'active dims (threshold: {last.z_dim * ACTIVE_DIM_FRAC:.0f})',
                recommendation='Increase free_bits (try 1.5-2.0) and/or '
                               'increase aux_weight (try 0.3-0.5)',
            ))

    # KL vanishing (CVAE only)
    if (run.model_type == 'cvae' and last.val_kl is not None
            and last.beta is not None and last.beta > 0 and last.val_kl < 0.01):
        issues.append(Issue(
            severity='WARNING',
            model_type=run.model_type,
            fold=run.fold,
            description=f'KL vanishing: val_kl={last.val_kl:.4f} while beta={last.beta:.3f}',
            recommendation='z1 may carry no information. '
                           'Increase free_bits or reduce beta_max',
        ))

    return issues


def detect_eval_issues(result: EvalResult) -> list[Issue]:
    """Detect issues in evaluation metrics for a single fold."""
    issues: list[Issue] = []

    # Low diversity
    if result.diversity_ratio is not None and result.diversity_ratio < DIVERSITY_RATIO_LOW:
        issues.append(Issue(
            severity='WARNING',
            model_type=result.model_type,
            fold=result.fold,
            description=f'Low diversity ratio: {result.diversity_ratio:.2f}x '
                        f'(threshold: {DIVERSITY_RATIO_LOW})',
            recommendation='Increase sampling temperature (try 1.5). '
                           'For CVAE, check posterior collapse (active dims)',
        ))

    # High diversity
    if result.diversity_ratio is not None and result.diversity_ratio > DIVERSITY_RATIO_HIGH:
        issues.append(Issue(
            severity='WARNING',
            model_type=result.model_type,
            fold=result.fold,
            description=f'High diversity ratio: {result.diversity_ratio:.2f}x '
                        f'(threshold: {DIVERSITY_RATIO_HIGH})',
            recommendation='Model produces too much variation. '
                           'Reduce beta_max or increase ema_decay closer to 1.0',
        ))

    # High MMD
    if result.mmd is not None and result.mmd > MMD_HIGH:
        issues.append(Issue(
            severity='CRITICAL',
            model_type=result.model_type,
            fold=result.fold,
            description=f'High MMD: {result.mmd:.4f} (threshold: {MMD_HIGH})',
            recommendation='Generated distribution differs significantly from real data. '
                           'Check training convergence, try more epochs or lower learning rate',
        ))
    elif result.mmd is not None and result.mmd > MMD_GOOD:
        issues.append(Issue(
            severity='WARNING',
            model_type=result.model_type,
            fold=result.fold,
            description=f'Moderate MMD: {result.mmd:.4f} (threshold: {MMD_GOOD})',
            recommendation='Consider more training epochs or tuning temperature',
        ))

    # Note: active dims / posterior collapse is already checked in
    # diagnose_training_run() from training logs, so we skip it here
    # to avoid duplicate issues.

    return issues


def detect_issues(
    train_runs: dict[str, dict[int, TrainingRun]],
    eval_results: dict[str, dict[int, EvalResult]],
) -> list[Issue]:
    """Run all diagnostic checks across all models and folds."""
    issues: list[Issue] = []

    for model_type, folds in train_runs.items():
        for fold, run in folds.items():
            issues.extend(diagnose_training_run(run))

    for model_type, folds in eval_results.items():
        for fold, result in folds.items():
            issues.extend(detect_eval_issues(result))

    # Sort: CRITICAL first, then WARNING, then by model and fold
    severity_order = {'CRITICAL': 0, 'WARNING': 1}
    issues.sort(key=lambda i: (severity_order.get(i.severity, 2),
                                i.model_type, i.fold or 0))
    return issues


# ------------------------------------------------------------------
# Report formatting
# ------------------------------------------------------------------

def format_training_report(
    writer: ReportWriter,
    train_runs: dict[str, dict[int, TrainingRun]],
) -> None:
    """Format the training diagnostics section of the report."""
    writer.write('=' * 70)
    writer.write('  TRAINING DIAGNOSTICS')
    writer.write('=' * 70)

    for model_type in MODEL_TYPES:
        folds = train_runs.get(model_type, {})
        if not folds:
            continue

        writer.write(f'\n--- {model_type.upper()} ---')

        if model_type == 'cvae':
            writer.write(
                f'{"Fold":>4}  {"Epochs":>6}  {"Best":>5}  '
                f'{"Train Rec":>10}  {"Val Rec":>10}  '
                f'{"Active":>8}  {"Status"}'
            )
            writer.write('  ' + '-' * 65)
            for fold in sorted(folds):
                run = folds[fold]
                if not run.epochs:
                    writer.write(f'{fold:4d}  {"(empty log)":>6}')
                    continue
                last = run.epochs[-1]
                active_str = (f'{last.val_active}/{last.z_dim}'
                              if last.val_active is not None else '--')
                status = _training_status(run)
                writer.write(
                    f'{fold:4d}  {last.epoch:6d}  {run.best_epoch or 0:5d}  '
                    f'{last.train_recon or 0:10.4f}  {last.val_recon or 0:10.4f}  '
                    f'{active_str:>8}  {status}'
                )
        else:
            loss_label = 'noise-pred' if model_type == 'ldm' else 'velocity'
            writer.write(
                f'{"Fold":>4}  {"Epochs":>6}  {"Best":>5}  '
                f'{"Train Loss":>11}  {"Val Loss":>11}  {"Status"}'
            )
            writer.write('  ' + '-' * 55)
            for fold in sorted(folds):
                run = folds[fold]
                if not run.epochs:
                    writer.write(f'{fold:4d}  {"(empty log)":>6}')
                    continue
                last = run.epochs[-1]
                status = _training_status(run)
                writer.write(
                    f'{fold:4d}  {last.epoch:6d}  {run.best_epoch or 0:5d}  '
                    f'{last.train_loss:11.6f}  {last.val_loss:11.6f}  {status}'
                )

    writer.write('')


def _training_status(run: TrainingRun) -> str:
    """Determine concise status label for a training run."""
    labels: list[str] = []
    if not run.completed:
        labels.append('INCOMPLETE')
    if run.early_stopped:
        labels.append(f'EARLY_STOP@{run.early_stop_epoch}')
    if run.epochs:
        last = run.epochs[-1]
        if (run.model_type == 'cvae' and last.val_active is not None
                and last.z_dim is not None
                and last.val_active < last.z_dim * ACTIVE_DIM_FRAC):
            labels.append('COLLAPSE')
        if not _check_convergence(run.epochs):
            labels.append('NOT_CONVERGED')
    return ', '.join(labels) if labels else 'OK'


def format_eval_table(
    writer: ReportWriter,
    eval_results: dict[str, dict[int, EvalResult]],
) -> None:
    """Format the evaluation results cross-comparison table."""
    writer.write('=' * 70)
    writer.write('  EVALUATION RESULTS')
    writer.write('=' * 70)

    # Collect all folds across all models
    all_folds: set[int] = set()
    for folds in eval_results.values():
        all_folds.update(folds.keys())

    if not all_folds:
        writer.write('\n  (no evaluation data found)\n')
        return

    # Determine which models have data
    active_models = [m for m in MODEL_TYPES if eval_results.get(m)]

    # Per-model tables (cleaner than a wide cross-model table)
    for model_type in active_models:
        folds = eval_results[model_type]
        writer.write(f'\n--- {model_type.upper()} ---')

        if model_type == 'cvae':
            writer.write(
                f'{"Fold":>4}  {"Design":<20}  '
                f'{"Recon MSE":>10}  {"Gen Div":>10}  '
                f'{"Ratio":>7}  {"MMD":>8}  {"Active":>8}'
            )
            writer.write('  ' + '-' * 75)
        else:
            writer.write(
                f'{"Fold":>4}  {"Design":<20}  '
                f'{"Gen Div":>10}  {"Ratio":>7}  {"MMD":>8}'
            )
            writer.write('  ' + '-' * 55)

        mmd_vals: list[float] = []
        ratio_vals: list[float] = []

        for fold in sorted(folds):
            r = folds[fold]
            design = r.design_name or f'fold_{fold}'

            # Truncate long design names
            if len(design) > 20:
                design = design[:17] + '...'

            gen_div = f'{r.gen_diversity:.6f}' if r.gen_diversity is not None else '--'
            ratio = f'{r.diversity_ratio:.2f}x' if r.diversity_ratio is not None else '--'
            mmd_str = f'{r.mmd:.4f}' if r.mmd is not None else '--'

            if r.mmd is not None:
                mmd_vals.append(r.mmd)
            if r.diversity_ratio is not None:
                ratio_vals.append(r.diversity_ratio)

            if model_type == 'cvae':
                recon = f'{r.val_recon_mse:.6f}' if r.val_recon_mse is not None else '--'
                active = (f'{r.active_dims}/{r.z_dim}'
                          if r.active_dims is not None else '--')
                writer.write(
                    f'{fold:4d}  {design:<20}  '
                    f'{recon:>10}  {gen_div:>10}  '
                    f'{ratio:>7}  {mmd_str:>8}  {active:>8}'
                )
            else:
                writer.write(
                    f'{fold:4d}  {design:<20}  '
                    f'{gen_div:>10}  {ratio:>7}  {mmd_str:>8}'
                )

        # Model summary line
        if mmd_vals:
            mean_mmd = np.mean(mmd_vals)
            mean_ratio = np.mean(ratio_vals) if ratio_vals else float('nan')
            writer.write(
                f'  {"mean":>4}  {"":20}  '
                + ('          ' if model_type == 'cvae' else '')
                + f'{"":>10}  {mean_ratio:6.2f}x  {mean_mmd:8.4f}'
            )

    # Best model comparison (by mean MMD)
    writer.write(f'\n{"Best model per fold (by MMD):"}')
    for fold in sorted(all_folds):
        best_model = None
        best_mmd = float('inf')
        for model_type in active_models:
            r = eval_results[model_type].get(fold)
            if r and r.mmd is not None and r.mmd < best_mmd:
                best_mmd = r.mmd
                best_model = model_type
        if best_model:
            design = eval_results[best_model][fold].design_name or f'fold_{fold}'
            writer.write(f'  Fold {fold} ({design}): {best_model.upper()} '
                         f'(MMD={best_mmd:.4f})')

    writer.write('')


def format_diagnosis_report(writer: ReportWriter, issues: list[Issue]) -> None:
    """Format the problem diagnosis section of the report."""
    writer.write('=' * 70)
    writer.write('  PROBLEM DIAGNOSIS')
    writer.write('=' * 70)

    if not issues:
        writer.write('\n  No issues detected. All folds appear healthy.\n')
        return

    critical = [i for i in issues if i.severity == 'CRITICAL']
    warnings = [i for i in issues if i.severity == 'WARNING']

    if critical:
        writer.write(f'\nCRITICAL ISSUES ({len(critical)}):')
        for issue in critical:
            fold_str = f'fold {issue.fold}' if issue.fold is not None else 'all folds'
            writer.write(f'  [{issue.model_type.upper()} {fold_str}] {issue.description}')
            writer.write(f'    -> {issue.recommendation}')

    if warnings:
        writer.write(f'\nWARNINGS ({len(warnings)}):')
        for issue in warnings:
            fold_str = f'fold {issue.fold}' if issue.fold is not None else 'all folds'
            writer.write(f'  [{issue.model_type.upper()} {fold_str}] {issue.description}')
            writer.write(f'    -> {issue.recommendation}')

    writer.write('')


def format_summary(
    writer: ReportWriter,
    train_runs: dict[str, dict[int, TrainingRun]],
    eval_results: dict[str, dict[int, EvalResult]],
    issues: list[Issue],
) -> None:
    """Format the summary section of the report."""
    writer.write('=' * 70)
    writer.write('  SUMMARY')
    writer.write('=' * 70)

    # Models and folds analysed
    active_models = [m for m in MODEL_TYPES
                     if train_runs.get(m) or eval_results.get(m)]
    all_folds: set[int] = set()
    for folds in {**train_runs, **eval_results}.values():
        if isinstance(folds, dict):
            all_folds.update(folds.keys())

    writer.write(f'  Models analysed : {", ".join(m.upper() for m in active_models)}')
    if all_folds:
        writer.write(f'  Folds analysed  : {min(all_folds)}-{max(all_folds)} '
                     f'({len(all_folds)} total)')
    n_critical = sum(1 for i in issues if i.severity == 'CRITICAL')
    n_warning = sum(1 for i in issues if i.severity == 'WARNING')
    writer.write(f'  Critical issues : {n_critical}')
    writer.write(f'  Warnings        : {n_warning}')

    # Best model by mean MMD
    writer.write('')
    model_mmds: dict[str, list[float]] = {}
    model_ratios: dict[str, list[float]] = {}
    for model_type in active_models:
        for fold, r in eval_results.get(model_type, {}).items():
            if r.mmd is not None:
                model_mmds.setdefault(model_type, []).append(r.mmd)
            if r.diversity_ratio is not None:
                model_ratios.setdefault(model_type, []).append(r.diversity_ratio)

    if model_mmds:
        best_model = min(model_mmds, key=lambda m: np.mean(model_mmds[m]))
        writer.write(f'  Best overall model (by mean MMD): {best_model.upper()} '
                     f'(mean MMD = {np.mean(model_mmds[best_model]):.4f})')
        for m in active_models:
            mmd_str = (f'mean MMD = {np.mean(model_mmds[m]):.4f}'
                       if m in model_mmds else 'no eval data')
            ratio_str = (f'mean div ratio = {np.mean(model_ratios[m]):.2f}x'
                         if m in model_ratios else '')
            sep = ', ' if ratio_str else ''
            writer.write(f'    {m.upper():>4}: {mmd_str}{sep}{ratio_str}')

    writer.write('')


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def create_report_figure(
    train_runs: dict[str, dict[int, TrainingRun]],
    eval_results: dict[str, dict[int, EvalResult]],
    save_path: str,
) -> None:
    """Create and save the multi-panel analysis figure.

    Panel 1 (top-left)  : Training loss curves per model
    Panel 2 (top-right) : MMD bar chart per fold × model
    Panel 3 (bottom)    : Diversity scatter (real vs gen)
    """
    active_models = [m for m in MODEL_TYPES
                     if train_runs.get(m) or eval_results.get(m)]
    if not active_models:
        return

    n_train_panels = sum(1 for m in active_models if train_runs.get(m))
    has_eval = any(eval_results.get(m) for m in active_models)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, max(n_train_panels, 2), figure=fig,
                           hspace=0.35, wspace=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # --- Panel 1: Training loss curves ---
    panel_idx = 0
    for model_type in active_models:
        folds = train_runs.get(model_type, {})
        if not folds:
            continue
        ax = fig.add_subplot(gs[0, panel_idx])
        for fold in sorted(folds):
            run = folds[fold]
            if not run.epochs:
                continue
            epochs_x = [e.epoch for e in run.epochs]
            if model_type == 'cvae':
                val_y = [e.val_recon if e.val_recon is not None else e.val_loss
                         for e in run.epochs]
            else:
                val_y = [e.val_loss for e in run.epochs]
            color = colors[fold % len(colors)]
            ax.plot(epochs_x, val_y, color=color, alpha=0.7,
                    label=f'fold {fold}', linewidth=1.0)
            if run.best_epoch is not None:
                best_idx = None
                for i, e in enumerate(run.epochs):
                    if e.epoch == run.best_epoch:
                        best_idx = i
                        break
                if best_idx is not None:
                    ax.plot(run.best_epoch, val_y[best_idx], '*',
                            color=color, markersize=8)

        loss_label = ('val_recon' if model_type == 'cvae'
                      else 'val_noise-pred' if model_type == 'ldm'
                      else 'val_velocity')
        ax.set_title(f'{model_type.upper()} -- Training Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(loss_label)
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # --- Panel 2: MMD bar chart ---
    if has_eval:
        ax_mmd = fig.add_subplot(gs[1, 0])
        eval_models = [m for m in active_models if eval_results.get(m)]
        all_folds_sorted = sorted(set().union(*(
            eval_results[m].keys() for m in eval_models)))

        if all_folds_sorted and eval_models:
            x = np.arange(len(all_folds_sorted))
            width = 0.8 / len(eval_models)
            model_colors = {'cvae': '#4C72B0', 'ldm': '#55A868', 'lfm': '#C44E52'}

            for i, model_type in enumerate(eval_models):
                mmds = []
                for fold in all_folds_sorted:
                    r = eval_results[model_type].get(fold)
                    mmds.append(r.mmd if r and r.mmd is not None else 0.0)
                offset = (i - len(eval_models) / 2 + 0.5) * width
                ax_mmd.bar(x + offset, mmds, width,
                           label=model_type.upper(),
                           color=model_colors.get(model_type, f'C{i}'),
                           alpha=0.8)

            ax_mmd.axhline(y=MMD_GOOD, color='green', linestyle='--',
                           alpha=0.5, label=f'Good ({MMD_GOOD})')
            ax_mmd.axhline(y=MMD_HIGH, color='red', linestyle='--',
                           alpha=0.5, label=f'High ({MMD_HIGH})')
            ax_mmd.set_xticks(x)
            ax_mmd.set_xticklabels([f'fold {f}' for f in all_folds_sorted],
                                   fontsize=8, rotation=45)
            ax_mmd.set_title('MMD per Fold')
            ax_mmd.set_ylabel('MMD')
            ax_mmd.legend(fontsize=7)
            ax_mmd.grid(True, alpha=0.3, axis='y')

        # --- Panel 3: Diversity scatter ---
        ax_div = fig.add_subplot(gs[1, 1])
        marker_map = {'cvae': 'o', 'ldm': 's', 'lfm': '^'}

        real_divs: list[float] = []
        gen_divs: list[float] = []

        for model_type in eval_models:
            rds, gds, fold_labels = [], [], []
            for fold in sorted(eval_results[model_type]):
                r = eval_results[model_type][fold]
                if r.real_diversity is not None and r.gen_diversity is not None:
                    rds.append(r.real_diversity)
                    gds.append(r.gen_diversity)
                    fold_labels.append(fold)
                    real_divs.append(r.real_diversity)
                    gen_divs.append(r.gen_diversity)
            if rds:
                ax_div.scatter(rds, gds,
                               marker=marker_map.get(model_type, 'o'),
                               label=model_type.upper(), s=60, alpha=0.7,
                               color=model_colors.get(model_type, None))
                for rd, gd, fl in zip(rds, gds, fold_labels):
                    ax_div.annotate(str(fl), (rd, gd), fontsize=6,
                                    textcoords='offset points',
                                    xytext=(4, 4))

        if real_divs and gen_divs:
            max_val = max(max(real_divs), max(gen_divs)) * 1.2
            min_val = 0
            diag = np.linspace(min_val, max_val, 100)
            ax_div.plot(diag, diag, 'k--', alpha=0.3, label='y = x')
            ax_div.fill_between(diag, diag * DIVERSITY_RATIO_LOW,
                                diag * DIVERSITY_RATIO_HIGH,
                                alpha=0.08, color='green',
                                label=f'{DIVERSITY_RATIO_LOW}-{DIVERSITY_RATIO_HIGH}x range')
            ax_div.set_xlim(min_val, max_val)
            ax_div.set_ylim(min_val, max_val)

        ax_div.set_title('Diversity: Real vs Generated')
        ax_div.set_xlabel('Real Diversity')
        ax_div.set_ylabel('Generated Diversity')
        ax_div.legend(fontsize=7)
        ax_div.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyse training and evaluation results for PCB Warpage models')
    parser.add_argument('--outputs-dir', type=str, default='outputs',
                        help='Root outputs directory (default: outputs/)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['cvae', 'ldm', 'lfm'],
                        help='Analyse specific model only (default: all)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualisation generation')
    parser.add_argument('--save-report', type=str, default=None,
                        help='Override report text file path '
                             '(default: {outputs-dir}/analysis_report.txt)')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Override plot file path '
                             '(default: {outputs-dir}/analysis_report.png)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    model_types = [args.model] if args.model else MODEL_TYPES

    # --- Discover and parse all log files ---
    log_files = discover_log_files(outputs_dir, model_types)

    train_runs: dict[str, dict[int, TrainingRun]] = {}
    eval_results: dict[str, dict[int, EvalResult]] = {}

    for model_type in model_types:
        train_runs[model_type] = {}
        for fold, path in log_files[model_type]['train'].items():
            if model_type == 'cvae':
                run = parse_cvae_training_log(path)
            else:
                run = parse_latent_training_log(path, model_type)
            run.fold = fold
            train_runs[model_type][fold] = run

        eval_results[model_type] = {}
        for fold, path in log_files[model_type]['eval'].items():
            result = parse_eval_log(path, model_type)
            if result.fold == -1:
                result.fold = fold
            eval_results[model_type][fold] = result

    # --- Check if any data was found ---
    total_train = sum(len(f) for f in train_runs.values())
    total_eval = sum(len(f) for f in eval_results.values())

    if total_train == 0 and total_eval == 0:
        print(f'No log files found in {outputs_dir}/logs_*/.')
        print(f'Expected structure:')
        print(f'  {outputs_dir}/logs_cvae/fold0.log       (training)')
        print(f'  {outputs_dir}/logs_cvae/eval_fold0.log  (evaluation)')
        return

    # --- Build report ---
    writer = ReportWriter()
    writer.write('')
    writer.write('=' * 70)
    writer.write('  PCB WARPAGE -- ANALYSIS REPORT')
    writer.write('=' * 70)
    writer.write(f'  Source: {outputs_dir.resolve()}')
    writer.write(f'  Training logs : {total_train}')
    writer.write(f'  Evaluation logs: {total_eval}')
    writer.write('')

    if total_train > 0:
        format_training_report(writer, train_runs)

    if total_eval > 0:
        format_eval_table(writer, eval_results)

    issues = detect_issues(train_runs, eval_results)
    format_diagnosis_report(writer, issues)
    format_summary(writer, train_runs, eval_results, issues)

    # --- Save text report ---
    report_path = args.save_report or str(outputs_dir / 'analysis_report.txt')
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(writer.get_text(), encoding='utf-8')
    print(f'Report saved to: {report_path}')

    # --- Generate plots ---
    if not args.no_plot:
        plot_path = args.save_plot or str(outputs_dir / 'analysis_report.png')
        create_report_figure(train_runs, eval_results, plot_path)
        print(f'Plot saved to: {plot_path}')


if __name__ == '__main__':
    main()
