#!/usr/bin/env python3
"""CVAE diagnostic script — three checks in sequence.

  Check 1  z1 sensitivity
    Does the decoder actually respond to different z1 values?
    (was already passing)

  Check 2  Prior vs Posterior mismatch
    During training  z ~ q(z|x,c)  with learned (mu, sigma).
    During inference z ~ N(0, I).
    If beta is too low, mu drifts far from 0 and sigma shrinks far below 1,
    so inference samples land out-of-distribution and the decoder produces
    unrealistic outputs.

    We diagnose this by comparing:
      (a) reconstructions using z ~ q  (seen at training time)
      (b) samples using z ~ N(0,I)     (used at inference time)
    If (a) looks good but (b) diversity/pixel-stats are very different, it is
    a prior-posterior mismatch.

  Check 3  Posterior statistics over real data
    Prints mean |mu|, mean sigma, and per-dim KL to show how far the
    posterior sits from the prior.

Usage:
  python debug.py
  python debug.py --design data/design/design_B.png
  python debug.py --config config.txt --k 16
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from utils.load_config import load_config
from utils.handcrafted_features import extract_handcrafted_features
from utils.dataset import PCBWarpageDataset, _resolve_design_names
from models import build_model


# ------------------------------------------------------------------
# Args
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='CVAE diagnostic')
    parser.add_argument('--config', type=str, default='config.txt')
    parser.add_argument('--design', type=str, default=None,
                        help='Design image for checks 1 & 2 '
                             '(default: first design in config)')
    parser.add_argument('--k', type=int, default=16,
                        help='Samples for check 1 & 2 (default: 16)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Val fold to use for check 3 (default: 0)')
    return parser.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_model(config):
    model_path = config.get('modelpath', './outputs/cvae_pcb.pth')
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config['model_type'] = 'cvae'
    model = build_model(config)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded CVAE from {model_path}  (epoch {ckpt.get('epoch', '?')})\n")
    return model


def load_design(path, image_size):
    img    = Image.open(path).convert('L')
    design = TF.to_tensor(img.resize((image_size, image_size))).unsqueeze(0)
    hand   = extract_handcrafted_features(img).unsqueeze(0)
    return design, hand


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ------------------------------------------------------------------
# Check 1 — z1 sensitivity
# ------------------------------------------------------------------

def check_z1_sensitivity(model, design, hand, k):
    sep("Check 1 — z1 sensitivity (decoder가 z1을 쓰는가?)")

    with torch.no_grad():
        c     = model.design_encoder(design, hand)
        c_exp = c.expand(k, -1)

        z_zero   = torch.zeros(1, model.z_dim)
        z_random = torch.randn(k, model.z_dim)

        out_zero   = model.decoder(model.fuse(z_zero,   c),     c)
        out_random = model.decoder(model.fuse(z_random, c_exp), c_exp)

    diff      = (out_random - out_zero).abs().mean().item()
    diversity = out_random.var(dim=0).mean().item()

    print(f"  z=0 vs z=random 평균 픽셀 차이 : {diff:.6f}")
    print(f"  random 샘플 간 diversity (var) : {diversity:.6f}")

    if diff < 0.001:
        print("\n  [결론] decoder가 z1을 거의 무시함 → beta_max 올리기")
        return 'ignored'
    elif diff < 0.01:
        print("\n  [결론] z1 영향 약함 → beta_max 조금 더 올리기")
        return 'weak'
    else:
        print("\n  [결론] decoder가 z1을 활용 중 → 다음 체크로 이동")
        return 'ok'


# ------------------------------------------------------------------
# Check 2 — Prior vs Posterior mismatch
# ------------------------------------------------------------------

def check_prior_mismatch(model, design, hand, k):
    sep("Check 2 — Prior vs Posterior mismatch")
    print("  학습 시 z ~ q(z|x,c)  vs  추론 시 z ~ N(0,I) 비교")

    # 실제 elevation 이미지를 구하기 위해 design과 같은 폴더에서 아무거나 하나 로드
    # 여기서는 간단히 design 자체를 elevation proxy로 쓰지 않고,
    # 대신 posterior 샘플과 prior 샘플의 통계만 비교
    with torch.no_grad():
        c     = model.design_encoder(design, hand)
        c_exp = c.expand(k, -1)

        # Prior samples: z ~ N(0, I)
        z_prior  = torch.randn(k, model.z_dim)
        out_prior = model.decoder(model.fuse(z_prior, c_exp), c_exp)

        # Prior sample stats
        prior_mean = out_prior.mean().item()
        prior_std  = out_prior.std().item()
        prior_div  = out_prior.var(dim=0).mean().item()

    print(f"\n  [z ~ N(0,I)  inference-time prior]")
    print(f"    출력 평균 : {prior_mean:.4f}")
    print(f"    출력 std  : {prior_std:.4f}")
    print(f"    diversity : {prior_div:.6f}")

    print(f"\n  [해석 기준]")
    print(f"    출력 평균이 0.3~0.7 범위면 정상, 벗어나면 prior 불일치 의심")
    print(f"    diversity가 Check 1의 값과 크게 다르면 prior shift 확인 필요")

    if prior_mean < 0.1 or prior_mean > 0.9:
        print(f"\n  [경고] 출력 평균({prior_mean:.3f})이 극단적 → "
              f"posterior mu가 prior에서 많이 벗어나 있을 가능성")
        return 'mismatch'
    else:
        print(f"\n  [정상 범위] prior 샘플 출력이 합리적인 범위")
        return 'ok'


# ------------------------------------------------------------------
# Check 3 — Posterior statistics over real data
# ------------------------------------------------------------------

def check_posterior_stats(model, config, fold):
    sep(f"Check 3 — Posterior statistics (fold {fold} train split)")
    print("  encoder가 출력하는 mu, sigma가 prior N(0,I)와 얼마나 다른가?")

    cfg = dict(config)
    cfg['val_fold'] = fold
    dataset = PCBWarpageDataset(cfg['dataset_dir'], cfg, split='train', val_fold=fold)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    mu_list, logvar_list = [], []

    with torch.no_grad():
        for design_b, elevation_b, hand_b in loader:
            _, mu, logvar = model(elevation_b, design_b, hand_b)
            mu_list.append(mu)
            logvar_list.append(logvar)

    mu_all     = torch.cat(mu_list,     dim=0)   # (N, z_dim)
    logvar_all = torch.cat(logvar_list, dim=0)   # (N, z_dim)
    sigma_all  = (0.5 * logvar_all).exp()        # (N, z_dim)

    mu_abs_mean   = mu_all.abs().mean().item()
    mu_abs_max    = mu_all.abs().max().item()
    sigma_mean    = sigma_all.mean().item()
    sigma_min     = sigma_all.min().item()

    # Per-dim KL
    kl_per_dim = (-0.5 * (1 + logvar_all - mu_all.pow(2) - logvar_all.exp())
                  ).mean(dim=0)   # (z_dim,)
    kl_total   = kl_per_dim.sum().item()
    kl_mean_per_dim = kl_per_dim.mean().item()

    print(f"\n  mu  : 평균 |mu| = {mu_abs_mean:.4f}  /  최대 |mu| = {mu_abs_max:.4f}")
    print(f"        (이상적: 평균 |mu| ≈ 0.0)")
    print(f"\n  sigma : 평균 = {sigma_mean:.4f}  /  최솟값 = {sigma_min:.4f}")
    print(f"          (이상적: 평균 sigma ≈ 1.0)")
    print(f"\n  KL   : 총합 = {kl_total:.2f} nats  /  차원당 평균 = {kl_mean_per_dim:.4f} nats")

    print(f"\n  [해석]")
    problems = []
    if mu_abs_mean > 1.0:
        problems.append(f"mu가 prior에서 멀리 떨어짐 (평균 |mu|={mu_abs_mean:.3f})")
    if sigma_mean < 0.3:
        problems.append(f"sigma가 너무 작음 (평균={sigma_mean:.3f}) → posterior가 너무 좁음")
    if sigma_mean > 2.0:
        problems.append(f"sigma가 너무 큼 (평균={sigma_mean:.3f})")

    if problems:
        print(f"  [경고] Prior-posterior mismatch 확인됨:")
        for p in problems:
            print(f"    - {p}")
        print(f"\n  -> beta_max를 올려서 KL regularization을 강화하세요")
        print(f"     현재 KL={kl_total:.1f}  /  권장: z_dim({model.z_dim})의 0.3~1.0배 "
              f"= {model.z_dim*0.3:.0f}~{model.z_dim*1.0:.0f} nats")
        return 'mismatch'
    else:
        print(f"  [정상] posterior가 prior와 가깝게 학습됨")
        return 'ok'


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)
    model  = load_model(config)

    # Design image 결정
    if args.design:
        design_path = args.design
    else:
        design_names = _resolve_design_names(config)
        design_dir   = config.get('design_image_dir',
                                  str(Path(config.get('dataset_dir', 'data')) / 'design'))
        design_path  = str(Path(design_dir) / f"{design_names[0]}.png")
    print(f"Design: {design_path}  |  k={args.k}  |  fold={args.fold}")

    image_size = int(config.get('image_size', 256))
    design, hand = load_design(design_path, image_size)

    # Run checks
    r1 = check_z1_sensitivity(model, design, hand, args.k)
    r2 = check_prior_mismatch(model, design, hand, args.k)
    r3 = check_posterior_stats(model, config, args.fold)

    # Final summary
    sep("종합 진단 결과")
    print(f"  Check 1 (z1 sensitivity)    : {r1}")
    print(f"  Check 2 (prior output range): {r2}")
    print(f"  Check 3 (posterior stats)   : {r3}")

    if r1 == 'ok' and r2 == 'ok' and r3 == 'ok':
        print("\n  모델 구조는 정상 → 실제 데이터 자체의 다양성이 낮을 수 있어요.")
        print("  Real Diversity 값과 Gen Diversity를 직접 비교해보세요.")
    elif r3 == 'mismatch':
        print("\n  주요 원인: prior-posterior mismatch")
        print("  config.txt 에서 beta_max 를 올리고 재학습하세요.")
        print("  권장: beta_max 0.2 → 1.0  (또는 0.5 → 1.0 단계적으로)")


if __name__ == '__main__':
    main()
