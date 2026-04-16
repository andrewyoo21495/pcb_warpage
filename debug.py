#!/usr/bin/env python3
"""Diagnostic script: check whether the CVAE decoder actually uses z1.

If the decoder ignores z1 (decoder-side posterior collapse), outputs will
be nearly identical regardless of which z1 is sampled from N(0, I).

Usage:
  python debug.py
  python debug.py --design data/design/design_B.png
  python debug.py --config config.txt --design data/design/design_A.png
"""

import argparse

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from utils.load_config import load_config
from utils.handcrafted_features import extract_handcrafted_features
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Debug CVAE decoder z1 sensitivity')
    parser.add_argument('--config', type=str, default='config.txt')
    parser.add_argument('--design', type=str, default='data/design/design_A.png',
                        help='Path to design image (default: design_A.png)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of random z1 samples to draw (default: 10)')
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    model_path = config.get('modelpath', './outputs/cvae_pcb.pth')
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config['model_type'] = 'cvae'
    model = build_model(config)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded CVAE from {model_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"Design: {args.design}  |  k={args.k}\n")

    img    = Image.open(args.design).convert('L')
    design = TF.to_tensor(img.resize((256, 256))).unsqueeze(0)
    hand   = extract_handcrafted_features(img).unsqueeze(0)

    with torch.no_grad():
        c       = model.design_encoder(design, hand)          # (1, c_dim)
        c_exp   = c.expand(args.k, -1)                        # (k, c_dim)

        z_zero   = torch.zeros(1, model.z_dim)
        z_random = torch.randn(args.k, model.z_dim)

        out_zero   = model.decoder(model.fuse(z_zero,   c),     c)      # (1, 1, H, W)
        out_random = model.decoder(model.fuse(z_random, c_exp), c_exp)  # (k, 1, H, W)

    diff      = (out_random - out_zero).abs().mean().item()
    diversity = out_random.var(dim=0).mean().item()

    print(f"z=0 vs z=random 평균 픽셀 차이 : {diff:.6f}")
    print(f"random 샘플 간 diversity (var) : {diversity:.6f}")
    print()

    if diff < 0.001:
        print("[결론] decoder가 z1을 거의 무시하고 있어요.")
        print("  -> beta_max 올리기 (0.2 -> 1.0), fusion_method=concat 시도")
    elif diff < 0.01:
        print("[결론] z1의 영향이 약해요. 개선 여지 있음.")
        print("  -> beta_max를 조금 더 올려보세요.")
    else:
        print("[결론] decoder가 z1을 잘 활용하고 있어요.")


if __name__ == '__main__':
    main()