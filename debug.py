# Python 인터프리터에서 실행
import torch
from utils.load_config import load_config
from models import build_model
from utils.handcrafted_features import extract_handcrafted_features
from PIL import Image
import torchvision.transforms.functional as TF

config = load_config('config.txt')
ckpt = torch.load(config['modelpath'], map_location='cpu', weights_only=False)
config['model_type'] = 'cvae'
model = build_model(config)
model.load_state_dict(ckpt['model_state'])
model.eval()

img = Image.open('data/design/design_A.png').convert('L')
design = TF.to_tensor(img.resize((256,256))).unsqueeze(0)
hand   = extract_handcrafted_features(img).unsqueeze(0)

with torch.no_grad():
    c = model.design_encoder(design, hand)
    z_zero   = torch.zeros(1, model.z_dim)
    z_random = torch.randn(10, model.z_dim)
    c_exp    = c.expand(10, -1)

    out_zero   = model.decoder(model.fuse(z_zero, c), c)
    out_random = model.decoder(model.fuse(z_random, c_exp), c_exp)

print(f"z=0 vs z=random 평균 픽셀 차이: {(out_random - out_zero).abs().mean().item():.6f}")
print(f"random 샘플 간 다양성: {out_random.var(dim=0).mean().item():.6f}")