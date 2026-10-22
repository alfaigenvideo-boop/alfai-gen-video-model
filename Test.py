import torch
from modules.flow.latent_warp import LatentWarper

w = LatentWarper(device="cpu")
x = torch.randn(1, 4, 72, 128)
y = x
for _ in range(23):
    y, m = w.warp_and_create_mask(y, torch.zeros(1, 2, 72, 128))
print("ortalama fark:", (x - y).abs().mean().item())
print("maske toplamı:", m.sum().item())