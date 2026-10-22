import torch
from huggingface_hub import hf_hub_download
p = hf_hub_download("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sdxl.bin")
sd = torch.load(p, map_location="cpu", weights_only=True)
print(list(sd["ip_adapter"].keys())[:6])