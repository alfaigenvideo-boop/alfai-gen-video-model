import sys, imageio, numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

def emb(img_rgb):
    faces = app.get(np.ascontiguousarray(img_rgb[:, :, ::-1]))   # BGR
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return f.normed_embedding

ref = emb(np.array(Image.open(r"C:\Users\HILALBASIBUYUK\Downloads\face.png").convert("RGB")))
for path in sys.argv[1:]:
    frames = imageio.mimread(path, memtest=False)
    for i, fr in enumerate(frames):
        e = emb(np.array(fr))
        print(path, "kare", i, "benzerlik:", None if e is None else round(float(ref @ e), 3))