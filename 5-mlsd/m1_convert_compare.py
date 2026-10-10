import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import litert_torch
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, "mlsd-pytorch")
from models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny

OFFICIAL = "mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite"
OFFICIAL_FP16 = "mlsd-official/tflite_models/M-LSD_512_tiny_fp16.tflite"

# ===== Model PyTorch gốc =====
model = MobileV2_MLSD_Tiny().eval()
model.load_state_dict(torch.load("mlsd-pytorch/models/mlsd_tiny_512_fp32.pth", map_location="cpu"), strict=True)

# ===== Bản có gói thêm decoder, để output giống file chính thức =====
class WithDecoder(torch.nn.Module):
    def __init__(self, net, topk=200, ksize=3):
        super().__init__()
        self.net, self.topk, self.ksize = net, topk, ksize

    def forward(self, x):
        out = self.net(x)                                   # (1, 9, 256, 256)
        center = torch.sigmoid(out[:, 0:1])
        hmax = F.max_pool2d(center, self.ksize, stride=1, padding=self.ksize // 2)
        center = center * (hmax == center).float()
        scores, idx = torch.topk(center.reshape(1, -1), self.topk)
        w = out.shape[3]
        pts = torch.stack([idx // w, idx % w], dim=-1)      # (1, 200, 2): toạ độ (y, x)
        disp = out[:, 1:5].permute(0, 2, 3, 1)              # (1, 256, 256, 4)
        return pts, scores, disp

# ===== Chuẩn bị input từ cùng 1 ảnh =====
img = cv2.cvtColor(cv2.imread("mlsd-pytorch/data/frame_1.jpg"), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
raw = np.concatenate([img, np.ones([512, 512, 1])], axis=-1)[None].astype(np.float32)  # (1,512,512,4), 0..255
x_official = raw                                                     # bản chính thức: pixel thô, NHWC
x_ours = (raw / 127.5 - 1.0).transpose(0, 3, 1, 2).copy()            # bản PyTorch: chuẩn hoá, NCHW

# ===== Convert =====
sample = torch.from_numpy(x_ours)
litert_torch.convert(model, (sample,)).export("mlsd_tiny_raw.tflite")
litert_torch.convert(WithDecoder(model), (sample,)).export("mlsd_tiny_decoder.tflite")
print("Da convert xong 2 file")

# ===== Hàm chạy TFLite =====
def run_tflite(path, x, threads=4):
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    return [it.get_tensor(d["index"]) for d in it.get_output_details()]

def snr(ref, out):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - out) ** 2))

# ===== So sánh 1: PyTorch và TFLite của mình (chỉ phần model) =====
with torch.no_grad():
    ref = model(sample).numpy()
ours_raw = run_tflite("mlsd_tiny_raw.tflite", x_ours)[0]
print(f"\n[1] PyTorch vs TFLite cua minh: SNR = {snr(ref, ours_raw):.1f} dB, max err = {np.abs(ref - ours_raw).max():.2e}")

# ===== So sánh 2: bản chính thức và bản của mình (có decoder) =====
off_pts, off_scores, off_vmap = run_tflite(OFFICIAL, x_official)
our_pts, our_scores, our_vmap = run_tflite("mlsd_tiny_decoder.tflite", x_ours)
same_pts = len({tuple(p) for p in off_pts[0]} & {tuple(p) for p in our_pts[0].astype(np.int32)})
print(f"[2] Chinh thuc vs cua minh: SNR vmap = {snr(off_vmap, our_vmap):.1f} dB | "
      f"SNR scores = {snr(off_scores, our_scores):.1f} dB | trung {same_pts}/200 diem")

# ===== Tốc độ =====
def benchmark(path, threads, warmup=10, runs=100):
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    it.set_tensor(d["index"], np.zeros(d["shape"], dtype=d["dtype"]))
    for _ in range(warmup):
        it.invoke()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        it.invoke()
        t.append((time.perf_counter() - t0) * 1000)
    return np.median(t)

print(f"\n{'Model':34s} {'1 thread(ms)':>13s} {'4 thread(ms)':>13s}")
for name, path in [("Cua minh - chi model", "mlsd_tiny_raw.tflite"),
                   ("Cua minh - co decoder", "mlsd_tiny_decoder.tflite"),
                   ("Chinh thuc FP32", OFFICIAL),
                   ("Chinh thuc FP16", OFFICIAL_FP16)]:
    print(f"{name:34s} {benchmark(path, 1):13.2f} {benchmark(path, 4):13.2f}")
