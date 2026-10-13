"""Thí nghiệm M-LSD tiny ở bit thấp (PTQ).

1. Metric mới: tách center (classification) và displacement (regression), chỉ đo displacement
   tại các điểm center thật sự được dùng, không phân biệt thứ tự start/end.
2. Đo lại FP16 / dynamic / static bằng metric mới.
3. Sửa đầu ra: bỏ 4 kênh không dùng, chuẩn hoá displacement, tách 2 head thành 2 tensor.
"""
import os
import sys
import glob
import time
import json
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch
from ai_edge_quantizer import quantizer, recipe, algorithm_manager
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.dirname(os.path.abspath(__file__))
MLSD = f"{ROOT}/../5-mlsd/mlsd-pytorch"
IMAGENETTE = f"{ROOT}/../4-quantization/imagenette2-160"
MODELS = f"{ROOT}/models"
os.makedirs(MODELS, exist_ok=True)
sys.path.insert(0, MLSD)
from models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny

SCORE_THR, DIST_THR, TOL_PX = 0.1, 20.0, 5.0


# ============================== Dữ liệu ==============================
def preprocess(img_rgb):
    img = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    x = np.concatenate([img, np.ones([512, 512, 1])], axis=-1).transpose(2, 0, 1)[None]
    return (x.astype(np.float32) / 127.5) - 1.0

def read_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

train_paths = sorted(glob.glob(f"{IMAGENETTE}/train/*/*.JPEG"))
val_paths = sorted(glob.glob(f"{IMAGENETTE}/val/*/*.JPEG"))
calib_paths = np.random.default_rng(0).choice(train_paths, 100, replace=False)           # giống m2
eval_paths = [f"{MLSD}/data/frame_1.jpg"] + list(np.random.default_rng(1).choice(val_paths, 20, replace=False))
calib_x = np.concatenate([preprocess(read_rgb(p)) for p in calib_paths])
eval_x = [preprocess(read_rgb(p)) for p in eval_paths]


# ============================== Các biến thể model (PyTorch) ==============================
def load_base():
    m = MobileV2_MLSD_Tiny().eval()
    m.load_state_dict(torch.load(f"{MLSD}/models/mlsd_tiny_512_fp32.pth", map_location="cpu"), strict=True)
    return m

def trunk(m, x):
    # chạy model tới trước conv cuối cùng (block16.conv3)
    c2, c3, c4 = m.backbone(x)
    x = m.block12(c3, c4)
    x = m.block13(x)
    x = m.block14(c2, x)
    x = m.block15(x)
    return m.block16.conv2(m.block16.conv1(x))

def up2(y):
    return F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=True)

def head_conv(m, rows, scale=None):
    # tạo conv 1x1 mới bằng cách cắt các hàng tương ứng của block16.conv3 (16 kênh ra)
    src = m.block16.conv3
    conv = nn.Conv2d(src.in_channels, len(rows), 1)
    w = src.weight.data[rows].clone()
    b = src.bias.data[rows].clone()
    if scale is not None:                      # chia từng kênh cho scale (gộp phép chia vào weight)
        s = torch.tensor(scale, dtype=w.dtype)
        w = w / s.view(-1, 1, 1, 1)
        b = b / s
    conv.weight.data, conv.bias.data = w, b
    return conv

class Heads5(nn.Module):
    """1 tensor 5 kênh: center (kênh 7) + displacement (kênh 8-11). Bỏ 4 kênh không dùng (12-15).
    k > 1: chia displacement cho k để khoảng giá trị gần với center."""
    def __init__(self, m, k=1.0):
        super().__init__()
        self.m, self.k = m, k
        self.head = head_conv(m, [7, 8, 9, 10, 11], [1.0, k, k, k, k])

    def forward(self, x):
        return up2(self.head(trunk(self.m, x)))

class Split(nn.Module):
    """2 tensor riêng: center (1 kênh) và displacement (4 kênh), mỗi tensor có scale riêng khi quantize."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.center = head_conv(m, [7])
        self.disp = head_conv(m, [8, 9, 10, 11])

    def forward(self, x):
        t = trunk(self.m, x)
        return up2(self.center(t)), up2(self.disp(t))


# ============================== Chuẩn hoá output về (center 256x256, disp 256x256x4) ==============================
def from_base(outs):
    o = outs[0][0]
    return o[0], o[1:5].transpose(1, 2, 0)

def from_heads5(k):
    def f(outs):
        o = outs[0][0]
        return o[0], o[1:5].transpose(1, 2, 0) * k
    return f

def from_split(outs):
    c = next(o for o in outs if o.shape[1] == 1)[0, 0]
    d = next(o for o in outs if o.shape[1] == 4)[0]
    return c, d.transpose(1, 2, 0)


# ============================== Chạy TFLite (xử lý cả I/O INT8/INT16) ==============================
class TFLiteRunner:
    def __init__(self, path, threads=4):
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()

    def __call__(self, x):
        d = self.inp
        if np.issubdtype(d["dtype"], np.integer):
            s, zp = d["quantization"]
            info = np.iinfo(d["dtype"])
            x = np.clip(np.round(x / s + zp), info.min, info.max).astype(d["dtype"])
        self.it.set_tensor(d["index"], x)
        self.it.invoke()
        res = []
        for o in self.outs:
            y = self.it.get_tensor(o["index"])
            if np.issubdtype(o["dtype"], np.integer):
                s, zp = o["quantization"]
                y = (y.astype(np.float32) - zp) * s
            res.append(y.astype(np.float32))
        return res

def benchmark(path, threads=4, warmup=10, runs=30):
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
    return float(np.median(t))


# ============================== Convert và quantize ==============================
def convert(module, path):
    if not os.path.exists(path):
        litert_torch.convert(module.eval(), (torch.from_numpy(eval_x[0]),)).export(path)
    return path

def global_min_max_update(qsv, new_qsv):
    if not qsv:
        return new_qsv
    return {"min": np.minimum(qsv["min"], new_qsv["min"]), "max": np.maximum(qsv["max"], new_qsv["max"])}

FP16_RECIPE = [
    {"regex": ".*", "operation": op, "algorithm_key": "float_casting",
     "op_config": {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT", "granularity": "TENSORWISE"},
                   "compute_precision": "FLOAT", "explicit_dequantize": True}}
    for op in ["CONV_2D", "DEPTHWISE_CONV_2D"]
]

def quantize(fp32_path, kind, out_path, calib="ema"):
    if os.path.exists(out_path):
        return out_path
    rcp = {"fp16": FP16_RECIPE, "dynamic": recipe.dynamic_wi8_afp32(),
           "weight_only": recipe.weight_only_wi8_afp32(), "static": recipe.static_wi8_ai8()}[kind]
    qt = quantizer.Quantizer(fp32_path, rcp)
    if kind == "static":
        original = algorithm_manager.get_update_qsv_func
        if calib == "global":
            algorithm_manager.get_update_qsv_func = lambda *a: global_min_max_update
        sig = list(Interpreter(fp32_path).get_signature_list().items())[0]
        name, inp = sig[0], sig[1]["inputs"][0]
        result = qt.calibrate({name: [{inp: calib_x[i:i + 1]} for i in range(len(calib_x))]})
        algorithm_manager.get_update_qsv_func = original
        qt.quantize(result).export_model(out_path, overwrite=True)
    else:
        qt.quantize().export_model(out_path, overwrite=True)
    return out_path


# ============================== Metric ==============================
def decode(center, topk=200, ksize=3):
    heat = torch.sigmoid(torch.from_numpy(center))[None, None]
    hmax = F.max_pool2d(heat, ksize, stride=1, padding=ksize // 2)
    heat = (heat * (hmax == heat).float()).reshape(-1)
    scores, idx = torch.topk(heat, topk)
    w = center.shape[1]
    return np.stack([(idx // w).numpy(), (idx % w).numpy()], 1), scores.numpy()

def kept_centers(center, disp):
    # các center thật sự được dùng để tạo đoạn thẳng (giống pred_lines)
    pts, scores = decode(center)
    keep = []
    for (y, x), s in zip(pts, scores):
        d = disp[y, x]
        if s > SCORE_THR and np.hypot(d[0] - d[2], d[1] - d[3]) > DIST_THR:
            keep.append((y, x))
    return keep

def lines_of(center, disp):
    out = []
    for y, x in kept_centers(center, disp):
        d = disp[y, x]
        out.append([x + d[0], y + d[1], x + d[2], y + d[3]])
    return np.array(out, dtype=np.float64).reshape(-1, 4) * 2          # toạ độ ảnh 512

def match_ratio(ref, got, tol=TOL_PX):
    if len(ref) == 0:
        return 1.0
    if len(got) == 0:
        return 0.0
    hit = 0
    for r in ref:
        d1 = np.maximum(np.hypot(*(got[:, :2] - r[:2]).T), np.hypot(*(got[:, 2:] - r[2:]).T))
        d2 = np.maximum(np.hypot(*(got[:, :2] - r[2:]).T), np.hypot(*(got[:, 2:] - r[:2]).T))
        hit += np.minimum(d1, d2).min() < tol
    return hit / len(ref)

def snr(ref, out):
    return 10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / max(np.sum((ref.astype(np.float64) - out) ** 2), 1e-30))

def evaluate(get_maps):
    """get_maps(i) -> (center, disp) của biến thể trên ảnh i. So với REF (PyTorch FP32)."""
    c_full, d_full, c_hit, n_ref_c, errs, swaps, kept, n_lines = [], [], 0, 0, [], 0, [], 0
    for i in range(len(eval_x)):
        rc, rd = REF[i]
        c, d = get_maps(i)
        c_full.append((rc, c)); d_full.append((rd, d))
        ref_keep = kept_centers(rc, rd)
        got_keep = kept_centers(c, d)
        got_set = set(got_keep)
        for y, x in ref_keep:                                         # center: có tìm lại được điểm (±1 px)?
            n_ref_c += 1
            if any((y + dy, x + dx) in got_set for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                c_hit += 1
            a, b = rd[y, x], d[y, x]                                  # displacement tại đúng điểm đó
            same = max(np.hypot(a[0] - b[0], a[1] - b[1]), np.hypot(a[2] - b[2], a[3] - b[3]))
            swap = max(np.hypot(a[0] - b[2], a[1] - b[3]), np.hypot(a[2] - b[0], a[3] - b[1]))
            errs.append(2 * min(same, swap))                         # nhân 2: đổi sang pixel ảnh 512
            swaps += swap < same
        rl, gl = lines_of(rc, rd), lines_of(c, d)
        kept.append(match_ratio(rl, gl))
        n_lines += len(gl)
    errs = np.array(errs) if errs else np.zeros(1)
    return {
        "snr_center_full": snr(np.stack([a for a, _ in c_full]), np.stack([b for _, b in c_full])),
        "snr_disp_full": snr(np.stack([a for a, _ in d_full]), np.stack([b for _, b in d_full])),
        "center_recall": 100 * c_hit / max(n_ref_c, 1),
        "disp_err_median_px": float(np.median(errs)),
        "disp_err_p90_px": float(np.percentile(errs, 90)),
        "disp_within_2px": 100 * float(np.mean(errs < 2)),
        "start_end_swapped": 100 * swaps / max(n_ref_c, 1),
        "lines_kept": 100 * float(np.mean(kept)),
        "n_lines": n_lines,
    }


# ============================== Chạy ==============================
base = load_base()
with torch.no_grad():
    REF = [from_base([base(torch.from_numpy(x)).numpy()]) for x in eval_x]
n_ref_lines = sum(len(lines_of(c, d)) for c, d in REF)
print(f"Anh danh gia: {len(eval_x)} | so doan cua FP32: {n_ref_lines}")

# Kiểm tra các model chỉnh sửa có cho ra đúng kết quả như model gốc (FP32)
with torch.no_grad():
    t = torch.from_numpy(eval_x[0])
    full = base(t)
    h5 = Heads5(base, 8.0)(t)
    sc, sd = Split(base)(t)
print("Kiem tra tuong duong FP32: heads5(k=8) lech max", float((h5[:, 0] - full[:, 0]).abs().max()),
      float((h5[:, 1:] * 8 - full[:, 1:5]).abs().max()), "| split lech max",
      float((sc - full[:, :1]).abs().max()), float((sd - full[:, 1:5]).abs().max()))

rows = []

def run_torch(name, model, dtype=torch.float32):
    with torch.no_grad():
        maps = [from_base([model(torch.from_numpy(x).to(dtype)).float().numpy()]) for x in eval_x]
    rows.append({"name": name, "size_mb": None, "ms": None, **evaluate(lambda i: maps[i])})

def run_tflite(name, path, adapter):
    r = TFLiteRunner(path)
    maps = [adapter(r(x)) for x in eval_x]
    rows.append({"name": name, "size_mb": os.path.getsize(path) / 1e6, "ms": benchmark(path),
                 **evaluate(lambda i: maps[i])})

# --- FP16 trong PyTorch: chỉ làm tròn weight, và tính toàn bộ bằng FP16
fp16w = copy.deepcopy(base)
for p in fp16w.parameters():
    p.data = p.data.half().float()
run_torch("Torch FP16 weight", fp16w)
run_torch("Torch FP16 toan bo", copy.deepcopy(base).half(), torch.float16)

# --- Model gốc (1 tensor 9 kênh)
p_base = convert(base, f"{MODELS}/base_fp32.tflite")
run_tflite("Goc FP32", p_base, from_base)
run_tflite("Goc FP16 weight", quantize(p_base, "fp16", f"{MODELS}/base_fp16.tflite"), from_base)
run_tflite("Goc INT8 weight-only", quantize(p_base, "weight_only", f"{MODELS}/base_wo8.tflite"), from_base)
run_tflite("Goc INT8 dynamic", quantize(p_base, "dynamic", f"{MODELS}/base_dyn.tflite"), from_base)
run_tflite("Goc INT8 static EMA", quantize(p_base, "static", f"{MODELS}/base_static_ema.tflite"), from_base)
run_tflite("Goc INT8 static global", quantize(p_base, "static", f"{MODELS}/base_static_glob.tflite", "global"), from_base)

# --- Bỏ 4 kênh không dùng (1 tensor 5 kênh)
p_h5 = convert(Heads5(base, 1.0), f"{MODELS}/h5_fp32.tflite")
run_tflite("5 kenh INT8 static EMA", quantize(p_h5, "static", f"{MODELS}/h5_static_ema.tflite"), from_heads5(1.0))
run_tflite("5 kenh INT8 static global", quantize(p_h5, "static", f"{MODELS}/h5_static_glob.tflite", "global"), from_heads5(1.0))

# --- 5 kênh + chuẩn hoá displacement (chia 8)
p_h5k = convert(Heads5(base, 8.0), f"{MODELS}/h5k8_fp32.tflite")
run_tflite("5 kenh /8 INT8 static EMA", quantize(p_h5k, "static", f"{MODELS}/h5k8_static_ema.tflite"), from_heads5(8.0))
run_tflite("5 kenh /8 INT8 static global", quantize(p_h5k, "static", f"{MODELS}/h5k8_static_glob.tflite", "global"), from_heads5(8.0))

# --- Tách 2 head thành 2 tensor
p_split = convert(Split(base), f"{MODELS}/split_fp32.tflite")
run_tflite("Tach head FP32", p_split, from_split)
run_tflite("Tach head INT8 dynamic", quantize(p_split, "dynamic", f"{MODELS}/split_dyn.tflite"), from_split)
run_tflite("Tach head INT8 static EMA", quantize(p_split, "static", f"{MODELS}/split_static_ema.tflite"), from_split)
run_tflite("Tach head INT8 static global", quantize(p_split, "static", f"{MODELS}/split_static_glob.tflite", "global"), from_split)

# ============================== In bảng ==============================
os.makedirs(f"{ROOT}/results", exist_ok=True)
with open(f"{ROOT}/results/exp1.json", "w") as f:
    json.dump(rows, f, indent=1)
hdr = f"{'Bien the':30s} {'MB':>5s} {'ms':>6s} | {'SNRc':>5s} {'SNRd':>5s} | {'Center':>6s} {'Dmed':>5s} {'Dp90':>6s} {'D<2px':>6s} {'Swap':>5s} | {'Giu':>5s} {'Doan':>5s}"
print("\n" + hdr)
print("-" * len(hdr))
for r in rows:
    mb = f"{r['size_mb']:.2f}" if r["size_mb"] else "-"
    ms = f"{r['ms']:.1f}" if r["ms"] else "-"
    print(f"{r['name']:30s} {mb:>5s} {ms:>6s} | {r['snr_center_full']:5.1f} {r['snr_disp_full']:5.1f} | "
          f"{r['center_recall']:6.1f} {r['disp_err_median_px']:5.2f} {r['disp_err_p90_px']:6.2f} "
          f"{r['disp_within_2px']:6.1f} {r['start_end_swapped']:5.1f} | {r['lines_kept']:5.1f} {r['n_lines']:5d}")
