"""QAT M-LSD tiny, bản 2: so PTQ và QAT trên CÙNG một cấu hình triển khai.

Cấu hình triển khai (giống PTQ tốt nhất ngày 13/10):
  - model tách head center / displacement, BatchNorm đã gộp vào conv
  - ai-edge-quantizer static INT8 (weight per-channel đối xứng, activation per-tensor bất đối xứng, calibration EMA)
  - giữ float 3 layer nhạy nhất (conv đầu, depthwise + pointwise của InvertedResidual_1)

QAT: fake quantize tự viết (chạy được trên CUDA / MPS / CPU), mô phỏng đúng cấu hình trên,
train bằng distillation (model FP32 làm thầy). Train xong gỡ fake quantize, quantize lại bằng
ai-edge-quantizer giống hệt bản PTQ để so công bằng.

Chạy: python qat_mlsd_v2.py --steps 1000 --bs 8
"""
import os
import re
import copy
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--freeze_ranges_at", type=float, default=0.2, help="tỉ lệ số bước thì chốt khoảng activation")
ap.add_argument("--also_full_int8", action="store_true", help="train thêm 1 bản QAT quantize tất cả layer")
ap.add_argument("--device", default="auto")
args = ap.parse_args()

if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dev = torch.device(args.device)
if dev.type == "cuda":
    # Colab: cuDNN đi kèm torch có thể khác phiên bản cuDNN có sẵn của máy -> lỗi khi train conv.
    # Thử 1 bước conv có backward, nếu lỗi thì tắt cuDNN (dùng kernel CUDA thường, chậm hơn nhưng chạy được).
    try:
        _c = torch.nn.Conv2d(4, 8, 3).to(dev)
        _c(torch.randn(2, 4, 16, 16, device=dev)).sum().backward()
        torch.cuda.synchronize()
    except RuntimeError as e:
        print("cuDNN loi, tat cuDNN:", str(e)[:80], flush=True)
        torch.backends.cudnn.enabled = False
print("Device:", dev, "| cuDNN:", torch.backends.cudnn.enabled if dev.type == "cuda" else "-", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
EXP1 = os.path.abspath(f"{HERE}/../../7-mlsd-lowbit/exp1_metrics_and_heads.py")
g = {"__file__": EXP1, "__name__": "exp1"}
src = open(EXP1).read()
exec(src[:src.index("rows = []")], g)                 # dữ liệu, model, metric, REF (output PyTorch FP32)
preprocess, read_rgb, load_base, Split = g["preprocess"], g["read_rgb"], g["load_base"], g["Split"]
from_split, TFLiteRunner, benchmark, evaluate = g["from_split"], g["TFLiteRunner"], g["benchmark"], g["evaluate"]
calib_x, eval_x, train_paths, calib_paths = g["calib_x"], g["eval_x"], g["train_paths"], g["calib_paths"]

import litert_torch
from ai_edge_quantizer import quantizer, recipe, algorithm_manager
from ai_edge_litert.interpreter import Interpreter

OUT = f"{HERE}/models"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{HERE}/results", exist_ok=True)
KEEP_FLOAT = {"m.backbone.features.0.0", "m.backbone.features.1.conv.0.0", "m.backbone.features.1.conv.1"}
KEEP_TFLITE = ["MobileNetV2_backbone/ConvBNReLU_0/conv.Conv2d_0",
               "InvertedResidual_1/container.Sequential_conv/ConvBNReLU_0/conv.Conv2d_0",
               "InvertedResidual_1/container.Sequential_conv/conv.Conv2d_1"]


# ============================== Model đã gộp BatchNorm ==============================
def fold_bn(model):
    for mod in model.modules():
        if isinstance(mod, nn.Sequential):
            keys = list(mod._modules.keys())
            for a, b in zip(keys, keys[1:]):
                ca, cb = mod._modules[a], mod._modules[b]
                if isinstance(ca, nn.Conv2d) and isinstance(cb, nn.BatchNorm2d):
                    mod._modules[a] = fuse_conv_bn_eval(ca.eval(), cb.eval())
                    mod._modules[b] = nn.Identity()
    return model

def folded_split():
    return Split(fold_bn(load_base().eval())).eval()


# ============================== Fake quantize tự viết ==============================
def fake_quant(x, scale, zp, qmin, qmax):
    q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
    return x + ((q - zp) * scale - x).detach()                          # STE

class ActFQ(nn.Module):
    """Activation INT8 per-tensor bất đối xứng; khoảng min/max cập nhật bằng EMA 0.95 (giống ai-edge-quantizer)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))
        self.observe, self.started = True, False

    def forward(self, x):
        if self.observe:
            mn, mx = x.detach().min(), x.detach().max()
            if not self.started:
                self.lo.copy_(mn); self.hi.copy_(mx); self.started = True
            else:
                self.lo.mul_(0.95).add_(0.05 * mn); self.hi.mul_(0.95).add_(0.05 * mx)
        lo = torch.minimum(self.lo, torch.zeros_like(self.lo))
        hi = torch.maximum(self.hi, torch.zeros_like(self.hi))
        s = torch.clamp((hi - lo) / 255.0, min=1e-8)
        zp = torch.round(-128 - lo / s)
        return fake_quant(x, s, zp, -128, 127)

class QConv(nn.Module):
    """Conv có fake quantize ở input (activation) và weight (INT8 per-channel đối xứng)."""
    def __init__(self, conv, quant=True):
        super().__init__()
        self.conv, self.quant, self.act = conv, quant, ActFQ()

    def forward(self, x):
        if not self.quant:
            return self.conv(x)
        x = self.act(x)
        w = self.conv.weight
        s = torch.clamp(w.detach().abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0, min=1e-8)
        w = fake_quant(w, s, 0, -127, 127)
        c = self.conv
        return F.conv2d(x, w, c.bias, c.stride, c.padding, c.dilation, c.groups)

class QATModel(nn.Module):
    def __init__(self, keep_float):
        super().__init__()
        self.net = folded_split()
        for name, mod in list(self.net.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Conv2d):
                    full = f"{name}.{cname}" if name else cname
                    setattr(mod, cname, QConv(child, quant=full not in keep_float))
        self.out_c, self.out_d = ActFQ(), ActFQ()                       # output 2 head cũng là INT8

    def forward(self, x):
        c, d = self.net(x)
        return self.out_c(c), self.out_d(d)

    def set_observe(self, on):
        for m in self.modules():
            if isinstance(m, ActFQ):
                m.observe = on

    def export_float(self):
        """Gỡ fake quantize, trả về model float (weight đã train) để convert."""
        net = copy.deepcopy(self.net).cpu().eval()
        for name, mod in list(net.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, QConv):
                    setattr(mod, cname, child.conv)
        return net


# ============================== Convert + quantize bằng ai-edge-quantizer ==============================
NO_Q = algorithm_manager.AlgorithmName.NO_QUANTIZE

def to_tflite(model, path):
    litert_torch.convert(model.cpu().eval(), (torch.from_numpy(eval_x[0]),)).export(path)
    return path

def keep_rules(fp32_path):
    it = Interpreter(fp32_path)
    it.allocate_tensors()
    td = it.get_tensor_details()
    rules = []
    for o in it._get_ops_details():
        if o["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            full = td[o["outputs"][0]]["name"]
            short = full.split(";;")[-1].replace("models.mbv2_mlsd_tiny.", "").replace("torch.nn.modules.", "")
            short = re.sub(r";\d*$", "", short)
            if any(short.endswith(k) for k in KEEP_TFLITE):
                rules.append({"regex": "^" + re.escape(full) + ";$", "operation": "*", "algorithm_key": NO_Q, "op_config": {}})
    assert len(rules) == 3, f"tim duoc {len(rules)} layer can giu float, can 3"
    return rules

def quantize(fp32_path, out_path, keep3):
    rcp = recipe.static_wi8_ai8() + (keep_rules(fp32_path) if keep3 else [])
    qt = quantizer.Quantizer(fp32_path, rcp)
    sig_name, sig = list(Interpreter(fp32_path).get_signature_list().items())[0]
    cal = qt.calibrate({sig_name: [{sig["inputs"][0]: calib_x[i:i + 1]} for i in range(len(calib_x))]})
    qt.quantize(cal).export_model(out_path, overwrite=True)
    return out_path

rows = []
def score(path, name, extra=None):
    r = TFLiteRunner(path)
    maps = [from_split(r(x)) for x in eval_x]
    m = evaluate(lambda i: maps[i])
    m.update({"name": name, "size_mb": os.path.getsize(path) / 1e6, "ms": benchmark(path), **(extra or {})})
    rows.append(m)
    print(f"{name:28s} center {m['center_recall']:5.1f}  Dmed {m['disp_err_median_px']:.2f}  D<2px {m['disp_within_2px']:5.1f}  "
          f"giu {m['lines_kept']:5.1f}  doan {m['n_lines']:5d} | {m['size_mb']:.2f} MB {m['ms']:.1f} ms", flush=True)


# ============================== Dữ liệu train ==============================
calib_set = set(calib_paths)
pool = [p for p in train_paths if p not in calib_set]
rng = np.random.default_rng(2)
train_list = list(rng.choice(pool, 3000, replace=False))

def batch_from(paths):
    xs = []
    for p in paths:
        img = read_rgb(p)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        xs.append(preprocess(np.ascontiguousarray(img)))
    return torch.from_numpy(np.concatenate(xs))

def loss_fn(sc, sd, tc, td):
    t = torch.sigmoid(tc)
    w = 1.0 + 20.0 * t
    l_center = (F.binary_cross_entropy_with_logits(sc, t, reduction="none") * w).mean()
    wd = t / (t.sum() + 1e-6)
    l_disp = (wd * (sd - td).abs().sum(1, keepdim=True)).sum()
    return l_center, l_disp


# ============================== QAT ==============================
teacher = load_base().eval().to(dev)

def run_qat(keep_float, tag):
    model = QATModel(keep_float).to(dev).eval()
    with torch.no_grad():                                               # calibration: EMA từng ảnh, giống ai-edge-quantizer
        for i in range(len(calib_x)):
            model(torch.from_numpy(calib_x[i:i + 1]).to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    freeze = int(args.steps * args.freeze_ranges_at)
    t0, hist = time.time(), []
    for step in range(args.steps):
        if step == freeze:
            model.set_observe(False)                                    # chốt khoảng activation, chỉ train weight
        x = batch_from([train_list[(step * args.bs + k) % len(train_list)] for k in range(args.bs)]).to(dev)
        with torch.no_grad():
            out = teacher(x)
        sc, sd = model(x)
        l_c, l_d = loss_fn(sc, sd, out[:, 0:1], out[:, 1:5])
        loss = l_c + 0.05 * l_d
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append([float(l_c), float(l_d)])
        if step % 50 == 0 or step == args.steps - 1:
            print(f"[{tag}] step {step:4d}  center {np.mean([h[0] for h in hist[-50:]]):.4f}  "
                  f"disp {np.mean([h[1] for h in hist[-50:]]):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return model, {"seconds": time.time() - t0, "loss_history": hist}

# 1. FP32 (đã gộp BN) và PTQ trên cùng cấu hình
p32 = to_tflite(folded_split(), f"{OUT}/v2_fp32.tflite")
score(p32, "FP32")
score(quantize(p32, f"{OUT}/v2_ptq_int8.tflite", keep3=False), "PTQ INT8 tat ca")
score(quantize(p32, f"{OUT}/v2_ptq_keep3.tflite", keep3=True), "PTQ INT8 giu 3 float")

# 2. QAT giữ 3 layer float -> quantize lại y hệt bản PTQ
qat, info = run_qat(KEEP_FLOAT, "keep3")
pq32 = to_tflite(qat.export_float(), f"{OUT}/v2_qat_keep3_fp32.tflite")
score(quantize(pq32, f"{OUT}/v2_qat_keep3.tflite", keep3=True), "QAT INT8 giu 3 float", {"train": info})

# 3. (tuỳ chọn) QAT quantize tất cả layer
if args.also_full_int8:
    qat_f, info_f = run_qat(set(), "full")
    pf32 = to_tflite(qat_f.export_float(), f"{OUT}/v2_qat_full_fp32.tflite")
    score(quantize(pf32, f"{OUT}/v2_qat_full.tflite", keep3=False), "QAT INT8 tat ca", {"train": info_f})

json.dump({"args": vars(args), "rows": rows}, open(f"{HERE}/results/qat_mlsd_v2.json", "w"), indent=1)
print("\nXong. Ket qua: results/qat_mlsd_v2.json")
