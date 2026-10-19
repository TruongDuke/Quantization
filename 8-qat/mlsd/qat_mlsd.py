"""QAT cho M-LSD tiny (bản tách head center/displacement) -> TFLite INT8.

Quy trình (PyTorch PT2E + torchao, xuất TFLite bằng litert-torch):
  1. Export model, chèn fake quantize (prepare_qat_pt2e).
  2. Calibration 100 ảnh để khởi tạo khoảng quantize  -> convert ngay = bản PTQ (mốc so sánh).
  3. Train tiếp bằng distillation: model FP32 làm thầy, không cần nhãn  -> convert = bản QAT.
  4. Đo cả 2 bản bằng metric của 7-mlsd-lowbit (center / displacement / % đoạn thẳng so với FP32).

Chạy: python qat_mlsd.py --steps 600 --bs 8
"""
import os
import sys
import time
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=4, help="cộng dồn gradient qua nhiều batch nhỏ (batch thật = bs * accum)")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--freeze_observer_at", type=float, default=0.5, help="tỉ lệ số bước thì ngừng cập nhật khoảng quantize")
ap.add_argument("--device", default="cpu")
ap.add_argument("--tag", default="")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
EXP1 = os.path.abspath(f"{HERE}/../../7-mlsd-lowbit/exp1_metrics_and_heads.py")

# ----- Dùng lại dữ liệu, model, metric của exp1 (dữ liệu đánh giá, REF = output PyTorch FP32)
g = {"__file__": EXP1, "__name__": "exp1"}
src = open(EXP1).read()
exec(src[:src.index("rows = []")], g)
preprocess, read_rgb, load_base, Split = g["preprocess"], g["read_rgb"], g["load_base"], g["Split"]
from_split, TFLiteRunner, benchmark, evaluate = g["from_split"], g["TFLiteRunner"], g["benchmark"], g["evaluate"]
calib_x, eval_x, train_paths, calib_paths = g["calib_x"], g["eval_x"], g["train_paths"], g["calib_paths"]

import litert_torch
from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
from litert_torch.quantize.quant_config import QuantConfig
import litert_torch.quantize.pt2e_quantizer_utils as qu
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torchao.quantization.pt2e import move_exported_model_to_eval, disable_observer

# ----- Vá lỗi không tương thích: litert-torch truyền hàm, torchao bản mới chỉ nhận nn.Module
_orig = qu._get_aten_graph_module_for_pattern
class _Wrap(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, *a, **k):
        return self.fn(*a, **k)
def _patched(pattern, example_inputs, is_cuda=False, **kw):
    if not isinstance(pattern, torch.nn.Module):
        pattern = _Wrap(pattern)
    return _orig(pattern, example_inputs, is_cuda, **kw)
qu._get_aten_graph_module_for_pattern = _patched

OUT = f"{HERE}/models"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{HERE}/results", exist_ok=True)
dev = torch.device(args.device)

# ----- Dữ liệu train: ảnh Imagenette train, không trùng 100 ảnh calibration (đánh giá dùng ảnh val)
calib_set = set(calib_paths)
pool = [p for p in train_paths if p not in calib_set]
rng = np.random.default_rng(2)
train_list = list(rng.choice(pool, 3000, replace=False))

def batch_from(paths):
    xs = []
    for p in paths:
        img = read_rgb(p)
        if rng.random() < 0.5:
            img = img[:, ::-1]                                                 # lật ngang
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)   # đổi độ sáng
        xs.append(preprocess(np.ascontiguousarray(img)))
    return torch.from_numpy(np.concatenate(xs))

# ----- Thầy (FP32) và trò (sẽ QAT)
teacher = load_base().eval().to(dev)
student = Split(load_base()).train()
q = PT2EQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
ex = torch.from_numpy(calib_x[:2])
gm = torch.export.export(student, (ex,), dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}}).module()
gm = prepare_qat_pt2e(gm, q)
move_exported_model_to_eval(gm)          # BatchNorm dùng thống kê có sẵn, không cập nhật (batch nhỏ)
gm = gm.to(dev)

def export_tflite(model, path):
    m = convert_pt2e(copy.deepcopy(model).to("cpu"), fold_quantize=False)
    litert_torch.convert(m, (torch.from_numpy(eval_x[0]),), quant_config=QuantConfig(pt2e_quantizer=q)).export(path)
    return path

def score(path, name):
    r = TFLiteRunner(path)
    maps = [from_split(r(x)) for x in eval_x]
    m = evaluate(lambda i: maps[i])
    m.update({"name": name, "size_mb": os.path.getsize(path) / 1e6, "ms": benchmark(path)})
    print(f"{name:12s} center {m['center_recall']:5.1f}  Dmed {m['disp_err_median_px']:.2f}  D<2px {m['disp_within_2px']:5.1f}  "
          f"giu {m['lines_kept']:5.1f}  doan {m['n_lines']}  | {m['size_mb']:.2f} MB {m['ms']:.1f} ms", flush=True)
    return m

# ----- 1. Calibration -> bản PTQ (cùng quy trình, chưa train)
with torch.no_grad():
    for i in range(0, len(calib_x), 8):
        gm(torch.from_numpy(calib_x[i:i + 8]).to(dev))
results = [score(export_tflite(gm, f"{OUT}/pt2e_ptq{args.tag}.tflite"), "PTQ (pt2e)")]

# ----- 2. QAT bằng distillation
def loss_fn(sc, sd, tc, td):
    t = torch.sigmoid(tc)
    w = 1.0 + 20.0 * t                                                  # center thật rất thưa: tăng trọng số chỗ có line
    l_center = (F.binary_cross_entropy_with_logits(sc, t, reduction="none") * w).mean()
    wd = t / (t.sum() + 1e-6)                                           # displacement chỉ quan trọng quanh center
    l_disp = (wd * (sd - td).abs().sum(1, keepdim=True)).sum()
    return l_center, l_disp

opt = torch.optim.Adam(gm.parameters(), lr=args.lr)
freeze_at = int(args.steps * args.freeze_observer_at)
t0, hist = time.time(), []
for step in range(args.steps):
    if step == freeze_at:
        gm.apply(disable_observer)                                      # chốt khoảng quantize, chỉ train weight
    opt.zero_grad()
    lc_sum = ld_sum = 0.0
    for a in range(args.accum):                                         # batch nhỏ để tiết kiệm bộ nhớ, cộng dồn gradient
        k0 = (step * args.accum + a) * args.bs
        x = batch_from([train_list[(k0 + k) % len(train_list)] for k in range(args.bs)]).to(dev)
        with torch.no_grad():
            out = teacher(x)
            tc, td = out[:, 0:1], out[:, 1:5]
        sc, sd = gm(x)
        l_c, l_d = loss_fn(sc, sd, tc, td)
        ((l_c + 0.05 * l_d) / args.accum).backward()
        lc_sum += float(l_c) / args.accum
        ld_sum += float(l_d) / args.accum
    opt.step()
    hist.append([lc_sum, ld_sum])
    if step % 25 == 0 or step == args.steps - 1:
        print(f"step {step:4d}  center {np.mean([h[0] for h in hist[-25:]]):.4f}  disp {np.mean([h[1] for h in hist[-25:]]):.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

results.append(score(export_tflite(gm, f"{OUT}/pt2e_qat{args.tag}.tflite"), "QAT"))
train_info = {"steps": args.steps, "bs": args.bs, "accum": args.accum, "lr": args.lr, "freeze_observer_at": freeze_at,
              "n_train_images": len(train_list), "seconds": time.time() - t0, "device": args.device, "loss_history": hist}
json.dump({"results": results, "train": train_info}, open(f"{HERE}/results/qat_mlsd{args.tag}.json", "w"), indent=1)
